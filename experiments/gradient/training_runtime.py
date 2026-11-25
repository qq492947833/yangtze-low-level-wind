"""Bounded prediction storage and complete-epoch recovery for this gradient run.

All mutable files are confined to run_root/hour_H/stageN. This module does not
alter precision, batch size, model architecture, loss, optimizer or split rules.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
import math
import os
import random
import time
import uuid

import numpy as np
import torch

BJ = timezone(timedelta(hours=8))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def stamp():
    return datetime.now(BJ).isoformat(timespec='seconds')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


class GradientTrainingRuntime:
    def __init__(self, run_root, hour, stage, config, adapter_path, *, allow_resume=False,
                 chunk_rows=128, resource_callback=None):
        self.run_root = Path(run_root).resolve()
        if isinstance(hour, bool) or int(hour) not in range(1, 7):
            raise ValueError('hour must be 1..6')
        if stage not in (1, 2):
            raise ValueError('stage must be 1 or 2')
        self.hour, self.stage = int(hour), int(stage)
        self.directory = self.run_root / f'hour_{self.hour}' / f'stage{self.stage}'
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.directory.resolve().is_relative_to(self.run_root):
            raise ValueError('runtime output escaped authorized run root')
        self.adapter_path = Path(adapter_path).resolve()
        if self.adapter_path != Path(__file__).with_name('network_adapter.py').resolve():
            raise ValueError('expected network_adapter.py beside the training module')
        self.config = json.loads(canonical(config))
        self.identity = dict(schema_version=1, run_root=str(self.run_root), hour=self.hour,
                             stage=self.stage, config=self.config,
                             adapter_sha256=sha256(self.adapter_path),
                             runtime_sha256=sha256(__file__))
        self.fingerprint = hashlib.sha256(canonical(self.identity).encode('utf-8')).hexdigest()
        self.allow_resume = bool(allow_resume)
        if int(chunk_rows) < 1:
            raise ValueError('chunk_rows must be positive')
        self.chunk_rows = int(chunk_rows)
        self.resource_callback = resource_callback
        self.completed_epochs, self.best_epoch = 0, 0
        self._max_epochs = None
        self._started_monotonic = None
        self._prediction_paths = []
        # Two slots preserve the previous committed generation while the next
        # checkpoint and its atomic pointer receipt are being published.
        self.checkpoint_path = self.directory / 'complete_epoch_state_a.pt'
        self.checkpoint_receipt = self.directory / 'complete_epoch_state.json'

    def _safe_path(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError(f'write outside current stage refused: {path}')
        return path

    def _publish(self, temporary, destination):
        temporary, destination = self._safe_path(temporary), self._safe_path(destination)
        for attempt in range(6):
            try:
                os.replace(temporary, destination)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.1 * (attempt + 1))

    def _json(self, path, value):
        path = self._safe_path(path)
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        payload = (canonical(value) + '\n').encode('utf-8')
        with temporary.open('xb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self._publish(temporary, path)
        if path.read_bytes() != payload:
            raise IOError(f'JSON readback mismatch: {path}')

    def _event(self, kind, **payload):
        event = dict(time_beijing=stamp(), kind=kind, hour=self.hour, stage=self.stage, **payload)
        with (self.directory / 'runtime_events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(canonical(event) + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    def allocate_prediction(self, shape, dtype=np.float64):
        shape = tuple(int(x) for x in shape)
        if len(shape) != 4 or any(x <= 0 for x in shape):
            raise ValueError('invalid prediction shape')
        dtype = np.dtype(dtype)
        if dtype != np.dtype('float64'):
            raise ValueError('accepted function prediction return dtype must remain float64')
        path = self.directory / ('prediction_' + uuid.uuid4().hex + '.mmap')
        size = math.prod(shape) * dtype.itemsize
        # Creating a new file reserves disk space without constructing or touching
        # a full RAM array. Each element is written once by the original loop.
        result = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
        self._prediction_paths.append(str(path))
        self._event('prediction_memmap_allocated', path=str(path), shape=shape,
                    dtype=str(dtype), bytes=size)
        return result

    def sanitize_prediction(self, array):
        if not isinstance(array, np.memmap) or np.dtype(array.dtype) != np.dtype('float64'):
            raise TypeError('prediction must use this run float64 memmap')
        self._safe_path(array.filename)
        replaced_nan = replaced_posinf = replaced_neginf = 0
        for start in range(0, len(array), self.chunk_rows):
            block = array[start:start+self.chunk_rows]
            replaced_nan += int(np.count_nonzero(np.isnan(block)))
            replaced_posinf += int(np.count_nonzero(np.isposinf(block)))
            replaced_neginf += int(np.count_nonzero(np.isneginf(block)))
            # Exact original nan_to_num semantics, with a bounded in-place block.
            np.nan_to_num(block, copy=False, nan=0)
        array.flush()
        self._event('prediction_sanitized', path=str(array.filename),
                    nan_count=replaced_nan, posinf_count=replaced_posinf,
                    neginf_count=replaced_neginf, unchanged_finite_values=True)
        if replaced_nan or replaced_posinf or replaced_neginf:
            # Original transformation is preserved, but scientifically invalid
            # outputs may not silently pass artifact acceptance.
            raise FloatingPointError('Nonfinite predictions were observed; counts saved, refuse acceptance')
        return array

    @staticmethod
    def _model_signature(model):
        return {name: dict(shape=list(value.shape), dtype=str(value.dtype))
                for name, value in model.state_dict().items()}

    @staticmethod
    def _precision():
        return dict(torch_version=str(torch.__version__), cuda_version=str(torch.version.cuda),
                    matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                    cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                    cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                    cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                    float32_matmul_precision=torch.get_float32_matmul_precision(),
                    amp_enabled=False, input_tensor_dtype='torch.float32')

    def training_start(self, model, opt, early_stopping, epochs, model_device):
        self.device = next(model.parameters()).device
        requested = torch.device(model_device)
        if self.device.type != requested.type or (requested.index is not None and self.device.index != requested.index) or any(p.device != self.device for p in model.parameters()):
            raise RuntimeError('All model parameters must reside on the configured training device')
        self._max_epochs = int(epochs)
        self._started_monotonic = time.monotonic()
        self.signature = self._model_signature(model)
        self.precision = self._precision()
        slots = [self.directory / f'complete_epoch_state_{letter}.pt' for letter in ('a', 'b')]
        if any(path.exists() for path in slots) or self.checkpoint_receipt.exists():
            if not self.allow_resume:
                raise RuntimeError('Existing complete-epoch state requires explicit allow_resume=True')
            if not self.checkpoint_receipt.exists():
                raise RuntimeError('Partial checkpoint publication: refuse to claim lossless resume')
            receipt = json.loads(self.checkpoint_receipt.read_text('utf-8'))
            self.checkpoint_path = self._safe_path(receipt['path'])
            if self.checkpoint_path not in slots or not self.checkpoint_path.exists():
                raise RuntimeError('Checkpoint pointer does not identify an available authorized slot')
            if receipt['fingerprint'] != self.fingerprint or receipt['sha256'] != sha256(self.checkpoint_path):
                raise RuntimeError('Checkpoint identity or SHA256 differs; refuse resume')
            payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            required = {'identity','fingerprint','model_signature','precision','model','optimizer','early_stopping',
                        'epoch','best_epoch','rng','max_epochs','complete_epoch','terminal','epoch_metrics'}
            if not required.issubset(payload) or payload['complete_epoch'] is not True:
                raise RuntimeError('Insufficient full-state checkpoint')
            if (payload['fingerprint'] != self.fingerprint or payload['identity'] != self.identity
                    or payload['model_signature'] != self.signature or payload['precision'] != self.precision
                    or payload['max_epochs'] != self._max_epochs):
                raise RuntimeError('Task/configuration/shape/precision changed; refuse resume')
            model.load_state_dict(payload['model'], strict=True)
            opt.load_state_dict(payload['optimizer'])
            for key, value in payload['early_stopping'].items():
                setattr(early_stopping, key, value)
            self.completed_epochs, self.best_epoch = int(payload['epoch']), int(payload['best_epoch'])
            if self.completed_epochs < 1 or self.completed_epochs > self._max_epochs:
                raise RuntimeError('Checkpoint epoch outside authorized range')
            metrics_path = self.directory / 'epoch_metrics.csv'
            if metrics_path.exists():
                with metrics_path.open(encoding='utf-8', newline='') as handle:
                    metrics_rows = list(csv.DictReader(handle))
                logged_epochs = [int(row['epoch']) for row in metrics_rows]
            else:
                logged_epochs = []
            expected = list(range(1, self.completed_epochs + 1))
            if logged_epochs == expected[:-1]:
                # A crash may occur after the checkpoint pointer is committed
                # and before its CSV row is appended. Recover the real row
                # saved in the snapshot; never fabricate a loss or epoch.
                row = payload['epoch_metrics']
                with metrics_path.open('a', encoding='utf-8', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    if not logged_epochs:
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                    os.fsync(handle.fileno())
                self._event('committed_epoch_log_recovered', epoch=self.completed_epochs)
            elif logged_epochs != expected:
                raise RuntimeError('Epoch log does not match committed snapshot history')
            rng = payload['rng']
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            # CPU RNG bytes are metadata, not CPU model training. map_location is
            # the selected device; RNG APIs expect CPU byte tensors.
            torch.set_rng_state(rng['torch_cpu'].cpu())
            if self.device.type == 'cuda':
                saved_cuda_rng = rng.get('torch_cuda', rng.get('torch_cuda0'))
                torch.cuda.set_rng_state(saved_cuda_rng.cpu(), device=self.device)
            terminal = bool(payload['terminal'])
            self._event('complete_epoch_state_restored', epoch=self.completed_epochs,
                        best_epoch=self.best_epoch, terminal=terminal,
                        checkpoint_sha256=receipt['sha256'])
            return self._max_epochs if terminal else self.completed_epochs
        self._json(self.directory / 'training_identity.json', dict(identity=self.identity,
                    fingerprint=self.fingerprint, model_signature=self.signature,
                    precision=self.precision, max_epochs=self._max_epochs, created_beijing=stamp()))
        self._event('training_started', seed=self.config.get('seed', 25), precision=self.precision)
        return 0

    def epoch_completed(self, epoch, model, opt, early_stopping, train_loss, valid_loss,
                        train_metric, valid_metric):
        epoch = int(epoch)
        if epoch != self.completed_epochs + 1:
            raise RuntimeError('Epoch sequence is not contiguous')
        if not np.isfinite(train_loss) or not np.isfinite(valid_loss):
            self._event('nonfinite_loss', epoch=epoch,
                        train_loss=None if not np.isfinite(train_loss) else train_loss,
                        valid_loss=None if not np.isfinite(valid_loss) else valid_loss)
            raise FloatingPointError('Nonfinite training/validation loss; last valid complete epoch retained')
        if early_stopping.counter == 0:
            self.best_epoch = epoch
        early = {key: getattr(early_stopping, key) for key in
                 ('patience','counter','best_score','early_stop','val_loss_min','delta','best_model_state_dict')}
        row = dict(epoch=epoch, train_loss=float(train_loss), valid_loss=float(valid_loss),
                   train_metric=None if not np.isfinite(train_metric) else float(train_metric),
                   valid_metric=None if not np.isfinite(valid_metric) else float(valid_metric),
                   best_epoch=self.best_epoch, early_stop_counter=int(early_stopping.counter),
                   early_stop=bool(early_stopping.early_stop), time_beijing=stamp())
        payload = dict(identity=self.identity, fingerprint=self.fingerprint,
                       model_signature=self.signature, precision=self.precision,
                       model=model.state_dict(), optimizer=opt.state_dict(), early_stopping=early,
                       epoch=epoch, best_epoch=self.best_epoch, max_epochs=self._max_epochs, epoch_metrics=row,
                       complete_epoch=True, terminal=bool(early_stopping.early_stop or epoch >= self._max_epochs),
                       rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                torch_cpu=torch.get_rng_state(),
                                torch_cuda=(torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None)))
        next_checkpoint = self.directory / ('complete_epoch_state_a.pt' if epoch % 2 else 'complete_epoch_state_b.pt')
        temporary = next_checkpoint.with_name(next_checkpoint.name + '.' + uuid.uuid4().hex + '.tmp')
        with temporary.open('xb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = sha256(temporary)
        self._publish(temporary, next_checkpoint)
        if sha256(next_checkpoint) != digest:
            raise IOError('Complete-epoch checkpoint readback SHA256 differs')
        self._json(self.checkpoint_receipt, dict(schema_version=1, fingerprint=self.fingerprint,
                   path=str(next_checkpoint), sha256=digest, bytes=next_checkpoint.stat().st_size,
                   epoch=epoch, best_epoch=self.best_epoch, complete_epoch=True,
                   terminal=payload['terminal'], committed_beijing=stamp()))
        self.checkpoint_path = next_checkpoint
        self.completed_epochs = epoch
        path = self.directory / 'epoch_metrics.csv'
        first = not path.exists()
        with path.open('a', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if first:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        self._json(self.directory / 'training_progress.json', dict(status='TRAINING', **row,
                   checkpoint_sha256=digest, checkpoint_complete_epoch=True))
        if self.resource_callback is not None:
            self.resource_callback(dict(hour=self.hour, stage=self.stage, **row))

    def training_finished(self, model, opt, early_stopping, completed_epochs):
        if int(completed_epochs) != self.completed_epochs or self.completed_epochs < 1:
            raise RuntimeError('Training completion epoch evidence missing')
        if not early_stopping.early_stop and self.completed_epochs != self._max_epochs:
            raise RuntimeError('Training did not reach early stopping or maximum epochs')
        best = early_stopping.best_model_state_dict
        if best is None or self.best_epoch < 1:
            raise RuntimeError('No best validation weights recorded')
        state = model.state_dict()
        if state.keys() != best.keys() or any(not torch.equal(value, best[name]) for name, value in state.items()):
            raise RuntimeError('Actual restored weights do not equal recorded best validation weights')
        digest = hashlib.sha256()
        for name in sorted(state):
            array = state[name].detach().cpu().contiguous().numpy()
            digest.update(name.encode('utf-8'))
            digest.update(str(array.dtype).encode('ascii'))
            digest.update(str(array.shape).encode('ascii'))
            digest.update(array.tobytes())
        self._json(self.directory / 'training_finished.json', dict(
            status='TRAINING_FINISHED_PENDING_TEST_EXPORT_AND_ACCEPTANCE', fingerprint=self.fingerprint,
            completed_epochs=self.completed_epochs, best_epoch=self.best_epoch,
            best_validation_loss=float(early_stopping.val_loss_min),
            actual_best_weights_restored_and_equal=True, best_weight_tensor_sha256=digest.hexdigest(),
            final_saved_model_source='best_validation_epoch',
            final_saved_optimizer_source='terminal_epoch_optimizer_from_original_function',
            optimizer_epoch=self.completed_epochs,
            complete_resume_checkpoint=str(self.checkpoint_path),
            complete_resume_checkpoint_sha256=sha256(self.checkpoint_path),
            elapsed_this_process_seconds=time.monotonic()-self._started_monotonic,
            finished_beijing=stamp(), precision=self.precision))
        self._event('best_weights_restored', completed_epochs=self.completed_epochs,
                    best_epoch=self.best_epoch, weights_sha256=digest.hexdigest())
