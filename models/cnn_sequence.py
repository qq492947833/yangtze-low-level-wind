def Auto_DeepLearning_pytorch(vy,vx,model_list,lat=None,lon=None,a_mode='d',r_number=None,u=None,v=None,timestep=None,outputstep=None,dilations=None,ifmulti_scale='no',num_channels=None,test_size=0.2,valid_size=0.1,k_fold=None,task_mode='regression',if_best_mode='no',modelpath=None,ifrandom_split='yes',cov_padding='same',cov_strides=1,pooling_strides=2,dropout=0.0,residual_deep=1,residual_activation='relu',residual_norm='batchnormalization',vit_patch_size=None,vit_dim_feedforward=16,vit_deep=1,vit_num_heads=1,swin_deep=[2,2],swin_num_heads=[1,1],cbam_reduction=16,chebnet_k=3,gsage_aggr='mean',gsage_lstm_units=None,transformer_encoder_deep=1,transformer_num_heads=2,transformer_dim_feedforward=256,tcn_kernel_size=2,nlp_units=64,nlp_num_layers=1,ltc_solver='fused',ltc_ode_unfolds=4,ltc_dt=1.0,ltc_tau_min=1e-4,ltc_activation='softplus',ltc_dropout=None,mamba_state=64,mamba_conv=4,mamba_expand=2,activation='relu',if_print_model='yes',loss_function='default',optimizer='SGD',metrics='default',learning_rate=0.01,lr_decay=None,regularization=None,regularization_lambda=1e-5,epochs=2000,batch_size=20,if_early_stopping=None,ifheatmap='yes',ifweight='yes',ifmute='no',ifsave='no',savepath=None,device='cpu'):
    from torch.nn import Module,BatchNorm1d,BatchNorm2d,LayerNorm,Conv2d,MaxPool2d,AvgPool2d,RNN,LSTM,GRU,Embedding,TransformerEncoderLayer,TransformerEncoder,Dropout,LeakyReLU,ReLU,PReLU,Sigmoid,Tanh,ELU,Softmax,Linear,Flatten,ZeroPad2d
    from torch.optim import Adam,SGD,NAdam
    from torch.optim.lr_scheduler import ExponentialLR
    import torch
    from torch import nn
    import torch.nn.functional as F
    import torchmetrics
    layer_types = {layer[0] for layer in model_list}
    if 'tcn' in layer_types:
        from pytorch_tcn import TCN
    if 'vit' in layer_types:
        from vit_pytorch import ViT
    if layer_types.intersection({'mamba', 'mamba2'}):
        import mamba_ssm.modules.mamba2 as mamba2_module
        mamba2_module.causal_conv1d_fn = None
        mamba2_module.causal_conv1d_update = None
        import mamba_ssm.modules.mamba_simple as mamba_module
        mamba_module.causal_conv1d_fn = None
        mamba_module.causal_conv1d_update = None
        from mamba_ssm import Mamba,Mamba2
    if 'swin' in layer_types:
        from timm.models.swin_transformer import SwinTransformer
    import math
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    import numpy as np
    from scipy.stats import pearsonr
    import os
    from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
    import sklearn
    import copy
    import shap
    import datetime
    import warnings
    from tqdm import tqdm
    from geopy.distance import geodesic
    warnings.filterwarnings('ignore')

    if lr_decay is not None:
        lr_decay = float(lr_decay)
        if lr_decay <= 0:
            raise ValueError("lr_decay 必须为 None 或正浮点数。例如 None、0.99、0.95。")

    if regularization is not None:
        regularization = str(regularization).lower()
        if regularization not in ['l1', 'l2']:
            raise ValueError("regularization 只能为 None、'l1' 或 'l2'。")
        if regularization_lambda is None:
            regularization_lambda = 1e-5
        regularization_lambda = float(regularization_lambda)
        if regularization_lambda < 0:
            raise ValueError("regularization_lambda 必须 >= 0。")

    def add_regularization_loss(base_loss, model_instance):
        if regularization is None or regularization_lambda == 0:
            return base_loss

        reg_loss = torch.zeros((), dtype=base_loss.dtype, device=base_loss.device)

        for name, param in model_instance.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim <= 1 or name.endswith('.bias'):
                continue

            if regularization == 'l1':
                reg_loss = reg_loss + torch.sum(torch.abs(param))
            elif regularization == 'l2':
                reg_loss = reg_loss + torch.sum(param ** 2)

        return base_loss + regularization_lambda * reg_loss


    vx_original_ndim = vx.ndim

    devices = torch.device(os.environ.get('YANGTZE_DEVICE', 'cuda') if device == 'gpu' else device)
    if task_mode=='binary_classify' or task_mode=='multi_classify':
        def get_class_weights_from_labels(vy, device=None, ignore_index=None, num_classes=None):
            y = torch.as_tensor(vy, device=device)

            if y.ndim >= 1 and y.shape[-1] == 1:
                y = y.squeeze(-1)

            y = y.reshape(-1)

            if ignore_index is not None:
                y = y[y != ignore_index]

            if y.is_floating_point():
                y = y[~torch.isnan(y)]

            y = y.long()

            if num_classes is None:
                num_classes = int(y.max().item()) + 1

            cnt = torch.bincount(y, minlength=num_classes).float()
            w = cnt.sum() / (num_classes * cnt.clamp_min(1.0))  # N/(C*count_c)
            return w  # shape: (C,)
        class_weights = get_class_weights_from_labels(vy, device=devices)
    if task_mode=='regression':
        if type(loss_function) is not str:
            loss=loss_function
        else:
            if loss_function=='default' or loss_function=='MSELoss':
                loss=torch.nn.MSELoss()
            elif loss_function=='L1Loss':
                loss=torch.nn.L1Loss
            elif loss_function=='PoissonNLLLoss':
                loss=torch.nn.PoissonNLLLoss()
            elif loss_function=='GaussianNLLLoss':
                loss=torch.nn.GaussianNLLLoss()
            elif loss_function=='KLDivLoss':
                loss=torch.nn.KLDivLoss()
            elif loss_function=='HuberLoss':
                loss=torch.nn.HuberLoss()
            elif loss_function=='SmoothL1Loss':
                loss=torch.nn.SmoothL1Loss()
            elif loss_function=='Pearsonr':
                class loss_pearsonr(nn.Module):
                    def __init__(self):
                        super().__init__()

                    def forward(self, y, x):
                        y_true_mean=torch.nanmean(y,dim=0,keepdim=True)
                        y_pred_mean=torch.nanmean(x,dim=0,keepdim=True)
                        cov=torch.nansum((y-y_true_mean)*(x-y_pred_mean),dim=0,keepdim=True)
                        y_true_v=torch.nansum(torch.square((y-y_true_mean)),dim=0,keepdim=True)
                        y_pred_v=torch.nansum(torch.square((x-y_pred_mean)),dim=0,keepdim=True)
                        y_true_v=torch.sqrt(y_true_v)
                        y_pred_v=torch.sqrt(y_pred_v)
                        pearson=cov/(y_true_v*y_pred_v)
                        return (1-pearson)**1.5
                loss=loss_pearsonr()
        if type(metrics) is not str:
            metric=metrics
        else:
            if metrics=='default' or metrics=='MSELoss':
                metric=torch.nn.MSELoss()
            elif metrics=='L1Loss':
                metric=torch.nn.L1Loss
            elif metrics=='PoissonNLLLoss':
                metric=torch.nn.PoissonNLLLoss()
            elif metrics=='GaussianNLLLoss':
                metric=torch.nn.GaussianNLLLoss()
            elif metrics=='KLDivLoss':
                metric=torch.nn.KLDivLoss()
            elif metrics=='HuberLoss':
                metric=torch.nn.HuberLoss()
            elif metrics=='SmoothL1Loss':
                metric=torch.nn.SmoothL1Loss()
            elif metrics=='Pearsonr':
                class metric_pearsonr(nn.Module):
                    def __init__(self):
                        super().__init__()

                    def forward(self, y, x):
                        y_true_mean=torch.nanmean(y,dim=0,keepdim=True)
                        y_pred_mean=torch.nanmean(x,dim=0,keepdim=True)
                        cov=torch.nansum((y-y_true_mean)*(x-y_pred_mean),dim=0,keepdim=True)
                        y_true_v=torch.nansum(torch.square((y-y_true_mean)),dim=0,keepdim=True)
                        y_pred_v=torch.nansum(torch.square((x-y_pred_mean)),dim=0,keepdim=True)
                        y_true_v=torch.sqrt(y_true_v)
                        y_pred_v=torch.sqrt(y_pred_v)
                        pearson=cov/(y_true_v*y_pred_v)
                        return (1-pearson)**1.5
                metric=metric_pearsonr()
    elif task_mode=='binary_classify':
        if type(loss_function) is not str:
            loss=loss_function
        else:
            if loss_function=='default' or loss_function=='BCELoss':
                w0, w1 = class_weights[0].to(devices), class_weights[1].to(devices)

                class WeightedBCELoss(nn.Module):
                    def __init__(self, w0, w1, eps=1e-7):
                        super().__init__()
                        self.register_buffer("w0", w0.detach().clone().float())
                        self.register_buffer("w1", w1.detach().clone().float())
                        self.eps = eps

                    def forward(self, y_pred, y_true):
                        y_true = y_true.float()
                        y_pred = y_pred.float().clamp(self.eps, 1.0 - self.eps)

                        weight_map = torch.where(
                            y_true > 0.5,
                            self.w1,
                            self.w0
                        )

                        return torch.nn.functional.binary_cross_entropy(
                            y_pred,
                            y_true,
                            weight=weight_map,
                            reduction="mean"
                        )

                loss = WeightedBCELoss(w0, w1).to(devices)
            elif loss_function=='BCEWithLogitsLoss':
                pos_w = (class_weights[1] / class_weights[0]).to(devices).view(1)
                loss=torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
            elif loss_function=='SoftMarginLoss':
                loss=torch.nn.SoftMarginLoss()
            elif loss_function=='MultiLabelSoftMarginLoss':
                loss=torch.nn.MultiLabelSoftMarginLoss()
        if type(metrics) is not str:
            metric=metrics
        else:
            if metrics=='default' or metrics=='f1':
                metric=torchmetrics.F1Score(task="binary").to(devices)
            elif metrics=='accuracy':
                metric=torchmetrics.Accuracy(task="binary").to(devices)
            elif metrics=='precision':
                metric=torchmetrics.Precision(task="binary").to(devices)
            elif metrics=='recall':
                metric=torchmetrics.Recall(task="binary").to(devices)
            elif metrics=='BCELoss':
                metric=torch.nn.BCELoss()
            elif metrics=='BCEWithLogitsLoss':
                metric=torch.nn.BCEWithLogitsLoss()
            elif metrics=='SoftMarginLoss':
                metric=torch.nn.SoftMarginLoss()
            elif metrics=='MultiLabelSoftMarginLoss':
                metric=torch.nn.MultiLabelSoftMarginLoss()
    elif task_mode=='multi_classify':
        if type(loss_function) is not str:
            loss=loss_function
        else:
            if loss_function=='default' or loss_function=='CrossEntropyLoss':
                loss=torch.nn.CrossEntropyLoss(weight=class_weights)
            elif loss_function=='NLLLoss':
                loss=torch.nn.NLLLoss(weight=class_weights)
            elif loss_function=='TripletMarginLoss':
                loss=torch.nn.TripletMarginLoss()
            elif loss_function=='KLDivergence':
                loss=torch.nn.KLDivergence()
            elif loss_function=='HingeEmbeddingLoss':
                loss=torch.nn.HingeEmbeddingLoss()
            elif loss_function=='MultiLabelMarginLoss':
                loss=torch.nn.MultiLabelMarginLoss()
            elif loss_function=='TripletMarginWithDistanceLoss':
                loss=torch.nn.TripletMarginWithDistanceLoss()
        if type(metrics) is not str:
            metric=metrics
        else:
            if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                metric=torchmetrics.Accuracy(task="multiclass",num_classes = int(np.max(vy))+1, average="macro").to(devices)
            elif metrics=='recall' :
                metric=torchmetrics.Recall(task="multiclass", num_classes=int(np.max(vy))+1, average="macro").to(devices)
            elif metrics=='precision' :
                metric=torchmetrics.Precision(task="multiclass", num_classes=int(np.max(vy))+1, average="macro").to(devices)
            elif metrics=='f1' :
                metric=torchmetrics.F1Score(task="multiclass", num_classes=int(np.max(vy))+1, average="macro").to(devices)
            elif metrics=='CrossEntropyLoss':
                metric=torch.nn.CrossEntropyLoss()
            elif metrics=='NLLLoss':
                metric=torch.nn.NLLLoss()
            elif metrics=='TripletMarginLoss':
                metric=torch.nn.TripletMarginLoss()
            elif metrics=='KLDivergence':
                metric=torch.nn.KLDivergence()
            elif metrics=='HingeEmbeddingLoss':
                metric=torch.nn.HingeEmbeddingLoss()
            elif metrics=='MultiLabelMarginLoss':
                metric=torch.nn.MultiLabelMarginLoss()
            elif metrics=='TripletMarginWithDistanceLoss':
                metric=torch.nn.TripletMarginWithDistanceLoss()
    heatmap=0
    weights=0
    model=0
    predicty=0
    testy=0
    r=0
    p=0

    class EarlyStopping:
        def __init__(self, patience, delta=0):
            self.patience = patience
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = np.Inf
            self.delta = delta
            self.best_model_state_dict = None

        def __call__(self, val_loss, model):
            if np.isnan(val_loss):
                print("Validation loss is NaN. Stopping early and will restore best model.")
                self.early_stop = True
                return

            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        def save_checkpoint(self, val_loss, model):
            self.best_model_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            self.val_loss_min = val_loss

        def load_best_checkpoint(self, model):
            if self.best_model_state_dict is not None:
                model.load_state_dict(self.best_model_state_dict)
            else:
                print("Warning: load_best_checkpoint called but no best_model_state_dict was saved.")
    class ShapWrapperCombinedInput(nn.Module):
        def __init__(self, original_model, x_sample_shape, pos_sample_shape):
            super().__init__()
            self.original_model = original_model
            self.x_sample_shape = x_sample_shape
            self.pos_sample_shape = pos_sample_shape
            self.x_flat_size = np.prod(x_sample_shape)
            self.pos_flat_size = np.prod(pos_sample_shape)
            self.total_flat_size = self.x_flat_size + self.pos_flat_size

        def forward(self, combined_input_float):
            if combined_input_float.dim() == 1:
                combined_input_float = combined_input_float.unsqueeze(0)
            x_flat = combined_input_float[:, :self.x_flat_size]
            pos_flat = combined_input_float[:, self.x_flat_size:]
            x_reshaped = x_flat.reshape(-1, *self.x_sample_shape)
            position_indices_int = pos_flat.long()
            model_output = self.original_model(x_reshaped, position_indices_int)
            if model_output.dim() > 2:
                model_output = model_output.reshape(model_output.shape[0], -1)
            return model_output


    vy_original_ndim = int(vy.ndim)
    vy_restore_mode = None
    vy_original_target_shape = None
    vy_channel_first_target_shape = None

    if vy.ndim == 1:
        vy = vy.reshape(vy.shape[0], 1)
    elif vy.ndim == 2:
        pass
    elif vy.ndim == 3:
        vy_restore_mode = 'station_feature'
        station_number_y = int(vy.shape[1])
        feature_number_y = int(vy.shape[2])
        vy_original_target_shape = (station_number_y, feature_number_y)
        vy_channel_first_target_shape = (feature_number_y, station_number_y)
        vy = vy.transpose(0, 2, 1).reshape(vy.shape[0], -1)
    elif vy.ndim == 4:
        vy_restore_mode = 'grid_feature'
        lat_number_y = int(vy.shape[1])
        lon_number_y = int(vy.shape[2])
        feature_number_y = int(vy.shape[3])
        vy_original_target_shape = (lat_number_y, lon_number_y, feature_number_y)
        vy_channel_first_target_shape = (feature_number_y, lat_number_y, lon_number_y)
        vy = vy.transpose(0, 3, 1, 2).reshape(vy.shape[0], -1)
    else:
        raise ValueError(
            'vy 目前仅支持 1D、2D、3D 或 4D。分别对应 '
            '(time,)、(time,feature_y)、(time,station,feature_y)、'
            '(time,lat,lon,feature_y)。'
        )

    vy_flat_size = int(vy.shape[1])

    def _adl_restore_y_array(y_array):
        """Restore internally flattened structured labels to feature-last layout."""
        y_array = np.asarray(y_array)
        if vy_restore_mode is None:
            return y_array

        if outputstep is None:
            if y_array.ndim != 2 or y_array.shape[1] != vy_flat_size:
                raise ValueError(
                    f'无法恢复标签形状：期望二维 (sample,{vy_flat_size})，实际为 {y_array.shape}。'
                )
            if vy_restore_mode == 'station_feature':
                feature_y, station_y = vy_channel_first_target_shape
                return y_array.reshape(y_array.shape[0], feature_y, station_y).transpose(0, 2, 1)
            feature_y, lat_y, lon_y = vy_channel_first_target_shape
            return y_array.reshape(y_array.shape[0], feature_y, lat_y, lon_y).transpose(0, 2, 3, 1)

        if y_array.ndim != 3 or y_array.shape[2] != vy_flat_size:
            raise ValueError(
                f'无法恢复多步标签形状：期望三维 (sample,outputstep,{vy_flat_size})，实际为 {y_array.shape}。'
            )
        if vy_restore_mode == 'station_feature':
            feature_y, station_y = vy_channel_first_target_shape
            return y_array.reshape(
                y_array.shape[0], y_array.shape[1], feature_y, station_y
            ).transpose(0, 1, 3, 2)
        feature_y, lat_y, lon_y = vy_channel_first_target_shape
        return y_array.reshape(
            y_array.shape[0], y_array.shape[1], feature_y, lat_y, lon_y
        ).transpose(0, 1, 3, 4, 2)

    def _adl_find_string_in_nested_list(target_string, nested_list):
        for item in nested_list:
            if isinstance(item, list):
                if _adl_find_string_in_nested_list(target_string, item):
                    return True
            elif isinstance(item, str):
                if item == target_string:
                    return True
        return False

    _adl_seq_layer_names = ['transformer', 'lstm', 'rnn', 'gru', 'ltc', 'tcn', 'mamba', 'mamba2']
    _adl_has_nlp_for_seq2seq = any(_adl_find_string_in_nested_list(layer_name, model_list) for layer_name in _adl_seq_layer_names)
    _adl_has_graph_for_window = any(_adl_find_string_in_nested_list(layer_name, model_list) for layer_name in ['gc', 'ga', 'chebnet', 'gsage'])

    if outputstep is not None:
        if not isinstance(outputstep, int) or outputstep <= 0:
            raise ValueError('outputstep 必须为 None 或正整数。')
        if timestep is None:
            raise ValueError('outputstep 不为 None 时，timestep 不能为 None。')
        if not _adl_has_nlp_for_seq2seq:
            raise ValueError('outputstep 不为 None 时，model_list 中必须至少包含一个 NLP/序列模块：transformer、lstm、rnn、gru、ltc、tcn、mamba 或 mamba2。')
        if vx.shape[0] != vy.shape[0]:
            raise ValueError('outputstep 不为 None 时，vx 和 vy 的第 0 维时间长度必须一致。')

        sample_number = vx.shape[0] - timestep - outputstep + 1
        if sample_number <= 0:
            raise ValueError('样本长度不足：vx.shape[0] 必须 >= timestep + outputstep。')

        if vx.ndim == 2:
            vxs = np.zeros((sample_number, timestep, vx.shape[1]), dtype=vx.dtype)
        elif vx.ndim == 3:
            vxs = np.zeros((sample_number, timestep, vx.shape[2], vx.shape[1]), dtype=vx.dtype)
        elif vx.ndim == 4:
            vxs = np.zeros((sample_number, timestep, vx.shape[1], vx.shape[2], vx.shape[3]), dtype=vx.dtype)
        else:
            raise ValueError('outputstep 不为 None 时，目前支持 vx.ndim==2、vx.ndim==3 或 vx.ndim==4。')

        vys = np.zeros((sample_number, outputstep, vy_flat_size), dtype=vy.dtype)
        for i in range(sample_number):
            if vx.ndim == 3:
                vxs[i,] = vx[i:i+timestep,].transpose(0, 2, 1)
            else:
                vxs[i,] = vx[i:i+timestep,]
            vys[i,] = vy[i+timestep:i+timestep+outputstep,]

        if vx.ndim == 4:
            vxs = vxs.transpose(0, 1, 4, 2, 3)

    else:
        if vx.ndim == 2 and timestep is not None:
            vxs = np.zeros((vx.shape[0]-timestep+1, timestep, vx.shape[1]), dtype=vx.dtype)
        elif vx.ndim == 3 and timestep is not None:
            vxs = np.zeros((vx.shape[0]-timestep+1, timestep, vx.shape[2], vx.shape[1]), dtype=vx.dtype)
        elif vx.ndim == 4 and timestep is not None:
            vxs = np.zeros((vx.shape[0]-timestep+1, timestep, vx.shape[1], vx.shape[2], vx.shape[3]), dtype=vx.dtype)
        elif timestep is None:
            if vx.ndim == 4:
                vxs = vx.transpose(0, 3, 1, 2)
            elif vx.ndim == 3:
                vxs = vx.transpose(0, 2, 1)
            else:
                vxs = vx
            vys = vy

        if timestep is not None:
            for i in range(vx.shape[0]-timestep+1):
                if vx.ndim == 3:
                    vxs[i,] = vx[i:i+timestep,].transpose(0, 2, 1)
                else:
                    vxs[i,] = vx[i:i+timestep,]
            vys = vy[timestep-1:]

        if vx.ndim == 4 and timestep is not None:
            vxs = vxs.transpose(0, 1, 4, 2, 3)

    if ifrandom_split=='yes':
        trainy,testy,trainx,testx = train_test_split(vys,vxs,test_size=test_size,random_state=25)
    elif ifrandom_split=='no':
        index=int((1-test_size)*vys.shape[0])
        trainy=vys[:index,]
        testy=vys[index:,]
        trainx=vxs[:index,]
        testx=vxs[index:,]
    elif ifrandom_split=='all_train' or ifrandom_split=='all_test' or ifrandom_split=='just_model':
        trainy=vys
        testy=vys
        trainx=vxs
        testx=vxs

    predicty_return = predicty
    testy_return = _adl_restore_y_array(testy)
    if timestep!=None:
        input_channel=trainx.shape[2]
        train_position=np.zeros((trainx.shape[0],trainx.shape[1]))
        test_position=np.zeros((testx.shape[0],testx.shape[1]))
        for i in range(trainx.shape[0]):
            train_position[i,:]=np.arange(0,timestep,1)
        for i in range(testx.shape[0]):
            test_position[i,:]=np.arange(0,timestep,1)
    else:
        input_channel=trainx.shape[1]
        train_position=np.zeros((trainx.shape[0],6))
        test_position=np.zeros((testx.shape[0],6))
        for i in range(trainx.shape[0]):
            train_position[i,:]=np.arange(0,6,1)
        for i in range(testx.shape[0]):
            test_position[i,:]=np.arange(0,6,1)
    class TimeDistributed(nn.Module):
        def __init__(self, module, batch_first=False):
            super(TimeDistributed, self).__init__()
            self.module = module
            self.batch_first = batch_first

        def forward(self, x):
            if len(x.size()) <= 2:
                return self.module(x)

            if self.batch_first:
                batch_size, seq_len = x.shape[0], x.shape[1]
            else:
                seq_len, batch_size = x.shape[0], x.shape[1]

            x_reshaped = x.reshape(batch_size * seq_len, *x.shape[2:])
            y = self.module(x_reshaped)

            if self.batch_first:
                y = y.reshape(batch_size, seq_len, *y.shape[1:])
            else:
                y = y.reshape(seq_len, batch_size, *y.shape[1:])
            return y
    class FlexibleBatchNorm(nn.Module):
        """
        BatchNorm wrapper that keeps the original Auto_DeepLearning_pytorch style,
        but supports tensors produced by CNN/GCN/GAT/NLP layers:
          2D: (B, F)
          3D: (B, T, F) or (B, N, F) when feature is the last dim
          4D: (B, C, H, W) or (B, T, C, N)
          5D: (B, T, C, H, W)
        It normalizes the channel/feature dimension and restores the input layout.
        """
        def __init__(self, num_features):
            super().__init__()
            self.num_features = int(num_features)
            self.bn1d = nn.BatchNorm1d(self.num_features)
            self.bn2d = nn.BatchNorm2d(self.num_features)

        def forward(self, x):
            if x.dim() == 2:
                return self.bn1d(x)
            elif x.dim() == 3:
                if x.shape[-1] == self.num_features:
                    y = x.transpose(1, 2).contiguous()
                    y = self.bn1d(y)
                    return y.transpose(1, 2).contiguous()
                elif x.shape[1] == self.num_features:
                    return self.bn1d(x)
                else:
                    raise ValueError(f"FlexibleBatchNorm cannot infer channel dim for shape {tuple(x.shape)} with num_features={self.num_features}")
            elif x.dim() == 4:
                if x.shape[1] == self.num_features:
                    return self.bn2d(x)
                elif x.shape[2] == self.num_features:
                    B, T, C, N = x.shape
                    y = x.reshape(B * T, C, N)
                    y = self.bn1d(y)
                    return y.reshape(B, T, C, N).contiguous()
                elif x.shape[-1] == self.num_features:
                    B = x.shape[0]
                    y = x.reshape(-1, self.num_features)
                    y = self.bn1d(y)
                    return y.reshape_as(x).contiguous()
                else:
                    raise ValueError(f"FlexibleBatchNorm cannot infer channel dim for shape {tuple(x.shape)} with num_features={self.num_features}")
            elif x.dim() == 5:
                if x.shape[2] == self.num_features:
                    B, T, C, H, W = x.shape
                    y = x.reshape(B * T, C, H, W)
                    y = self.bn2d(y)
                    return y.reshape(B, T, C, H, W).contiguous()
                else:
                    raise ValueError(f"FlexibleBatchNorm cannot infer channel dim for shape {tuple(x.shape)} with num_features={self.num_features}")
            else:
                raise ValueError(f"FlexibleBatchNorm only supports 2D-5D input, got {tuple(x.shape)}")

    class FlexibleLayerNorm(nn.Module):
        """LayerNorm wrapper that normalizes the current feature/channel dimension and restores layout."""
        def __init__(self, normalized_shape):
            super().__init__()
            self.normalized_shape = int(normalized_shape)
            self.ln = nn.LayerNorm(self.normalized_shape)

        def forward(self, x):
            if x.shape[-1] == self.normalized_shape:
                return self.ln(x)
            elif x.dim() == 4 and x.shape[2] == self.normalized_shape:
                y = x.permute(0, 1, 3, 2).contiguous()
                y = self.ln(y)
                return y.permute(0, 1, 3, 2).contiguous()
            elif x.dim() == 4 and x.shape[1] == self.normalized_shape:
                y = x.permute(0, 2, 3, 1).contiguous()
                y = self.ln(y)
                return y.permute(0, 3, 1, 2).contiguous()
            elif x.dim() == 5 and x.shape[2] == self.normalized_shape:
                y = x.permute(0, 1, 3, 4, 2).contiguous()
                y = self.ln(y)
                return y.permute(0, 1, 4, 2, 3).contiguous()
            elif x.dim() == 3 and x.shape[1] == self.normalized_shape:
                y = x.transpose(1, 2).contiguous()
                y = self.ln(y)
                return y.transpose(1, 2).contiguous()
            else:
                raise ValueError(f"FlexibleLayerNorm cannot infer feature dim for shape {tuple(x.shape)} with normalized_shape={self.normalized_shape}")

    class LearnableGCNGraph(nn.Module):
        """
        物理先验 + 可学习修正的 GCN。

        A_final = normalize(adj_prior * softplus(edge_kernel))

        支持:
        x: (B, T, C, H, W)
        x: (B, C, H, W)
        x: (B, C, N)
        """
        def __init__(self, in_channels, out_channels, n_nodes, adj_prior=None, bias=True):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.n_nodes = n_nodes

            if adj_prior is None:
                adj_prior = np.ones((n_nodes, n_nodes), dtype=np.float32)
                adj_prior = adj_prior / (adj_prior.sum(axis=1, keepdims=True) + 1e-12)

            adj_prior = torch.tensor(adj_prior, dtype=torch.float32)
            if adj_prior.shape != (n_nodes, n_nodes):
                raise ValueError(f"adj_prior shape {adj_prior.shape} != ({n_nodes}, {n_nodes})")

            self.register_buffer("adj_prior", adj_prior)

            self.edge_kernel = nn.Parameter(torch.full((n_nodes, n_nodes), 0.54132485))

            self.linear = nn.Linear(in_channels, out_channels, bias=bias)

        def _get_adj(self):
            edge_scale = torch.nn.functional.softplus(self.edge_kernel)
            A = self.adj_prior * edge_scale
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
            return A

        def _graph_forward(self, x_nodes):
            A = self._get_adj()                         # (N, N)
            h = torch.matmul(A.unsqueeze(0), x_nodes)   # (B, N, F)
            h = self.linear(h)                          # (B, N, out_channels)
            return h

        def forward(self, x):
            if x.dim() == 5:
                B, T, C, H, W = x.shape
                N = H * W
                if N != self.n_nodes:
                    raise ValueError(f"GCN n_nodes mismatch: layer={self.n_nodes}, input={N}")

                x_nodes = x.permute(0, 1, 3, 4, 2).contiguous().view(B * T, N, C)
                y_nodes = self._graph_forward(x_nodes)
                y = y_nodes.view(B, T, H, W, self.out_channels).permute(0, 1, 4, 2, 3).contiguous()
                return y

            elif x.dim() == 4:
                if x.shape[-1] == self.n_nodes and x.shape[2] == self.in_channels:
                    B, T, C, N = x.shape
                    x_nodes = x.permute(0, 1, 3, 2).contiguous().view(B * T, N, C)
                    y_nodes = self._graph_forward(x_nodes)
                    y = y_nodes.view(B, T, N, self.out_channels).permute(0, 1, 3, 2).contiguous()
                    return y

                if x.shape[-1] == self.in_channels and x.shape[1] * x.shape[2] == self.n_nodes:
                    B, H, W, C = x.shape
                    N = H * W
                    x_nodes = x.contiguous().view(B, N, C)
                    y_nodes = self._graph_forward(x_nodes)
                    y = y_nodes.view(B, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()
                    return y

                B, C, H, W = x.shape
                N = H * W
                if N != self.n_nodes:
                    raise ValueError(f"GCN n_nodes mismatch: layer={self.n_nodes}, input={N}")

                x_nodes = x.permute(0, 2, 3, 1).contiguous().view(B, N, C)
                y_nodes = self._graph_forward(x_nodes)
                y = y_nodes.view(B, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()
                return y

            elif x.dim() == 3:
                if x.shape[1] == self.in_channels and x.shape[2] == self.n_nodes:
                    x_nodes = x.permute(0, 2, 1).contiguous()
                    y_nodes = self._graph_forward(x_nodes)
                    return y_nodes.permute(0, 2, 1).contiguous()
                if x.shape[1] == self.n_nodes and x.shape[2] == self.in_channels:
                    return self._graph_forward(x)
                raise ValueError(
                    f"GCN expects (B,C,N)=(*,{self.in_channels},{self.n_nodes}) "
                    f"or legacy (B,N,F)=(*,{self.n_nodes},{self.in_channels}), got {tuple(x.shape)}"
                )

            else:
                raise ValueError(f"GCN only supports 3D/4D/5D input, got shape {x.shape}")
    class DenseGATGraph(nn.Module):
        """
        物理先验 mask/bias + 可学习 attention 的 GAT。

        adj_prior > 0 的位置允许 attention；
        log(adj_prior) 作为初始物理偏置；
        edge_bias 是可学习边偏置。

        支持:
        x: (B, T, C, H, W)
        x: (B, C, H, W)
        x: (B, C, N)
        """
        def __init__(self, in_channels, out_channels, n_nodes,
                     adj_prior=None, heads=2, dropout=0.0, negative_slope=0.2):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.n_nodes = n_nodes
            self.heads = heads

            if adj_prior is None:
                adj_prior = np.ones((n_nodes, n_nodes), dtype=np.float32)
                adj_prior = adj_prior / (adj_prior.sum(axis=1, keepdims=True) + 1e-12)

            adj_prior = torch.tensor(adj_prior, dtype=torch.float32)
            if adj_prior.shape != (n_nodes, n_nodes):
                raise ValueError(f"adj_prior shape {adj_prior.shape} != ({n_nodes}, {n_nodes})")

            self.register_buffer("adj_prior", adj_prior)
            self.register_buffer("adj_mask", adj_prior > 0)

            self.edge_bias = nn.Parameter(torch.zeros(n_nodes, n_nodes))

            self.linear = nn.Linear(in_channels, out_channels * heads, bias=False)
            self.attn_l = nn.Parameter(torch.empty(heads, out_channels))
            self.attn_r = nn.Parameter(torch.empty(heads, out_channels))

            self.leaky_relu = nn.LeakyReLU(negative_slope)
            self.dropout = nn.Dropout(dropout)

            nn.init.xavier_uniform_(self.attn_l)
            nn.init.xavier_uniform_(self.attn_r)

        def _graph_forward(self, x_nodes):
            B, N, F = x_nodes.shape

            h = self.linear(x_nodes)  # (B, N, heads*out_channels)
            h = h.view(B, N, self.heads, self.out_channels).permute(0, 2, 1, 3)

            score_l = (h * self.attn_l.view(1, self.heads, 1, self.out_channels)).sum(dim=-1)
            score_r = (h * self.attn_r.view(1, self.heads, 1, self.out_channels)).sum(dim=-1)

            e = score_l.unsqueeze(-1) + score_r.unsqueeze(-2)  # (B, heads, N, N)
            e = self.leaky_relu(e)

            prior_bias = torch.log(self.adj_prior.clamp_min(1e-12))
            e = e + prior_bias.view(1, 1, N, N) + self.edge_bias.view(1, 1, N, N)

            e = e.masked_fill(~self.adj_mask.view(1, 1, N, N), -1e9)

            alpha = torch.softmax(e, dim=-1)
            alpha = self.dropout(alpha)

            out = torch.matmul(alpha, h)  # (B, heads, N, out_channels)

            out = out.mean(dim=1)  # (B, N, out_channels)
            return out

        def forward(self, x):
            if x.dim() == 5:
                B, T, C, H, W = x.shape
                N = H * W
                if N != self.n_nodes:
                    raise ValueError(f"GAT n_nodes mismatch: layer={self.n_nodes}, input={N}")

                x_nodes = x.permute(0, 1, 3, 4, 2).contiguous().view(B * T, N, C)
                y_nodes = self._graph_forward(x_nodes)
                y = y_nodes.view(B, T, H, W, self.out_channels).permute(0, 1, 4, 2, 3).contiguous()
                return y

            elif x.dim() == 4:
                if x.shape[-1] == self.n_nodes and x.shape[2] == self.in_channels:
                    B, T, C, N = x.shape
                    x_nodes = x.permute(0, 1, 3, 2).contiguous().view(B * T, N, C)
                    y_nodes = self._graph_forward(x_nodes)
                    y = y_nodes.view(B, T, N, self.out_channels).permute(0, 1, 3, 2).contiguous()
                    return y

                if x.shape[-1] == self.in_channels and x.shape[1] * x.shape[2] == self.n_nodes:
                    B, H, W, C = x.shape
                    N = H * W
                    x_nodes = x.contiguous().view(B, N, C)
                    y_nodes = self._graph_forward(x_nodes)
                    y = y_nodes.view(B, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()
                    return y

                B, C, H, W = x.shape
                N = H * W
                if N != self.n_nodes:
                    raise ValueError(f"GAT n_nodes mismatch: layer={self.n_nodes}, input={N}")

                x_nodes = x.permute(0, 2, 3, 1).contiguous().view(B, N, C)
                y_nodes = self._graph_forward(x_nodes)
                y = y_nodes.view(B, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()
                return y

            elif x.dim() == 3:
                if x.shape[1] == self.in_channels and x.shape[2] == self.n_nodes:
                    x_nodes = x.permute(0, 2, 1).contiguous()
                    y_nodes = self._graph_forward(x_nodes)
                    return y_nodes.permute(0, 2, 1).contiguous()
                if x.shape[1] == self.n_nodes and x.shape[2] == self.in_channels:
                    return self._graph_forward(x)
                raise ValueError(
                    f"GAT expects (B,C,N)=(*,{self.in_channels},{self.n_nodes}) "
                    f"or legacy (B,N,F)=(*,{self.n_nodes},{self.in_channels}), got {tuple(x.shape)}"
                )

            else:
                raise ValueError(f"GAT only supports 3D/4D/5D input, got shape {x.shape}")

    class DenseChebNetGraph(nn.Module):
        """
        ChebNet graph convolution with physics prior adjacency and learnable edge scaling.

        The K-hop order is controlled by cheb_k. It uses Chebyshev-style recurrence:
            T0(X) = X
            T1(X) = A X
            Tk(X) = 2 A T{k-1}(X) - T{k-2}(X)

        Supports the same tensor layouts as LearnableGCNGraph/DenseGATGraph:
            x: (B, T, C, H, W)
            x: (B, T, C, N)
            x: (B, C, H, W)
            x: (B, C, N)
        """
        def __init__(self, in_channels, out_channels, n_nodes, cheb_k=3, adj_prior=None, bias=True):
            super().__init__()
            self.in_channels = int(in_channels)
            self.out_channels = int(out_channels)
            self.n_nodes = int(n_nodes)
            self.cheb_k = max(1, int(cheb_k))

            if adj_prior is None:
                adj_prior = np.ones((n_nodes, n_nodes), dtype=np.float32)
                adj_prior = adj_prior / (adj_prior.sum(axis=1, keepdims=True) + 1e-12)

            adj_prior = torch.tensor(adj_prior, dtype=torch.float32)
            if adj_prior.shape != (n_nodes, n_nodes):
                raise ValueError(f"adj_prior shape {adj_prior.shape} != ({n_nodes}, {n_nodes})")

            self.register_buffer("adj_prior", adj_prior)
            self.edge_kernel = nn.Parameter(torch.full((n_nodes, n_nodes), 0.54132485))
            self.linears = nn.ModuleList([
                nn.Linear(self.in_channels, self.out_channels, bias=bias) for _ in range(self.cheb_k + 1)
            ])

        def _get_adj(self):
            edge_scale = torch.nn.functional.softplus(self.edge_kernel)
            A = self.adj_prior * edge_scale
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
            return A

        def _graph_forward(self, x_nodes):
            A = self._get_adj()
            supports = [x_nodes]
            if self.cheb_k >= 1:
                supports.append(torch.matmul(A.unsqueeze(0), x_nodes))
            for kk in range(2, self.cheb_k + 1):
                supports.append(2.0 * torch.matmul(A.unsqueeze(0), supports[-1]) - supports[-2])

            out = 0.0
            for kk, support in enumerate(supports):
                out = out + self.linears[kk](support)
            return out

        def forward(self, x):
            if x.dim() == 5:
                B, T, C, H, W = x.shape
                N = H * W
                if N != self.n_nodes:
                    raise ValueError(f"ChebNet n_nodes mismatch: layer={self.n_nodes}, input={N}")
                x_nodes = x.permute(0, 1, 3, 4, 2).contiguous().view(B * T, N, C)
                y_nodes = self._graph_forward(x_nodes)
                y = y_nodes.view(B, T, H, W, self.out_channels).permute(0, 1, 4, 2, 3).contiguous()
                return y
            elif x.dim() == 4:
                if x.shape[-1] == self.n_nodes and x.shape[2] == self.in_channels:
                    B, T, C, N = x.shape
                    x_nodes = x.permute(0, 1, 3, 2).contiguous().view(B * T, N, C)
                    y_nodes = self._graph_forward(x_nodes)
                    y = y_nodes.view(B, T, N, self.out_channels).permute(0, 1, 3, 2).contiguous()
                    return y
                if x.shape[-1] == self.in_channels and x.shape[1] * x.shape[2] == self.n_nodes:
                    B, H, W, C = x.shape
                    N = H * W
                    x_nodes = x.contiguous().view(B, N, C)
                    y_nodes = self._graph_forward(x_nodes)
                    y = y_nodes.view(B, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()
                    return y

                B, C, H, W = x.shape
                N = H * W
                if N != self.n_nodes:
                    raise ValueError(f"ChebNet n_nodes mismatch: layer={self.n_nodes}, input={N}")
                x_nodes = x.permute(0, 2, 3, 1).contiguous().view(B, N, C)
                y_nodes = self._graph_forward(x_nodes)
                y = y_nodes.view(B, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()
                return y
            elif x.dim() == 3:
                if x.shape[1] == self.in_channels and x.shape[2] == self.n_nodes:
                    x_nodes = x.permute(0, 2, 1).contiguous()
                    y_nodes = self._graph_forward(x_nodes)
                    return y_nodes.permute(0, 2, 1).contiguous()
                if x.shape[1] == self.n_nodes and x.shape[2] == self.in_channels:
                    return self._graph_forward(x)
                raise ValueError(
                    f"ChebNet expects (B,C,N)=(*,{self.in_channels},{self.n_nodes}) "
                    f"or legacy (B,N,F)=(*,{self.n_nodes},{self.in_channels}), got {tuple(x.shape)}"
                )
            else:
                raise ValueError(f"ChebNet only supports 3D/4D/5D input, got shape {x.shape}")

    class DenseGraphSAGEGraph(nn.Module):
        """
        Dense GraphSAGE layer with mean/max/lstm aggregators and physics prior adjacency.

        model_list usage:
            ['gsage', out_channels]
            ['gsage', out_channels, 'mean'/'max'/'lstm']
            ['gsage', out_channels, 'lstm', lstm_units]
        """
        def __init__(self, in_channels, out_channels, n_nodes, adj_prior=None, aggr='mean', lstm_units=None, bias=True):
            super().__init__()
            self.in_channels = int(in_channels)
            self.out_channels = int(out_channels)
            self.n_nodes = int(n_nodes)
            self.aggr = str(aggr).lower()
            if self.aggr not in ['mean', 'max', 'lstm']:
                raise ValueError("gsage_aggr must be one of: 'mean', 'max', 'lstm'.")

            if adj_prior is None:
                adj_prior = np.ones((n_nodes, n_nodes), dtype=np.float32)
                adj_prior = adj_prior / (adj_prior.sum(axis=1, keepdims=True) + 1e-12)

            adj_prior = torch.tensor(adj_prior, dtype=torch.float32)
            if adj_prior.shape != (n_nodes, n_nodes):
                raise ValueError(f"adj_prior shape {adj_prior.shape} != ({n_nodes}, {n_nodes})")

            self.register_buffer("adj_prior", adj_prior)
            self.register_buffer("adj_mask", adj_prior > 0)
            self.edge_kernel = nn.Parameter(torch.full((n_nodes, n_nodes), 0.54132485))

            if self.aggr == 'lstm':
                self.lstm_units = int(lstm_units) if lstm_units is not None else self.in_channels
                self.neigh_lstm = nn.LSTM(self.in_channels, self.lstm_units, batch_first=True)
                neigh_dim = self.lstm_units
            else:
                self.lstm_units = None
                neigh_dim = self.in_channels

            self.update_linear = nn.Linear(self.in_channels + neigh_dim, self.out_channels, bias=bias)

        def _get_adj(self):
            edge_scale = torch.nn.functional.softplus(self.edge_kernel)
            A = self.adj_prior * edge_scale
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
            return A

        def _aggregate(self, x_nodes):
            A = self._get_adj()
            if self.aggr == 'mean':
                return torch.matmul(A.unsqueeze(0), x_nodes)

            if self.aggr == 'max':
                neighbors = x_nodes[:, None, :, :] * A[None, :, :, None]
                mask = self.adj_mask[None, :, :, None]
                neighbors = neighbors.masked_fill(~mask, -1e9)
                neigh = torch.max(neighbors, dim=2)[0]
                neigh = torch.where(torch.isfinite(neigh), neigh, torch.zeros_like(neigh))
                return neigh

            B, N, F_in = x_nodes.shape
            neighbors = x_nodes[:, None, :, :] * A[None, :, :, None]  # (B,N,N,F)
            neighbors = neighbors.reshape(B * N, N, F_in)
            _, (h_n, _) = self.neigh_lstm(neighbors)
            neigh = h_n[-1].reshape(B, N, self.lstm_units)
            return neigh

        def _graph_forward(self, x_nodes):
            neigh = self._aggregate(x_nodes)
            h = torch.cat([x_nodes, neigh], dim=-1)
            return self.update_linear(h)

        def forward(self, x):
            if x.dim() == 5:
                B, T, C, H, W = x.shape
                N = H * W
                if N != self.n_nodes:
                    raise ValueError(f"GraphSAGE n_nodes mismatch: layer={self.n_nodes}, input={N}")
                x_nodes = x.permute(0, 1, 3, 4, 2).contiguous().view(B * T, N, C)
                y_nodes = self._graph_forward(x_nodes)
                y = y_nodes.view(B, T, H, W, self.out_channels).permute(0, 1, 4, 2, 3).contiguous()
                return y
            elif x.dim() == 4:
                if x.shape[-1] == self.n_nodes and x.shape[2] == self.in_channels:
                    B, T, C, N = x.shape
                    x_nodes = x.permute(0, 1, 3, 2).contiguous().view(B * T, N, C)
                    y_nodes = self._graph_forward(x_nodes)
                    y = y_nodes.view(B, T, N, self.out_channels).permute(0, 1, 3, 2).contiguous()
                    return y
                if x.shape[-1] == self.in_channels and x.shape[1] * x.shape[2] == self.n_nodes:
                    B, H, W, C = x.shape
                    N = H * W
                    x_nodes = x.contiguous().view(B, N, C)
                    y_nodes = self._graph_forward(x_nodes)
                    y = y_nodes.view(B, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()
                    return y

                B, C, H, W = x.shape
                N = H * W
                if N != self.n_nodes:
                    raise ValueError(f"GraphSAGE n_nodes mismatch: layer={self.n_nodes}, input={N}")
                x_nodes = x.permute(0, 2, 3, 1).contiguous().view(B, N, C)
                y_nodes = self._graph_forward(x_nodes)
                y = y_nodes.view(B, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()
                return y
            elif x.dim() == 3:
                if x.shape[1] == self.in_channels and x.shape[2] == self.n_nodes:
                    x_nodes = x.permute(0, 2, 1).contiguous()
                    y_nodes = self._graph_forward(x_nodes)
                    return y_nodes.permute(0, 2, 1).contiguous()
                if x.shape[1] == self.n_nodes and x.shape[2] == self.in_channels:
                    return self._graph_forward(x)
                raise ValueError(
                    f"GraphSAGE expects (B,C,N)=(*,{self.in_channels},{self.n_nodes}) "
                    f"or legacy (B,N,F)=(*,{self.n_nodes},{self.in_channels}), got {tuple(x.shape)}"
                )
            else:
                raise ValueError(f"GraphSAGE only supports 3D/4D/5D input, got shape {x.shape}")

    def find_string_in_nested_list(target_string, nested_list):
        for item in nested_list:
            if isinstance(item, list):
                if find_string_in_nested_list(target_string, item):
                    return True
            elif isinstance(item, str):
                if item == target_string:
                    return True
        return False
    def _keep_topk_per_row(weights, r_number):
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        if weights.ndim == 2:
            N = weights.shape[0]
            if r_number is None:
                r_number = N
            r_number = min(r_number, N)

            out = np.zeros_like(weights, dtype=np.float32)
            for i in range(N):
                indexs = np.argsort(weights[i, :])[-r_number:]
                out[i, indexs] = weights[i, indexs]
            return out

        elif weights.ndim == 3:
            T, N, _ = weights.shape
            if r_number is None:
                r_number = N
            r_number = min(r_number, N)

            out = np.zeros_like(weights, dtype=np.float32)
            for t in range(T):
                for i in range(N):
                    indexs = np.argsort(weights[t, i, :])[-r_number:]
                    out[t, i, indexs] = weights[t, i, indexs]
            return out

        else:
            raise ValueError("weights must be 2D or 3D")


    def build_physics_adj_prior(vx, lat, lon, a_mode='d', r_number=None,
                                u=None, v=None, timestep=None):
        """
        构建物理先验邻接矩阵 adj_prior: (N, N)

        a_mode='d':
            基于 lat/lon 的 1/distance 邻接矩阵。

        a_mode='uvd':
            基于 1/distance * cos(direction - wind_direction) 构建逐时次邻接，
            然后对时间平均，得到一个静态 adj_prior，用于初始化可学习邻接矩阵。

        注意：
            这里返回的是静态 prior adjacency。
            这样不用改训练循环，仍然保持 model(x, position) 的调用方式。
        """
        if lat is None or lon is None:
            raise ValueError("使用 gc/ga/chebnet/gsage 且需要物理邻接矩阵时，lat 和 lon 不能为 None。")

        lat_arr = np.asarray(lat)
        lon_arr = np.asarray(lon)

        if vx.ndim == 4:
            if lon_arr.ndim == 1 and lat_arr.ndim == 1:
                H_lat = lat_arr.shape[0]
                W_lon = lon_arr.shape[0]
                if vx.shape[1] == H_lat and vx.shape[2] == W_lon:
                    H, W = vx.shape[1], vx.shape[2]
                elif vx.shape[2] == H_lat and vx.shape[3] == W_lon:
                    H, W = vx.shape[2], vx.shape[3]
                else:
                    H, W = vx.shape[1], vx.shape[2]
                lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
            else:
                lon_grid, lat_grid = lon_arr, lat_arr
                H, W = lat_grid.shape[0], lat_grid.shape[1]
            N = H * W

            lon_flat = lon_grid.reshape(N)
            lat_flat = lat_grid.reshape(N)

        elif vx.ndim == 3:
            N = vx.shape[1]
            lon_flat = lon_arr.reshape(N)
            lat_flat = lat_arr.reshape(N)

        else:
            raise ValueError("gc/ga/chebnet/gsage 的物理邻接矩阵目前只支持 vx.ndim == 4 或 vx.ndim == 3。")

        distance = np.zeros((N, N), dtype=np.float32)
        for i in tqdm(range(N), desc="Build graph distance"):
            for j in range(N):
                distance[i, j] = geodesic(
                    (float(lat_flat[i]), float(lon_flat[i])),
                    (float(lat_flat[j]), float(lon_flat[j]))
                ).km

        distance_safe = distance.copy()
        distance_safe[distance_safe == 0] = np.nan
        base_weights = 1.0 / distance_safe
        base_weights = np.nan_to_num(base_weights, nan=0.0, posinf=0.0, neginf=0.0)

        if a_mode == 'd':
            adj_prior = _keep_topk_per_row(base_weights, r_number)
            np.fill_diagonal(adj_prior, 1.0)

        elif a_mode == 'uvd':
            if u is None or v is None:
                raise ValueError("a_mode='uvd' 时，u 和 v 不能为 None。")

            u_arr = np.asarray(u)
            v_arr = np.asarray(v)

            if u_arr.ndim == 3:
                u_arr = u_arr.reshape(u_arr.shape[0], -1)
                v_arr = v_arr.reshape(v_arr.shape[0], -1)
            elif u_arr.ndim == 4:
                raise ValueError("u/v 当前建议传入 (time, H, W) 或 (time, N)。")

            if u_arr.ndim == 2:
                pass
            else:
                raise ValueError("u/v 当前应为 (time, H*W) 或 (time, N)。")

            if timestep is not None:
                u_arr = u_arr[timestep-1:]
                v_arr = v_arr[timestep-1:]

            wind_dir = (np.degrees(np.arctan2(v_arr, u_arr)) + 360.0) % 360.0  # (time, N)

            dis_dir = (
                np.degrees(
                    np.arctan2(
                        lat_flat[:, None] - lat_flat[None, :],
                        lon_flat[:, None] - lon_flat[None, :]
                    )
                ) + 360.0
            ) % 360.0  # (N, N)

            eweights = base_weights[None, :, :] * np.cos(
                (dis_dir[None, :, :] - wind_dir[:, :, None]) * np.pi / 180.0
            )
            eweights[eweights < 0] = 0.0
            eweights = np.nan_to_num(eweights, nan=0.0, posinf=0.0, neginf=0.0)

            eweights = _keep_topk_per_row(eweights, r_number)

            adj_prior = np.nanmean(eweights, axis=0)
            np.fill_diagonal(adj_prior, 1.0)

        else:
            raise ValueError("a_mode 只能是 'd' 或 'uvd'。")

        adj_prior = np.nan_to_num(adj_prior, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum = adj_prior.sum(axis=1, keepdims=True)
        adj_prior = adj_prior / (row_sum + 1e-12)

        return adj_prior.astype(np.float32)
    has_gc = find_string_in_nested_list('gc', model_list)
    has_ga = find_string_in_nested_list('ga', model_list)
    has_chebnet = find_string_in_nested_list('chebnet', model_list)
    has_gsage = find_string_in_nested_list('gsage', model_list)
    has_graph = has_gc or has_ga or has_chebnet or has_gsage
    one_d_spatial_sequence_mode = bool(timestep is not None and vx_original_ndim == 3)
    graph_station_sequence_mode = bool(has_graph and one_d_spatial_sequence_mode)

    if has_graph:
        adj_prior = build_physics_adj_prior(
            vx=vx,
            lat=lat,
            lon=lon,
            a_mode=a_mode,
            r_number=r_number,
            u=u,
            v=v,
            timestep=timestep
        )
    else:
        adj_prior = None
    class Model(nn.Module):
        def __init__(self,model_list,timestep,dilations,ifmulti_scale,num_channels,trainx,cov_padding,cov_strides,pooling_strides,dropout,residual_deep,residual_activation,residual_norm,vit_patch_size,vit_dim_feedforward,vit_deep,vit_num_heads,swin_deep,swin_num_heads,cbam_reduction,chebnet_k,gsage_aggr,gsage_lstm_units,transformer_encoder_deep,transformer_num_heads,transformer_dim_feedforward,tcn_kernel_size,nlp_units,nlp_num_layers,ltc_solver,ltc_ode_unfolds,ltc_dt,ltc_tau_min,ltc_activation,ltc_dropout,mamba_state,mamba_conv,mamba_expand,activation,adj_prior=None,outputstep=None,seq2seq_target_shape=None):
            super(Model,self).__init__()
            exec('from torch import nn', globals(), self.__dict__)
            exec('from torch.nn import Module,BatchNorm1d,BatchNorm2d,LayerNorm,Conv2d,MaxPool2d,AvgPool2d,RNN,LSTM,GRU,Embedding,TransformerEncoderLayer,TransformerEncoder,Dropout,LeakyReLU,ReLU,PReLU,Sigmoid,Tanh,ELU,Softmax,Linear,Flatten,ZeroPad2d,AdaptiveMaxPool2d,AdaptiveAvgPool2d', globals(), self.__dict__)
            exec('import numpy as np', globals(), self.__dict__)
            exec('import torch', globals(), self.__dict__)
            layer_types = {layer[0] for layer in model_list}
            if 'tcn' in layer_types:
                exec('from pytorch_tcn import TCN', globals(), self.__dict__)
            if 'vit' in layer_types:
                exec('from vit_pytorch import ViT', globals(), self.__dict__)
            if layer_types.intersection({'mamba', 'mamba2'}):
                exec('import mamba_ssm.modules.mamba_simple as mamba_module', globals(), self.__dict__)
                exec('mamba_module.causal_conv1d_fn = None', globals(), self.__dict__)
                exec('mamba_module.causal_conv1d_update = None', globals(), self.__dict__)
                exec('import mamba_ssm.modules.mamba2 as mamba2_module', globals(), self.__dict__)
                exec('mamba2_module.causal_conv1d_fn = None', globals(), self.__dict__)
                exec('mamba2_module.causal_conv1d_update = None', globals(), self.__dict__)
                exec('from mamba_ssm import Mamba,Mamba2', globals(), self.__dict__)
            if 'swin' in layer_types:
                exec('from timm.models.swin_transformer import SwinTransformer', globals(), self.__dict__)
            exec('import math', globals(), self.__dict__)
            exec('import torch', globals(), self.__dict__)
            exec('import torch.nn.functional as F', globals(), self.__dict__)

            class LTCLayer(nn.Module):
                """
                Liquid Time-constant (LTC) sequence layer.

                Interface is intentionally RNN-like:
                    output, h_n = LTCLayer(input, h0=None)
                input:  (batch, seq_len, input_size)
                output: (batch, seq_len, hidden_size)
                h_n:    (num_layers, batch, hidden_size)

                The default solver is the fused/semi-implicit update used by the LTC paper:
                    h_next = (h + dt * f * A) / (1 + dt * (1/tau + f))
                """
                def __init__(self, input_size, hidden_size, num_layers=1,
                             solver='fused', ode_unfolds=4, dt=1.0,
                             tau_min=1e-4, activation='softplus', dropout=0.0):
                    super().__init__()
                    self.input_size = int(input_size)
                    self.hidden_size = int(hidden_size)
                    self.num_layers = int(num_layers)
                    self.solver = str(solver).lower()
                    self.ode_unfolds = max(1, int(ode_unfolds))
                    self.dt = float(dt)
                    self.tau_min = float(tau_min)
                    self.activation = str(activation).lower()
                    self.dropout_p = float(dropout) if dropout is not None else 0.0

                    self.input_linears = nn.ModuleList()
                    self.recurrent_linears = nn.ModuleList()
                    self.raw_taus = nn.ParameterList()
                    self.As = nn.ParameterList()

                    for layer_i in range(self.num_layers):
                        layer_input_size = self.input_size if layer_i == 0 else self.hidden_size
                        self.input_linears.append(nn.Linear(layer_input_size, self.hidden_size, bias=True))
                        self.recurrent_linears.append(nn.Linear(self.hidden_size, self.hidden_size, bias=False))
                        self.raw_taus.append(nn.Parameter(torch.zeros(self.hidden_size)))
                        self.As.append(nn.Parameter(torch.empty(self.hidden_size).uniform_(-1.0, 1.0)))

                    self.dropout_layer = nn.Dropout(self.dropout_p)

                def _positive_activation(self, z):
                    if self.activation == 'softplus':
                        return F.softplus(z)
                    elif self.activation == 'sigmoid':
                        return torch.sigmoid(z)
                    elif self.activation == 'relu':
                        return F.relu(z)
                    elif self.activation == 'tanh':
                        return 0.5 * (torch.tanh(z) + 1.0)
                    elif self.activation == 'elu':
                        return F.elu(z) + 1.0
                    else:
                        return F.softplus(z)

                def _gate(self, layer_i, input_t, h):
                    z = self.input_linears[layer_i](input_t) + self.recurrent_linears[layer_i](h)
                    return self._positive_activation(z)

                def _rhs(self, layer_i, input_t, h):
                    tau = F.softplus(self.raw_taus[layer_i]) + self.tau_min
                    f = self._gate(layer_i, input_t, h)
                    return -(1.0 / tau + f) * h + f * self.As[layer_i]

                def _step(self, layer_i, input_t, h):
                    dt_sub = self.dt / self.ode_unfolds
                    tau = F.softplus(self.raw_taus[layer_i]) + self.tau_min
                    A = self.As[layer_i]

                    for _ in range(self.ode_unfolds):
                        if self.solver in ['fused', 'semi_implicit', 'implicit', 'ltc']:
                            f = self._gate(layer_i, input_t, h)
                            h = (h + dt_sub * f * A) / (1.0 + dt_sub * (1.0 / tau + f))
                        elif self.solver in ['euler', 'explicit_euler']:
                            h = h + dt_sub * self._rhs(layer_i, input_t, h)
                        elif self.solver in ['rk4', 'runge_kutta', 'runge-kutta']:
                            k1 = self._rhs(layer_i, input_t, h)
                            k2 = self._rhs(layer_i, input_t, h + 0.5 * dt_sub * k1)
                            k3 = self._rhs(layer_i, input_t, h + 0.5 * dt_sub * k2)
                            k4 = self._rhs(layer_i, input_t, h + dt_sub * k3)
                            h = h + (dt_sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                        else:
                            raise ValueError("ltc_solver must be one of: fused, semi_implicit, implicit, euler, rk4")
                    return h

                def forward(self, input, h0=None):
                    if input.dim() != 3:
                        raise ValueError('LTCLayer expects input with shape (batch, seq_len, feature).')
                    batch_size, seq_len, _ = input.shape
                    x_seq = input
                    h_list = []

                    for layer_i in range(self.num_layers):
                        if h0 is None:
                            h = input.new_zeros(batch_size, self.hidden_size)
                        else:
                            h = h0[layer_i]
                        layer_outputs = []
                        for t_i in range(seq_len):
                            h = self._step(layer_i, x_seq[:, t_i, :], h)
                            layer_outputs.append(h.unsqueeze(1))
                        x_seq = torch.cat(layer_outputs, dim=1)
                        if self.dropout_p > 0 and layer_i < self.num_layers - 1:
                            x_seq = self.dropout_layer(x_seq)
                        h_list.append(h.unsqueeze(0))

                    h_n = torch.cat(h_list, dim=0)
                    return x_seq, h_n

            self.__dict__['LTCLayer'] = LTCLayer

            self.model_list=model_list
            self.timestep=timestep
            self.dilations=dilations
            self.ifmulti_scale=ifmulti_scale
            self.num_channels=num_channels
            self.trainx=trainx
            self.cov_padding=cov_padding
            self.cov_strides=cov_strides
            self.pooling_strides=pooling_strides
            self.dropout=dropout
            self.residual_deep=residual_deep
            self.residual_activation=residual_activation
            self.residual_norm=residual_norm
            self.vit_patch_size=vit_patch_size
            self.vit_dim_feedforward=vit_dim_feedforward
            self.vit_deep=vit_deep
            self.vit_num_heads=vit_num_heads
            self.swin_deep=swin_deep
            self.swin_num_heads=swin_num_heads
            self.cbam_reduction=cbam_reduction
            self.chebnet_k=chebnet_k
            self.gsage_aggr=gsage_aggr
            self.gsage_lstm_units=gsage_lstm_units
            self.transformer_encoder_deep=transformer_encoder_deep
            self.transformer_num_heads=transformer_num_heads
            self.transformer_dim_feedforward=transformer_dim_feedforward
            self.tcn_kernel_size=tcn_kernel_size
            self.nlp_units=nlp_units
            self.nlp_num_layers=nlp_num_layers
            self.ltc_solver=ltc_solver
            self.ltc_ode_unfolds=ltc_ode_unfolds
            self.ltc_dt=ltc_dt
            self.ltc_tau_min=ltc_tau_min
            self.ltc_activation=ltc_activation
            self.ltc_dropout=dropout if ltc_dropout is None else ltc_dropout
            self.mamba_state=mamba_state
            self.mamba_conv=mamba_conv
            self.mamba_expand=mamba_expand
            self.activation=activation
            self.outputstep=outputstep
            self.seq2seq_mode = outputstep is not None
            self.seq2seq_target_shape = tuple(seq2seq_target_shape) if seq2seq_target_shape is not None else None
            self.nlp_layer_names = ['transformer', 'lstm', 'rnn', 'gru', 'ltc', 'tcn', 'mamba', 'mamba2']
            self.nlp_indices = []
            self.nlp_out_channels = {}
            self.adj_prior = adj_prior
            self.__dict__['self']=self
            self.one_d_spatial_sequence_mode = one_d_spatial_sequence_mode
            self.graph_station_sequence_mode = graph_station_sequence_mode
            if self.one_d_spatial_sequence_mode:
                self.station_num = self.trainx.shape[3]
            elif trainx.ndim>3 and timestep!=None:
                self.hight=self.trainx.shape[3]
                self.weight=self.trainx.shape[4]
            elif trainx.ndim>3 and timestep==None:
                self.hight=self.trainx.shape[2]
                self.weight=self.trainx.shape[3]
            if timestep!=None:
                exec('in_channels=self.trainx.shape[2]', globals(), self.__dict__)
            elif timestep==None:
                exec('in_channels=self.trainx.shape[1]', globals(), self.__dict__)
            self.__dict__['TimeDistributed']=TimeDistributed
            self.__dict__['FlexibleBatchNorm']=FlexibleBatchNorm
            self.__dict__['FlexibleLayerNorm']=FlexibleLayerNorm
            self.__dict__['LearnableGCNGraph'] = LearnableGCNGraph
            self.__dict__['DenseGATGraph'] = DenseGATGraph
            self.__dict__['DenseChebNetGraph'] = DenseChebNetGraph
            self.__dict__['DenseGraphSAGEGraph'] = DenseGraphSAGEGraph
            if ['flatten'] in model_list:
                self.flattened=False
            else:
                self.flattened=True
            if ifmulti_scale=='yes':
                exec('self.dense_input_len=0', globals(), self.__dict__)

            if find_string_in_nested_list('vit', self.model_list) or find_string_in_nested_list('swin', self.model_list):
                def find_best_common_divisor(a, b):
                    def get_common_divisors(a, b):
                        """获取 a 和 b 的所有公约数"""
                        def gcd(x, y):
                            while y:
                                x, y = y, x % y
                            return x

                        common_gcd = gcd(a, b)
                        divisors = []
                        for i in range(2, int(common_gcd**0.5) + 1):
                            if common_gcd % i == 0:
                                divisors.append(i)
                                if i != common_gcd // i:
                                    divisors.append(common_gcd // i)
                        divisors.append(common_gcd)
                        return sorted(divisors)

                    def find_valid_divisor(a, b):
                        """寻找满足条件的公约数"""
                        divisors = get_common_divisors(a, b)
                        best_divisor = None
                        min_value = float('inf')

                        for d in divisors:
                            a_div, b_div = a // d, b // d
                            if d == 1 or a_div == 1 or b_div == 1:
                                continue
                            if a_div * b_div > 16:
                                value = np.abs((a_div - d)) + np.abs((b_div - d))
                                if value < min_value:
                                    min_value = value
                                    best_divisor = d
                        return best_divisor

                    def adjust_values(a, b):
                        """顺序尝试增加 a、b 或轮流增加 a 和 b"""
                        steps = 0
                        while True:
                            steps += 1

                            new_a = a + steps
                            divisor = find_valid_divisor(new_a, b)
                            if divisor:
                                return new_a, b, divisor

                            new_b = b + steps
                            divisor = find_valid_divisor(a, new_b)
                            if divisor:
                                return a, new_b, divisor

                            if (steps % 2) == 1:
                                new_a, new_b = a + steps, b + steps-1
                            else:
                                new_a, new_b = a + steps, b + steps
                            divisor = find_valid_divisor(new_a, new_b)
                            if divisor:
                                return new_a, new_b, divisor

                            if (steps % 2) == 1:
                                new_a, new_b = a + steps-1, b + steps
                            else:
                                new_a, new_b = a + steps, b + steps
                            divisor = find_valid_divisor(new_a, new_b)
                            if divisor:
                                return new_a, new_b, divisor
                    divisor = find_valid_divisor(a, b)
                    if divisor:
                        return a, b, divisor
                    else:
                        new_a, new_b, new_divisor = adjust_values(a, b)
                        return new_a, new_b, new_divisor
                def find_new_a_b(a, b, divisor):
                    """
                    找到新的 a 和 b，使得 divisor 成为它们的公约数。
                    """
                    def is_common_divisor(x, y, d):
                        return x % d == 0 and y % d == 0
                    if is_common_divisor(a, b, divisor):
                            return a, b
                    a1, b1 = a, b  # 只增加 a
                    a2, b2 = a, b  # 只增加 b
                    a3, b3 = a, b  # a 和 b 轮流增加
                    a4, b4 = a, b  # a 和 b 轮流增加

                    while True:
                        a1 += 1
                        if is_common_divisor(a1, b1, divisor):
                            return a1, b1

                        b2 += 1
                        if is_common_divisor(a2, b2, divisor):
                            return a2, b2

                        if (a3 + b3) % 2 == 0:  # 偶数轮，增加 a
                            a3 += 1
                        else:  # 奇数轮，增加 b
                            b3 += 1
                        if is_common_divisor(a3, b3, divisor):
                            return a3, b3
                        if (a4 + b4) % 2 == 0:  # 偶数轮，增加 b
                            b4 += 1
                        else:  # 奇数轮，增加 a
                            a4 += 1
                        if is_common_divisor(a4, b4, divisor):
                            return a4, b4
            for i in range(len(self.model_list)):
                self.__dict__['i']=i
                if self.model_list[i][0] == 'cov' and timestep!=None:
                    if self.cov_padding=='valid':
                        self.hight=1+math.floor((self.hight-self.model_list[i][1][1])/self.cov_strides)
                        self.weight=1+math.floor((self.weight-self.model_list[i][1][2])/self.cov_strides)
                        if self.hight < 1 or self.weight < 1:
                            print('卷积层数过多')
                            break
                    exec('self.conv'+str(i+1)+'=TimeDistributed(Conv2d(in_channels,self.model_list[i][1][0],(self.model_list[i][1][1],self.model_list[i][1][2]),stride=self.cov_strides,padding=self.cov_padding), batch_first=True)', globals(), self.__dict__)
                    exec('in_channels=self.model_list[i][1][0]', globals(), self.__dict__)
                elif self.model_list[i][0] == 'cov' and timestep==None:
                    if self.cov_padding=='valid':
                        self.hight=1+math.floor((self.hight-self.model_list[i][1][1])/self.cov_strides)
                        self.weight=1+math.floor((self.weight-self.model_list[i][1][2])/self.cov_strides)
                        if self.hight < 1 or self.weight < 1:
                            print('卷积层数过多')
                            break
                    exec('self.conv'+str(i+1)+'=Conv2d(in_channels,self.model_list[i][1][0],(self.model_list[i][1][1],self.model_list[i][1][2]),stride=self.cov_strides,padding=self.cov_padding)', globals(), self.__dict__)
                    exec('in_channels=self.model_list[i][1][0]', globals(), self.__dict__)
                elif self.model_list[i][0] == 'resnet' and timestep!=None:
                    if cov_padding=='same':
                        exec('self.pad'+str(i+1)+'=TimeDistributed(ZeroPad2d(padding=(math.floor(math.floor((self.trainx.shape[4]-(self.trainx.shape[4]/self.cov_strides)))/2),math.ceil(math.floor((self.trainx.shape[4]-(self.trainx.shape[4]/self.cov_strides)))/2),math.floor(math.floor((self.trainx.shape[3]-(self.trainx.shape[3]/self.cov_strides)))/2),math.ceil(math.floor((self.trainx.shape[3]-(self.trainx.shape[3]/self.cov_strides)))/2))), batch_first=True)', globals(), self.__dict__)
                        exec('self.conv'+str(i+1)+'=TimeDistributed(Conv2d(in_channels,self.model_list[i][1][0],(self.model_list[i][1][1],self.model_list[i][1][2]),stride=self.cov_strides,padding=self.cov_padding), batch_first=True)', globals(), self.__dict__)
                    else:
                        exec('self.pad'+str(i+1)+'=TimeDistributed(ZeroPad2d(padding=(math.floor(math.floor((self.trainx.shape[4]-((self.trainx.shape[4]-self.model_list[i][1][1]+1)/self.cov_strides)))/2),math.ceil(math.floor((self.trainx.shape[4]-((self.trainx.shape[4]-self.model_list[i][1][1]+1)/self.cov_strides)))/2),math.floor(math.floor((self.trainx.shape[3]-((self.trainx.shape[3]-self.model_list[i][1][2]+1)/self.cov_strides)))/2),math.ceil(math.floor((self.trainx.shape[3]-((self.trainx.shape[3]-self.model_list[i][1][2]+1)/self.cov_strides)))/2))), batch_first=True)', globals(), self.__dict__)
                        exec('self.conv'+str(i+1)+'=TimeDistributed(Conv2d(in_channels,self.model_list[i][1][0],(self.model_list[i][1][1],self.model_list[i][1][2]),stride=self.cov_strides,padding=self.cov_padding), batch_first=True)', globals(), self.__dict__)
                    exec('self.convs'+str(i+1)+'=TimeDistributed(Conv2d(in_channels,self.model_list[i][1][0],(1,1),stride=1,padding=self.cov_padding), batch_first=True)', globals(), self.__dict__)
                    exec('in_channels=self.model_list[i][1][0]', globals(), self.__dict__)
                    for j in range(self.residual_deep):
                        if self.residual_norm=='batchnormalization':
                            exec('self.norm'+str(i+1)+'_'+str(j+1)+'=TimeDistributed(BatchNorm2d(in_channels), batch_first=True)', globals(), self.__dict__)
                        elif self.residual_norm=='layernormalization':
                            exec('self.norm'+str(i+1)+'_'+str(j+1)+'=LayerNorm(in_channels)', globals(), self.__dict__)
                        if self.activation=='elu':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=ELU()', globals(), self.__dict__)
                        elif self.activation=='leakyrelu':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=LeakyReLU()', globals(), self.__dict__)
                        elif self.activation=='prelu':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=PReLU()', globals(), self.__dict__)
                        elif self.activation=='relu':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=ReLU()', globals(), self.__dict__)
                        elif self.activation=='sigmoid':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=Sigmoid()', globals(), self.__dict__)
                        elif self.activation=='tanh':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=Tanh()', globals(), self.__dict__)
                        elif self.activation=='softmax':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=Softmax()', globals(), self.__dict__)
                        if cov_padding=='same':
                            exec('self.pad'+str(i+1)+'_'+str(j+1)+'=TimeDistributed(ZeroPad2d(padding=(math.floor(math.floor((self.weight-(self.weight/self.cov_strides)))/2),math.ceil(math.floor((self.weight-(self.weight/self.cov_strides)))/2),math.floor(math.floor((self.hight-(self.hight/self.cov_strides)))/2),math.ceil(math.floor((self.hight-(self.hight/self.cov_strides)))/2))), batch_first=True)', globals(), self.__dict__)
                            exec('self.conv'+str(i+1)+'_'+str(j+1)+'=TimeDistributed(Conv2d(in_channels,self.model_list[i][1][0],(self.model_list[i][1][1],self.model_list[i][1][2]),stride=self.cov_strides,padding=self.cov_padding), batch_first=True)', globals(), self.__dict__)
                        else:
                            exec('self.pad'+str(i+1)+'_'+str(j+1)+'=TimeDistributed(ZeroPad2d(padding=(math.floor(math.floor((self.weight-((self.weight-self.model_list[i][1][1]+1)/self.cov_strides)))/2),math.ceil(math.floor((self.weight-((self.weight-self.model_list[i][1][1]+1)/self.cov_strides)))/2),math.floor(math.floor((self.hight-((self.hight-self.model_list[i][1][2]+1)/self.cov_strides)))/2),math.ceil(math.floor((self.hight-((self.hight-self.model_list[i][1][2]+1)/self.cov_strides)))/2))), batch_first=True)', globals(), self.__dict__)
                            exec('self.conv'+str(i+1)+'_'+str(j+1)+'=TimeDistributed(Conv2d(in_channels,self.model_list[i][1][0],(self.model_list[i][1][1],self.model_list[i][1][2]),stride=self.cov_strides,padding=self.cov_padding), batch_first=True)', globals(), self.__dict__)
                    exec('self.add'+str(i+1)+'=torch.add', globals(), self.__dict__)
                elif self.model_list[i][0] == 'resnet' and timestep==None:
                    if cov_padding=='same':
                        exec('self.pad'+str(i+1)+'=ZeroPad2d(padding=(math.floor(math.floor((self.trainx.shape[3]-(self.trainx.shape[3]/self.cov_strides)))/2),math.ceil(math.floor((self.trainx.shape[3]-(self.trainx.shape[3]/self.cov_strides)))/2),math.floor(math.floor((self.trainx.shape[2]-(self.trainx.shape[2]/self.cov_strides)))/2),math.ceil(math.floor((self.trainx.shape[2]-(self.trainx.shape[2]/self.cov_strides)))/2)))', globals(), self.__dict__)
                        exec('self.conv'+str(i+1)+'=Conv2d(in_channels,self.model_list[i][1][0],(self.model_list[i][1][1],self.model_list[i][1][2]),stride=self.cov_strides,padding=self.cov_padding)', globals(), self.__dict__)
                    else:
                        exec('self.pad'+str(i+1)+'=ZeroPad2d(padding=(math.floor(math.floor((self.trainx.shape[3]-((self.trainx.shape[3]-self.model_list[i][1][1]+1)/self.cov_strides)))/2),math.ceil(math.floor((self.trainx.shape[3]-((self.trainx.shape[3]-self.model_list[i][1][1]+1)/self.cov_strides)))/2),math.floor(math.floor((self.trainx.shape[2]-((self.trainx.shape[2]-self.model_list[i][1][2]+1)/self.cov_strides)))/2),math.ceil(math.floor((self.trainx.shape[2]-((self.trainx.shape[2]-self.model_list[i][1][2]+1)/self.cov_strides)))/2)))', globals(), self.__dict__)
                        exec('self.conv'+str(i+1)+'=Conv2d(in_channels,self.model_list[i][1][0],(self.model_list[i][1][1],self.model_list[i][1][2]),stride=self.cov_strides,padding=self.cov_padding)', globals(), self.__dict__)
                    exec('self.convs'+str(i+1)+'=Conv2d(in_channels,self.model_list[i][1][0],(1,1),stride=1,padding=self.cov_padding)', globals(), self.__dict__)
                    exec('in_channels=self.model_list[i][1][0]', globals(), self.__dict__)
                    for j in range(self.residual_deep):
                        if self.residual_norm=='batchnormalization':
                            exec('self.norm'+str(i+1)+'_'+str(j+1)+'=BatchNorm2d(in_channels)', globals(), self.__dict__)
                        elif self.residual_norm=='layernormalization':
                            exec('self.norm'+str(i+1)+'_'+str(j+1)+'=LayerNorm(in_channels)', globals(), self.__dict__)
                        if self.activation=='elu':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=ELU()', globals(), self.__dict__)
                        elif self.activation=='leakyrelu':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=LeakyReLU()', globals(), self.__dict__)
                        elif self.activation=='prelu':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=PReLU()', globals(), self.__dict__)
                        elif self.activation=='relu':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=ReLU()', globals(), self.__dict__)
                        elif self.activation=='sigmoid':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=Sigmoid()', globals(), self.__dict__)
                        elif self.activation=='tanh':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=Tanh()', globals(), self.__dict__)
                        elif self.activation=='softmax':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=Softmax()', globals(), self.__dict__)
                        if cov_padding=='same':
                            exec('self.pad'+str(i+1)+'_'+str(j+1)+'=ZeroPad2d(padding=(math.floor(math.floor((self.weight-(self.weight/self.cov_strides)))/2),math.ceil(math.floor((self.weight-(self.weight/self.cov_strides)))/2),math.floor(math.floor((self.hight-(self.hight/self.cov_strides)))/2),math.ceil(math.floor((self.hight-(self.hight/self.cov_strides)))/2)))', globals(), self.__dict__)
                            exec('self.conv'+str(i+1)+'_'+str(j+1)+'=Conv2d(in_channels,self.model_list[i][1][0],(self.model_list[i][1][1],self.model_list[i][1][2]),stride=self.cov_strides,padding=self.cov_padding)', globals(), self.__dict__)
                        else:
                            exec('self.pad'+str(i+1)+'_'+str(j+1)+'=ZeroPad2d(padding=(math.floor(math.floor((self.weight-((self.weight-self.model_list[i][1][1]+1)/self.cov_strides)))/2),math.ceil(math.floor((self.weight-((self.weight-self.model_list[i][1][1]+1)/self.cov_strides)))/2),math.floor(math.floor((self.hight-((self.hight-self.model_list[i][1][2]+1)/self.cov_strides)))/2),math.ceil(math.floor((self.hight-((self.hight-self.model_list[i][1][2]+1)/self.cov_strides)))/2)))', globals(), self.__dict__)
                            exec('self.conv'+str(i+1)+'_'+str(j+1)+'=Conv2d(in_channels,self.model_list[i][1][0],(self.model_list[i][1][1],self.model_list[i][1][2]),stride=self.cov_strides,padding=self.cov_padding)', globals(), self.__dict__)
                    exec('self.add'+str(i+1)+'=torch.add', globals(), self.__dict__)
                elif self.model_list[i][0] == 'vit' and timestep!=None:
                    if self.vit_patch_size==None:
                        self.new_hights,self.new_weights,self.vit_patch_size=find_best_common_divisor(self.hight,self.weight)
                    else:
                        self.new_hights,self.new_weights=find_new_a_b(self.hight,self.weight,self.vit_patch_size)
                    exec('self.vit'+str(i+1)+'=TimeDistributed(ViT(image_size = (self.new_hights,self.new_weights),patch_size = self.vit_patch_size,num_classes = (self.model_list[i][1]*self.hight*self.weight),dim = self.vit_dim_feedforward,depth = self.vit_deep,heads = self.vit_num_heads,mlp_dim = self.vit_dim_feedforward,channels=in_channels,dropout = self.dropout), batch_first=True)', globals(), self.__dict__)
                    exec('in_channels=self.model_list[i][1]', globals(), self.__dict__)
                elif self.model_list[i][0] == 'swin' and timestep!=None:
                    if self.vit_patch_size==None:
                        self.new_hights,self.new_weights,self.vit_patch_size=find_best_common_divisor(self.hight,self.weight)
                    else:
                        self.new_hights,self.new_weights=find_new_a_b(self.hight,self.weight,self.vit_patch_size)
                    exec('self.swin'+str(i+1)+'=TimeDistributed(SwinTransformer(img_size = (self.new_hights,self.new_weights),patch_size = self.vit_patch_size,num_classes = (self.model_list[i][1]*self.hight*self.weight),window_size=self.model_list[i][2],embed_dim = 96,depths = self.swin_deep,num_heads = self.swin_num_heads,in_chans=in_channels,drop_rate= self.dropout), batch_first=True)', globals(), self.__dict__)
                    exec('in_channels=self.model_list[i][1]', globals(), self.__dict__)
                elif self.model_list[i][0] == 'vit' and timestep==None:
                    if self.vit_patch_size==None:
                        self.new_hights,self.new_weights,self.vit_patch_size=find_best_common_divisor(self.hight,self.weight)
                    else:
                        self.new_hights,self.new_weights=find_new_a_b(self.hight,self.weight,self.vit_patch_size)
                    exec('self.vit'+str(i+1)+'=ViT(image_size = (self.new_hights,self.new_weights),patch_size = self.vit_patch_size,num_classes = (self.model_list[i][1]*self.hight*self.weight),dim = self.vit_dim_feedforward,depth = self.vit_deep,heads = self.vit_num_heads,mlp_dim = self.vit_dim_feedforward,channels=in_channels,dropout = self.dropout)', globals(), self.__dict__)
                    exec('in_channels=self.model_list[i][1]', globals(), self.__dict__)
                elif self.model_list[i][0] == 'swin' and timestep==None:
                    if self.vit_patch_size==None:
                        self.new_hights,self.new_weights,self.vit_patch_size=find_best_common_divisor(self.hight,self.weight)
                    else:
                        self.new_hights,self.new_weights=find_new_a_b(self.hight,self.weight,self.vit_patch_size)
                    exec('self.swin'+str(i+1)+'=SwinTransformer(img_size = (self.new_hights,self.new_weights),patch_size = self.vit_patch_size,num_classes = (self.model_list[i][1]*self.hight*self.weight),window_size=self.model_list[i][2],embed_dim = 96,depths = self.swin_deep,num_heads = self.swin_num_heads,in_chans=in_channels,drop_rate= self.dropout)', globals(), self.__dict__)
                    exec('in_channels=self.model_list[i][1]', globals(), self.__dict__)
                elif self.model_list[i][0] == 'cbam' and timestep!=None:
                    exec('self.cbam_max_pool'+str(i+1)+'=TimeDistributed(AdaptiveMaxPool2d(1), batch_first=True)', globals(), self.__dict__)
                    exec('self.cbam_ave_pool'+str(i+1)+'=TimeDistributed(AdaptiveAvgPool2d(1), batch_first=True)', globals(), self.__dict__)
                    exec('self.cbam_conv'+str(i+1)+'_1=TimeDistributed(Conv2d(in_channels,max(1, in_channels // cbam_reduction),1,bias=False), batch_first=True)', globals(), self.__dict__)
                    exec('self.cbam_act'+str(i+1)+'_1=ReLU()', globals(), self.__dict__)
                    exec('self.cbam_conv'+str(i+1)+'_2=TimeDistributed(Conv2d(max(1, in_channels // cbam_reduction),in_channels,1,bias=False), batch_first=True)', globals(), self.__dict__)
                    exec('self.cbam_act'+str(i+1)+'_2=Sigmoid()', globals(), self.__dict__)
                    exec('self.cbam_conv'+str(i+1)+'_3=TimeDistributed(Conv2d(2,1,kernel_size=7,padding=7 // 2,bias=False), batch_first=True)', globals(), self.__dict__)
                elif self.model_list[i][0] == 'cbam' and timestep ==None:
                    exec('self.cbam_max_pool'+str(i+1)+'=AdaptiveMaxPool2d(1)', globals(), self.__dict__)
                    exec('self.cbam_ave_pool'+str(i+1)+'=AdaptiveAvgPool2d(1)', globals(), self.__dict__)
                    exec('self.cbam_conv'+str(i+1)+'_1=Conv2d(in_channels,max(1, in_channels // cbam_reduction),1,bias=False)', globals(), self.__dict__)
                    exec('self.cbam_act'+str(i+1)+'_1=ReLU()', globals(), self.__dict__)
                    exec('self.cbam_conv'+str(i+1)+'_2=Conv2d(max(1, in_channels // cbam_reduction),in_channels,1,bias=False)', globals(), self.__dict__)
                    exec('self.cbam_act'+str(i+1)+'_2=Sigmoid()', globals(), self.__dict__)
                    exec('self.cbam_conv'+str(i+1)+'_3=Conv2d(2,1,kernel_size=7,padding=7 // 2,bias=False)', globals(), self.__dict__)
                elif self.model_list[i][0] == 'maxpooling' and timestep!=None:
                    if ifmulti_scale=='yes':
                        exec('self.fla'+str(i+1)+'=Flatten(start_dim=2, end_dim=-1)', globals(), self.__dict__)
                        exec('self.conc'+str(i+1)+'=torch.concatenate', globals(), self.__dict__)
                        exec('self.dense_input_len=self.dense_input_len+self.hight*self.weight*in_channels', globals(), self.__dict__)
                    self.hight=1+math.floor((self.hight-self.model_list[i][1][0])/self.pooling_strides)
                    self.weight=1+math.floor((self.weight-self.model_list[i][1][1])/self.pooling_strides)
                    if self.hight < 1 or self.weight < 1:
                        print('池化层数过多')
                        break
                    exec('self.pool'+str(i+1)+'=TimeDistributed(MaxPool2d((self.model_list[i][1][0],self.model_list[i][1][1]),stride=self.pooling_strides), batch_first=True)', globals(), self.__dict__)
                elif self.model_list[i][0] == 'maxpooling' and timestep==None:
                    if ifmulti_scale=='yes':
                        exec('self.fla'+str(i+1)+'=Flatten()', globals(), self.__dict__)
                        exec('self.conc'+str(i+1)+'=torch.concatenate', globals(), self.__dict__)
                        exec('self.dense_input_len=self.dense_input_len+self.hight*self.weight*in_channels', globals(), self.__dict__)
                    self.hight=1+math.floor((self.hight-self.model_list[i][1][0])/self.pooling_strides)
                    self.weight=1+math.floor((self.weight-self.model_list[i][1][1])/self.pooling_strides)
                    if self.hight < 1 or self.weight < 1:
                        print('池化层数过多')
                        break
                    exec('self.pool'+str(i+1)+'=MaxPool2d((self.model_list[i][1][0],self.model_list[i][1][1]),stride=self.pooling_strides)', globals(), self.__dict__)
                elif self.model_list[i][0] == 'avepooling' and timestep!=None:
                    if ifmulti_scale=='yes':
                        exec('self.fla'+str(i+1)+'=Flatten(start_dim=2, end_dim=-1)', globals(), self.__dict__)
                        exec('self.conc'+str(i+1)+'=torch.concatenate', globals(), self.__dict__)
                        exec('self.dense_input_len=self.dense_input_len+self.hight*self.weight*in_channels', globals(), self.__dict__)
                    self.hight=1+math.floor((self.hight-self.model_list[i][1][0])/self.pooling_strides)
                    self.weight=1+math.floor((self.weight-self.model_list[i][1][1])/self.pooling_strides)
                    if self.hight < 1 or self.weight < 1:
                        print('池化层数过多')
                        break
                    exec('self.pool'+str(i+1)+'=TimeDistributed(AvgPool2d((self.model_list[i][1][0],self.model_list[i][1][1]),stride=self.pooling_strides), batch_first=True)', globals(), self.__dict__)
                elif self.model_list[i][0] == 'avepooling' and timestep==None:
                    if ifmulti_scale=='yes':
                        exec('self.fla'+str(i+1)+'=Flatten()', globals(), self.__dict__)
                        exec('self.conc'+str(i+1)+'=torch.concatenate', globals(), self.__dict__)
                        exec('self.dense_input_len=self.dense_input_len+self.hight*self.weight*in_channels', globals(), self.__dict__)
                    self.hight=1+math.floor((self.hight-self.model_list[i][1][0])/self.pooling_strides)
                    self.weight=1+math.floor((self.weight-self.model_list[i][1][1])/self.pooling_strides)
                    if self.hight < 1 or self.weight < 1:
                        print('池化层数过多')
                        break
                    exec('self.pool'+str(i+1)+'=AvgPool2d((self.model_list[i][1][0],self.model_list[i][1][1]),stride=self.pooling_strides)', globals(), self.__dict__)
                elif self.model_list[i][0] == 'batchnormalization':
                    exec('self.norm'+str(i+1)+'=FlexibleBatchNorm(in_channels)', globals(), self.__dict__)
                elif self.model_list[i][0] == 'layernormalization':
                    exec('self.norm'+str(i+1)+'=FlexibleLayerNorm(in_channels)', globals(), self.__dict__)
                elif self.model_list[i][0] == 'activation':
                    if self.model_list[i][1]=='elu':
                        exec('self.act'+str(i+1)+'=ELU()', globals(), self.__dict__)
                    elif self.model_list[i][1]=='leakyrelu':
                        exec('self.act'+str(i+1)+'=LeakyReLU()', globals(), self.__dict__)
                    elif self.model_list[i][1]=='prelu':
                        exec('self.act'+str(i+1)+'=PReLU()', globals(), self.__dict__)
                    elif self.model_list[i][1]=='relu':
                        exec('self.act'+str(i+1)+'=ReLU()', globals(), self.__dict__)
                    elif self.model_list[i][1]=='sigmoid':
                        exec('self.act'+str(i+1)+'=Sigmoid()', globals(), self.__dict__)
                    elif self.model_list[i][1]=='tanh':
                        exec('self.act'+str(i+1)+'=Tanh()', globals(), self.__dict__)
                    elif self.model_list[i][1]=='softmax':
                        exec('self.act'+str(i+1)+'=Softmax()', globals(), self.__dict__)
                elif self.model_list[i][0] == 'gc':
                    if self.graph_station_sequence_mode:
                        exec('graph_n_nodes = self.station_num', globals(), self.__dict__)
                        exec('graph_in_channels = in_channels', globals(), self.__dict__)
                    elif trainx.ndim > 3:
                        exec('graph_n_nodes = self.hight * self.weight', globals(), self.__dict__)
                        exec('graph_in_channels = in_channels', globals(), self.__dict__)
                    elif trainx.ndim == 3 and timestep is None:
                        exec('graph_n_nodes = self.trainx.shape[2]', globals(), self.__dict__)
                        exec('graph_in_channels = in_channels', globals(), self.__dict__)
                    else:
                        raise ValueError("gc 目前只支持格点输入 (B,T,C,H,W)/(B,C,H,W) 或站点/一维空间输入 (B,C,N)。")

                    if self.adj_prior is None:
                        exec('graph_adj_prior = np.ones((graph_n_nodes, graph_n_nodes), dtype=np.float32)', globals(), self.__dict__)
                        exec('graph_adj_prior = graph_adj_prior / (graph_adj_prior.sum(axis=1, keepdims=True) + 1e-12)', globals(), self.__dict__)
                    else:
                        exec('graph_adj_prior = self.adj_prior', globals(), self.__dict__)


                    exec(
                        'self.gc'+str(i+1)+'=LearnableGCNGraph(graph_in_channels, self.model_list[i][1], graph_n_nodes, adj_prior=graph_adj_prior)',
                        globals(), self.__dict__
                    )
                    exec('in_channels=self.model_list[i][1]', globals(), self.__dict__)

                elif self.model_list[i][0] == 'ga':
                    if self.graph_station_sequence_mode:
                        exec('graph_n_nodes = self.station_num', globals(), self.__dict__)
                        exec('graph_in_channels = in_channels', globals(), self.__dict__)
                    elif trainx.ndim > 3:
                        exec('graph_n_nodes = self.hight * self.weight', globals(), self.__dict__)
                        exec('graph_in_channels = in_channels', globals(), self.__dict__)
                    elif trainx.ndim == 3 and timestep is None:
                        exec('graph_n_nodes = self.trainx.shape[2]', globals(), self.__dict__)
                        exec('graph_in_channels = in_channels', globals(), self.__dict__)
                    else:
                        raise ValueError("ga 目前只支持格点输入 (B,T,C,H,W)/(B,C,H,W) 或站点/一维空间输入 (B,C,N)。")

                    if self.adj_prior is None:
                        exec('graph_adj_prior = np.ones((graph_n_nodes, graph_n_nodes), dtype=np.float32)', globals(), self.__dict__)
                        exec('graph_adj_prior = graph_adj_prior / (graph_adj_prior.sum(axis=1, keepdims=True) + 1e-12)', globals(), self.__dict__)
                    else:
                        exec('graph_adj_prior = self.adj_prior', globals(), self.__dict__)


                    exec(
                        'self.gc'+str(i+1)+'=DenseGATGraph(graph_in_channels, self.model_list[i][1], graph_n_nodes, adj_prior=graph_adj_prior, heads=self.transformer_num_heads, dropout=self.dropout)',
                        globals(), self.__dict__
                    )
                    exec('in_channels=self.model_list[i][1]', globals(), self.__dict__)
                elif self.model_list[i][0] == 'chebnet':
                    if self.graph_station_sequence_mode:
                        exec('graph_n_nodes = self.station_num', globals(), self.__dict__)
                        exec('graph_in_channels = in_channels', globals(), self.__dict__)
                    elif trainx.ndim > 3:
                        exec('graph_n_nodes = self.hight * self.weight', globals(), self.__dict__)
                        exec('graph_in_channels = in_channels', globals(), self.__dict__)
                    elif trainx.ndim == 3 and timestep is None:
                        exec('graph_n_nodes = self.trainx.shape[2]', globals(), self.__dict__)
                        exec('graph_in_channels = in_channels', globals(), self.__dict__)
                    else:
                        raise ValueError("chebnet 目前只支持格点输入 (B,T,C,H,W)/(B,C,H,W) 或站点/一维空间输入 (B,C,N)。")

                    if self.adj_prior is None:
                        exec('graph_adj_prior = np.ones((graph_n_nodes, graph_n_nodes), dtype=np.float32)', globals(), self.__dict__)
                        exec('graph_adj_prior = graph_adj_prior / (graph_adj_prior.sum(axis=1, keepdims=True) + 1e-12)', globals(), self.__dict__)
                    else:
                        exec('graph_adj_prior = self.adj_prior', globals(), self.__dict__)

                    cheb_k_layer = int(self.model_list[i][2]) if len(self.model_list[i]) > 2 else int(self.chebnet_k)
                    self.__dict__['cheb_k_layer'] = cheb_k_layer
                    exec(
                        'self.gc'+str(i+1)+'=DenseChebNetGraph(graph_in_channels, self.model_list[i][1], graph_n_nodes, cheb_k=cheb_k_layer, adj_prior=graph_adj_prior)',
                        globals(), self.__dict__
                    )
                    exec('in_channels=self.model_list[i][1]', globals(), self.__dict__)

                elif self.model_list[i][0] == 'gsage':
                    if self.graph_station_sequence_mode:
                        exec('graph_n_nodes = self.station_num', globals(), self.__dict__)
                        exec('graph_in_channels = in_channels', globals(), self.__dict__)
                    elif trainx.ndim > 3:
                        exec('graph_n_nodes = self.hight * self.weight', globals(), self.__dict__)
                        exec('graph_in_channels = in_channels', globals(), self.__dict__)
                    elif trainx.ndim == 3 and timestep is None:
                        exec('graph_n_nodes = self.trainx.shape[2]', globals(), self.__dict__)
                        exec('graph_in_channels = in_channels', globals(), self.__dict__)
                    else:
                        raise ValueError("gsage 目前只支持格点输入 (B,T,C,H,W)/(B,C,H,W) 或站点/一维空间输入 (B,C,N)。")

                    if self.adj_prior is None:
                        exec('graph_adj_prior = np.ones((graph_n_nodes, graph_n_nodes), dtype=np.float32)', globals(), self.__dict__)
                        exec('graph_adj_prior = graph_adj_prior / (graph_adj_prior.sum(axis=1, keepdims=True) + 1e-12)', globals(), self.__dict__)
                    else:
                        exec('graph_adj_prior = self.adj_prior', globals(), self.__dict__)

                    gsage_aggr_layer = str(self.model_list[i][2]).lower() if len(self.model_list[i]) > 2 else str(self.gsage_aggr).lower()
                    gsage_lstm_units_layer = self.model_list[i][3] if len(self.model_list[i]) > 3 else self.gsage_lstm_units
                    self.__dict__['gsage_aggr_layer'] = gsage_aggr_layer
                    self.__dict__['gsage_lstm_units_layer'] = gsage_lstm_units_layer
                    exec(
                        'self.gc'+str(i+1)+'=DenseGraphSAGEGraph(graph_in_channels, self.model_list[i][1], graph_n_nodes, adj_prior=graph_adj_prior, aggr=gsage_aggr_layer, lstm_units=gsage_lstm_units_layer)',
                        globals(), self.__dict__
                    )
                    exec('in_channels=self.model_list[i][1]', globals(), self.__dict__)

                elif trainx.ndim>3 and self.model_list[i][0] == 'flatten' and timestep!=None:
                    exec('self.fla'+str(i+1)+'=Flatten(start_dim=2, end_dim=-1)', globals(), self.__dict__)
                    self.flattened=True
                    if self.one_d_spatial_sequence_mode:
                        current_flat_len = self.station_num * self.__dict__['in_channels']
                    else:
                        current_flat_len = self.hight * self.weight * self.__dict__['in_channels']
                    self.__dict__['current_flat_len'] = current_flat_len
                    if ifmulti_scale=='yes':
                        exec('self.conc'+str(i+1)+'=torch.concatenate', globals(), self.__dict__)
                        exec('self.dense_input_len=self.dense_input_len+self.current_flat_len', globals(), self.__dict__)
                        exec('in_channels=self.dense_input_len', globals(), self.__dict__)
                    else:
                        exec('in_channels=self.current_flat_len', globals(), self.__dict__)
                elif trainx.ndim>3 and self.model_list[i][0] == 'flatten' and timestep==None:
                    exec('self.fla'+str(i+1)+'=Flatten()', globals(), self.__dict__)
                    self.flattened=True
                    if ifmulti_scale=='yes':
                        exec('self.conc'+str(i+1)+'=torch.concatenate', globals(), self.__dict__)
                        exec('self.dense_input_len=self.dense_input_len+self.hight*self.weight*in_channels', globals(), self.__dict__)
                        exec('in_channels=self.dense_input_len', globals(), self.__dict__)
                    else:
                        exec('in_channels=self.hight*self.weight*in_channels', globals(), self.__dict__)
                elif trainx.ndim<=3 and self.model_list[i][0] == 'flatten':
                    exec('self.fla'+str(i+1)+'=Flatten()', globals(), self.__dict__)
                    self.flattened=True
                    if trainx.ndim == 3:
                        exec('in_channels=self.trainx.shape[2]*in_channels', globals(), self.__dict__)
                elif self.model_list[i][0] =='fc':
                    exec('self.fc'+str(i+1)+'=Linear(in_channels,self.model_list[i][1])', globals(), self.__dict__)
                    exec('in_channels=self.model_list[i][1]', globals(), self.__dict__)
                elif self.model_list[i][0] == 'dropout':
                    exec('self.drop'+str(i+1)+'=Dropout(self.model_list[i][1])', globals(), self.__dict__)
                elif self.model_list[i][0] == 'transformer':
                    exec('self.embedding'+str(i+1)+'=Embedding(self.timestep+1,in_channels)', globals(), self.__dict__)
                    exec('self.transformer_add'+str(i+1)+'=torch.add', globals(), self.__dict__)
                    exec('self.transformer'+str(i+1)+'=TransformerEncoder(TransformerEncoderLayer(in_channels,self.transformer_num_heads,dim_feedforward=self.transformer_dim_feedforward,dropout=self.dropout,activation=self.activation, batch_first=True),self.transformer_encoder_deep)', globals(), self.__dict__)
                    self.nlp_indices.append(i)
                    self.nlp_out_channels[i]=self.__dict__['in_channels']
                elif self.model_list[i][0] == 'mamba' or self.model_list[i][0] == 'mamba2':
                    exec('self.mamba_inner = mamba_expand * in_channels', globals(), self.__dict__)
                    exec('self.mamba_ssm = None', globals(), self.__dict__)
                    exec('self.mamba_headdim = 64 if mamba_inner % 64 == 0 else \
                              32 if mamba_inner % 32 == 0 else \
                              16 if mamba_inner % 16 == 0 else \
                               8 if mamba_inner % 8 == 0 else \
                               4 if mamba_inner % 4 == 0 else \
                               2 if mamba_inner % 2 == 0 else 1', globals(), self.__dict__)
                    if self.model_list[i][0] == 'mamba':
                        exec('self.mamba'+str(i+1)+'=Mamba(d_model=in_channels, d_state=self.mamba_state, d_conv=self.mamba_conv, expand=self.mamba_expand)', globals(), self.__dict__)
                    elif self.model_list[i][0] == 'mamba2':
                        exec('self.mamba'+str(i+1)+'=Mamba2(d_model=in_channels, d_state=self.mamba_state, d_conv=self.mamba_conv, expand=self.mamba_expand ,headdim=self.mamba_headdim, d_ssm=self.mamba_ssm,use_mem_eff_path=False)', globals(), self.__dict__)
                    self.nlp_indices.append(i)
                    self.nlp_out_channels[i]=self.__dict__['in_channels']
                elif self.model_list[i][0] == 'tcn':
                    exec('self.tcn'+str(i+1)+'=TCN(in_channels,self.num_channels,kernel_size=self.tcn_kernel_size,dilations=self.dilations,dropout=self.dropout,activation=self.activation,input_shape="NLC")', globals(), self.__dict__)
                    exec('in_channels=self.num_channels[-1]', globals(), self.__dict__)
                    self.nlp_indices.append(i)
                    self.nlp_out_channels[i]=self.__dict__['in_channels']
                elif self.model_list[i][0] == 'lstm':
                    exec('self.lstm'+str(i+1)+'=LSTM(in_channels,self.nlp_units,num_layers=self.nlp_num_layers,batch_first=True,dropout=self.dropout)', globals(), self.__dict__)
                    exec('in_channels=self.nlp_units', globals(), self.__dict__)
                    self.nlp_indices.append(i)
                    self.nlp_out_channels[i]=self.__dict__['in_channels']
                elif self.model_list[i][0] == 'rnn':
                    exec('self.rnn'+str(i+1)+'=RNN(in_channels,self.nlp_units,num_layers=self.nlp_num_layers,batch_first=True,dropout=self.dropout)', globals(), self.__dict__)
                    exec('in_channels=self.nlp_units', globals(), self.__dict__)
                    self.nlp_indices.append(i)
                    self.nlp_out_channels[i]=self.__dict__['in_channels']
                elif self.model_list[i][0] == 'ltc':
                    exec('self.ltc'+str(i+1)+'=LTCLayer(in_channels,self.nlp_units,num_layers=self.nlp_num_layers,solver=self.ltc_solver,ode_unfolds=self.ltc_ode_unfolds,dt=self.ltc_dt,tau_min=self.ltc_tau_min,activation=self.ltc_activation,dropout=self.ltc_dropout)', globals(), self.__dict__)
                    exec('in_channels=self.nlp_units', globals(), self.__dict__)
                    self.nlp_indices.append(i)
                    self.nlp_out_channels[i]=self.__dict__['in_channels']
                elif self.model_list[i][0] == 'gru':
                    exec('self.gru'+str(i+1)+'=GRU(in_channels,self.nlp_units,num_layers=self.nlp_num_layers,batch_first=True,dropout=self.dropout)', globals(), self.__dict__)
                    exec('in_channels=self.nlp_units', globals(), self.__dict__)
                    self.nlp_indices.append(i)
                    self.nlp_out_channels[i]=self.__dict__['in_channels']
            if self.seq2seq_mode:
                if len(self.nlp_indices) == 0:
                    raise ValueError("outputstep 不为 None 时，model_list 中必须至少包含一个 NLP/序列模块。")
                self.last_nlp_index = self.nlp_indices[-1]
                self.first_head_index = self.last_nlp_index + 1
                dec_in_channels = int(self.nlp_out_channels[self.last_nlp_index])
                for decoder_order, nlp_i in enumerate(self.nlp_indices):
                    nlp_name = self.model_list[nlp_i][0]
                    self.__dict__['dec_in_channels'] = dec_in_channels
                    if nlp_name == 'transformer':
                        exec('self.decoder_embedding'+str(nlp_i+1)+'=Embedding(self.outputstep+1,dec_in_channels)', globals(), self.__dict__)
                        exec('self.decoder_transformer'+str(nlp_i+1)+'=nn.TransformerDecoder(nn.TransformerDecoderLayer(dec_in_channels,self.transformer_num_heads,dim_feedforward=self.transformer_dim_feedforward,dropout=self.dropout,activation=self.activation,batch_first=True),self.transformer_encoder_deep)', globals(), self.__dict__)
                    elif nlp_name == 'mamba' or nlp_name == 'mamba2':
                        decoder_mamba_inner = self.mamba_expand * dec_in_channels
                        decoder_mamba_headdim = 64 if decoder_mamba_inner % 64 == 0 else 32 if decoder_mamba_inner % 32 == 0 else 16 if decoder_mamba_inner % 16 == 0 else 8 if decoder_mamba_inner % 8 == 0 else 4 if decoder_mamba_inner % 4 == 0 else 2 if decoder_mamba_inner % 2 == 0 else 1
                        self.__dict__['decoder_mamba_headdim'] = decoder_mamba_headdim
                        if nlp_name == 'mamba':
                            exec('self.decoder_mamba'+str(nlp_i+1)+'=Mamba(d_model=dec_in_channels,d_state=self.mamba_state,d_conv=self.mamba_conv,expand=self.mamba_expand)', globals(), self.__dict__)
                        else:
                            exec('self.decoder_mamba'+str(nlp_i+1)+'=Mamba2(d_model=dec_in_channels,d_state=self.mamba_state,d_conv=self.mamba_conv,expand=self.mamba_expand,headdim=self.decoder_mamba_headdim,d_ssm=None,use_mem_eff_path=False)', globals(), self.__dict__)
                    elif nlp_name == 'tcn':
                        exec('self.decoder_tcn'+str(nlp_i+1)+'=TCN(dec_in_channels,self.num_channels,kernel_size=self.tcn_kernel_size,dilations=self.dilations,dropout=self.dropout,activation=self.activation,input_shape="NLC")', globals(), self.__dict__)
                        dec_in_channels = self.num_channels[-1]
                    elif nlp_name == 'lstm':
                        exec('self.decoder_lstm'+str(nlp_i+1)+'=LSTM(dec_in_channels,self.nlp_units,num_layers=self.nlp_num_layers,batch_first=True,dropout=self.dropout)', globals(), self.__dict__)
                        dec_in_channels = self.nlp_units
                    elif nlp_name == 'rnn':
                        exec('self.decoder_rnn'+str(nlp_i+1)+'=RNN(dec_in_channels,self.nlp_units,num_layers=self.nlp_num_layers,batch_first=True,dropout=self.dropout)', globals(), self.__dict__)
                        dec_in_channels = self.nlp_units
                    elif nlp_name == 'ltc':
                        exec('self.decoder_ltc'+str(nlp_i+1)+'=LTCLayer(dec_in_channels,self.nlp_units,num_layers=self.nlp_num_layers,solver=self.ltc_solver,ode_unfolds=self.ltc_ode_unfolds,dt=self.ltc_dt,tau_min=self.ltc_tau_min,activation=self.ltc_activation,dropout=self.ltc_dropout)', globals(), self.__dict__)
                        dec_in_channels = self.nlp_units
                    elif nlp_name == 'gru':
                        exec('self.decoder_gru'+str(nlp_i+1)+'=GRU(dec_in_channels,self.nlp_units,num_layers=self.nlp_num_layers,batch_first=True,dropout=self.dropout)', globals(), self.__dict__)
                        dec_in_channels = self.nlp_units
            else:
                self.last_nlp_index = -1
                self.first_head_index = len(self.model_list)

        def forward(self, x, position=None):
            self.__dict__['x']=x
            self.__dict__['position']=position
            self.__dict__['self']=self
            try:
                exec('del model_conc', globals(), self.__dict__)
            except:
                pass
            if timestep!=None:
                def is_last_occurrence(target_string, full_nested_list, current_i):

                    if not isinstance(target_string, str):
                        return False

                    if not (0 <= current_i < len(full_nested_list)) or \
                       not isinstance(full_nested_list[current_i], list) or \
                       len(full_nested_list[current_i]) == 0:
                        return False

                    def search_in_segment(segment):
                        for item in segment:
                            if isinstance(item, list):
                                if search_in_segment(item):
                                    return True
                            elif isinstance(item, str):
                                if item == target_string:
                                    return True
                        return False

                    remaining_list_segment = full_nested_list[current_i + 1:]
                    return not search_in_segment(remaining_list_segment)
                def is_last_sequence_layer(full_nested_list, current_i):
                    seq_layer_names = ['transformer', 'lstm', 'rnn', 'gru', 'ltc', 'tcn', 'mamba', 'mamba2']
                    for item in full_nested_list[current_i + 1:]:
                        if isinstance(item, list) and len(item) > 0 and item[0] in seq_layer_names:
                            return False
                    return True

            if find_string_in_nested_list('vit', self.model_list) or find_string_in_nested_list('swin', self.model_list):
                def find_best_common_divisor(a, b):
                    def get_common_divisors(a, b):
                        """获取 a 和 b 的所有公约数"""
                        def gcd(x, y):
                            while y:
                                x, y = y, x % y
                            return x

                        common_gcd = gcd(a, b)
                        divisors = []
                        for i in range(2, int(common_gcd**0.5) + 1):
                            if common_gcd % i == 0:
                                divisors.append(i)
                                if i != common_gcd // i:
                                    divisors.append(common_gcd // i)
                        divisors.append(common_gcd)
                        return sorted(divisors)

                    def find_valid_divisor(a, b):
                        """寻找满足条件的公约数"""
                        divisors = get_common_divisors(a, b)
                        best_divisor = None
                        min_value = float('inf')

                        for d in divisors:
                            a_div, b_div = a // d, b // d
                            if d == 1 or a_div == 1 or b_div == 1:
                                continue
                            if a_div * b_div > 16:
                                value = np.abs((a_div - d)) + np.abs((b_div - d))
                                if value < min_value:
                                    min_value = value
                                    best_divisor = d
                        return best_divisor

                    def adjust_values(a, b):
                        """顺序尝试增加 a、b 或轮流增加 a 和 b"""
                        steps = 0
                        while True:
                            steps += 1

                            new_a = a + steps
                            divisor = find_valid_divisor(new_a, b)
                            if divisor:
                                return new_a, b, divisor

                            new_b = b + steps
                            divisor = find_valid_divisor(a, new_b)
                            if divisor:
                                return a, new_b, divisor

                            if (steps % 2) == 1:
                                new_a, new_b = a + steps, b + steps-1
                            else:
                                new_a, new_b = a + steps, b + steps
                            divisor = find_valid_divisor(new_a, new_b)
                            if divisor:
                                return new_a, new_b, divisor

                            if (steps % 2) == 1:
                                new_a, new_b = a + steps-1, b + steps
                            else:
                                new_a, new_b = a + steps, b + steps
                            divisor = find_valid_divisor(new_a, new_b)
                            if divisor:
                                return new_a, new_b, divisor
                    divisor = find_valid_divisor(a, b)
                    if divisor:
                        return a, b, divisor
                    else:
                        new_a, new_b, new_divisor = adjust_values(a, b)
                        return new_a, new_b, new_divisor
                def find_new_a_b(a, b, divisor):
                    """
                    找到新的 a 和 b，使得 divisor 成为它们的公约数。
                    """
                    def is_common_divisor(x, y, d):
                        return x % d == 0 and y % d == 0
                    if is_common_divisor(a, b, divisor):
                            return a, b
                    a1, b1 = a, b  # 只增加 a
                    a2, b2 = a, b  # 只增加 b
                    a3, b3 = a, b  # a 和 b 轮流增加
                    a4, b4 = a, b  # a 和 b 轮流增加

                    while True:
                        a1 += 1
                        if is_common_divisor(a1, b1, divisor):
                            return a1, b1

                        b2 += 1
                        if is_common_divisor(a2, b2, divisor):
                            return a2, b2

                        if (a3 + b3) % 2 == 0:  # 偶数轮，增加 a
                            a3 += 1
                        else:  # 奇数轮，增加 b
                            b3 += 1
                        if is_common_divisor(a3, b3, divisor):
                            return a3, b3
                        if (a4 + b4) % 2 == 0:  # 偶数轮，增加 b
                            b4 += 1
                        else:  # 奇数轮，增加 a
                            a4 += 1
                        if is_common_divisor(a4, b4, divisor):
                            return a4, b4
            if timestep!=None:
                exec('in_channels=self.trainx.shape[2]', globals(), self.__dict__)
            elif timestep==None:
                exec('in_channels=self.trainx.shape[1]', globals(), self.__dict__)
            if self.one_d_spatial_sequence_mode:
                self.station_num=self.trainx.shape[3]
            elif trainx.ndim>3 and timestep!=None:
                self.hight=self.trainx.shape[3]
                self.weight=self.trainx.shape[4]
            elif trainx.ndim>3 and timestep==None:
                self.hight=self.trainx.shape[2]
                self.weight=self.trainx.shape[3]
            for i in range((self.last_nlp_index + 1) if self.outputstep is not None else len(self.model_list)):
                self.__dict__['i']=i
                if self.model_list[i][0] == 'cov':
                    if i==0:
                        exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(x)', globals(), self.__dict__)
                    else:
                        if self.model_list[i-1][0]=='cov':
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_conv'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='resnet':
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_add'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='vit':
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_vit'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='swin':
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_swin'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='cbam':
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_cbam_out'+str(i)+'_2)', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='maxpooling' or self.model_list[i-1][0]=='avepooling':
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_pool'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='batchnormalization' or self.model_list[i-1][0]=='layernormalization':
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='activation' :
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='dropout' :
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                elif self.model_list[i][0] == 'resnet':
                    if i==0:
                        exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(x)', globals(), self.__dict__)
                        exec('model_pad'+str(i+1)+'=self.pad'+str(i+1)+'(model_conv'+str(i+1)+')', globals(), self.__dict__)
                        exec('model_convs'+str(i+1)+'=self.convs'+str(i+1)+'(x)', globals(), self.__dict__)
                        for j in range(self.residual_deep):
                            exec('model_norm'+str(i+1)+'_'+str(j+1)+'=self.norm'+str(i+1)+'_'+str(j+1)+'(model_pad'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_act'+str(i+1)+'_'+str(j+1)+'=self.act'+str(i+1)+'_'+str(j+1)+'(model_norm'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                            exec('model_conv'+str(i+1)+'_'+str(j+1)+'=self.conv'+str(i+1)+'_'+str(j+1)+'(model_act'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                            exec('model_pad'+str(i+1)+'_'+str(j+1)+'=self.pad'+str(i+1)+'_'+str(j+1)+'(model_conv'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                        exec('model_add'+str(i+1)+'=self.add'+str(i+1)+'(model_convs'+str(i+1)+',model_pad'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                    else:
                        if self.model_list[i-1][0]=='conv':
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_conv'+str(i)+')', globals(), self.__dict__)
                            exec('model_pad'+str(i+1)+'=self.pad'+str(i+1)+'(model_conv'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_convs'+str(i+1)+'=self.convs'+str(i+1)+'(model_conv'+str(i)+')', globals(), self.__dict__)
                            for j in range(self.residual_deep):
                                exec('model_norm'+str(i+1)+'_'+str(j+1)+'=self.norm'+str(i+1)+'_'+str(j+1)+'(model_pad'+str(i+1)+')', globals(), self.__dict__)
                                exec('model_act'+str(i+1)+'_'+str(j+1)+'=self.act'+str(i+1)+'_'+str(j+1)+'(model_norm'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_conv'+str(i+1)+'_'+str(j+1)+'=self.conv'+str(i+1)+'_'+str(j+1)+'(model_act'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_pad'+str(i+1)+'_'+str(j+1)+'=self.pad'+str(i+1)+'_'+str(j+1)+'(model_conv'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                            exec('model_add'+str(i+1)+'=self.add'+str(i+1)+'(model_convs'+str(i+1)+',model_pad'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='resnet':
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_add'+str(i)+')', globals(), self.__dict__)
                            exec('model_pad'+str(i+1)+'=self.pad'+str(i+1)+'(model_conv'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_convs'+str(i+1)+'=self.convs'+str(i+1)+'(model_add'+str(i)+')', globals(), self.__dict__)
                            for j in range(self.residual_deep):
                                exec('model_norm'+str(i+1)+'_'+str(j+1)+'=self.norm'+str(i+1)+'_'+str(j+1)+'(model_pad'+str(i+1)+')', globals(), self.__dict__)
                                exec('model_act'+str(i+1)+'_'+str(j+1)+'=self.act'+str(i+1)+'_'+str(j+1)+'(model_norm'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_conv'+str(i+1)+'_'+str(j+1)+'=self.conv'+str(i+1)+'_'+str(j+1)+'(model_act'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_pad'+str(i+1)+'_'+str(j+1)+'=self.pad'+str(i+1)+'_'+str(j+1)+'(model_conv'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                            exec('model_add'+str(i+1)+'=self.add'+str(i+1)+'(model_convs'+str(i+1)+',model_pad'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='vit':
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_vit'+str(i)+')', globals(), self.__dict__)
                            exec('model_pad'+str(i+1)+'=self.pad'+str(i+1)+'(model_conv'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_convs'+str(i+1)+'=self.convs'+str(i+1)+'(model_vit'+str(i)+')', globals(), self.__dict__)
                            for j in range(self.residual_deep):
                                exec('model_norm'+str(i+1)+'_'+str(j+1)+'=self.norm'+str(i+1)+'_'+str(j+1)+'(model_pad'+str(i+1)+')', globals(), self.__dict__)
                                exec('model_act'+str(i+1)+'_'+str(j+1)+'=self.act'+str(i+1)+'_'+str(j+1)+'(model_norm'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_conv'+str(i+1)+'_'+str(j+1)+'=self.conv'+str(i+1)+'_'+str(j+1)+'(model_act'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_pad'+str(i+1)+'_'+str(j+1)+'=self.pad'+str(i+1)+'_'+str(j+1)+'(model_conv'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                            exec('model_add'+str(i+1)+'=self.add'+str(i+1)+'(model_convs'+str(i+1)+',model_pad'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='swin':
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_swin'+str(i)+')', globals(), self.__dict__)
                            exec('model_pad'+str(i+1)+'=self.pad'+str(i+1)+'(model_conv'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_convs'+str(i+1)+'=self.convs'+str(i+1)+'(model_swin'+str(i)+')', globals(), self.__dict__)
                            for j in range(self.residual_deep):
                                exec('model_norm'+str(i+1)+'_'+str(j+1)+'=self.norm'+str(i+1)+'_'+str(j+1)+'(model_pad'+str(i+1)+')', globals(), self.__dict__)
                                exec('model_act'+str(i+1)+'_'+str(j+1)+'=self.act'+str(i+1)+'_'+str(j+1)+'(model_norm'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_conv'+str(i+1)+'_'+str(j+1)+'=self.conv'+str(i+1)+'_'+str(j+1)+'(model_act'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_pad'+str(i+1)+'_'+str(j+1)+'=self.pad'+str(i+1)+'_'+str(j+1)+'(model_conv'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                            exec('model_add'+str(i+1)+'=self.add'+str(i+1)+'(model_convs'+str(i+1)+',model_pad'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='cbam':
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_cbam_out'+str(i)+'_2)', globals(), self.__dict__)
                            exec('model_pad'+str(i+1)+'=self.pad'+str(i+1)+'(model_conv'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_convs'+str(i+1)+'=self.convs'+str(i+1)+'(model_cbam_out'+str(i)+'_2)', globals(), self.__dict__)
                            for j in range(self.residual_deep):
                                exec('model_norm'+str(i+1)+'_'+str(j+1)+'=self.norm'+str(i+1)+'_'+str(j+1)+'(model_pad'+str(i+1)+')', globals(), self.__dict__)
                                exec('model_act'+str(i+1)+'_'+str(j+1)+'=self.act'+str(i+1)+'_'+str(j+1)+'(model_norm'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_conv'+str(i+1)+'_'+str(j+1)+'=self.conv'+str(i+1)+'_'+str(j+1)+'(model_act'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_pad'+str(i+1)+'_'+str(j+1)+'=self.pad'+str(i+1)+'_'+str(j+1)+'(model_conv'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                            exec('model_add'+str(i+1)+'=self.add'+str(i+1)+'(model_convs'+str(i+1)+',model_pad'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='maxpooling' or self.model_list[i-1][0]=='avepooling':
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_pool'+str(i)+')', globals(), self.__dict__)
                            exec('model_pad'+str(i+1)+'=self.pad'+str(i+1)+'(model_conv'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_convs'+str(i+1)+'=self.pad'+str(i+1)+'(model_pool'+str(i)+')', globals(), self.__dict__)
                            for j in range(self.residual_deep):
                                exec('model_norm'+str(i+1)+'_'+str(j+1)+'=self.norm'+str(i+1)+'_'+str(j+1)+'(model_pad'+str(i+1)+')', globals(), self.__dict__)
                                exec('model_act'+str(i+1)+'_'+str(j+1)+'=self.act'+str(i+1)+'_'+str(j+1)+'(model_norm'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_conv'+str(i+1)+'_'+str(j+1)+'=self.conv'+str(i+1)+'_'+str(j+1)+'(model_act'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_pad'+str(i+1)+'_'+str(j+1)+'=self.pad'+str(i+1)+'_'+str(j+1)+'(model_conv'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                            exec('model_add'+str(i+1)+'=self.add'+str(i+1)+'(model_convs'+str(i+1)+',model_pad'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='batchnormalization' or self.model_list[i-1][0]=='layernormalization':
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            exec('model_pad'+str(i+1)+'=self.pad'+str(i+1)+'(model_conv'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_convs'+str(i+1)+'=self.convs'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            for j in range(self.residual_deep):
                                exec('model_norm'+str(i+1)+'_'+str(j+1)+'=self.norm'+str(i+1)+'_'+str(j+1)+'(model_pad'+str(i+1)+')', globals(), self.__dict__)
                                exec('model_act'+str(i+1)+'_'+str(j+1)+'=self.act'+str(i+1)+'_'+str(j+1)+'(model_norm'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_conv'+str(i+1)+'_'+str(j+1)+'=self.conv'+str(i+1)+'_'+str(j+1)+'(model_act'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_pad'+str(i+1)+'_'+str(j+1)+'=self.pad'+str(i+1)+'_'+str(j+1)+'(model_conv'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                            exec('model_add'+str(i+1)+'=self.add'+str(i+1)+'(model_convs'+str(i+1)+',model_pad'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='activation' :
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            exec('model_pad'+str(i+1)+'=self.pad'+str(i+1)+'(model_conv'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_convs'+str(i+1)+'=self.convs'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            for j in range(self.residual_deep):
                                exec('model_norm'+str(i+1)+'_'+str(j+1)+'=self.norm'+str(i+1)+'_'+str(j+1)+'(model_pad'+str(i+1)+')', globals(), self.__dict__)
                                exec('model_act'+str(i+1)+'_'+str(j+1)+'=self.act'+str(i+1)+'_'+str(j+1)+'(model_norm'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_conv'+str(i+1)+'_'+str(j+1)+'=self.conv'+str(i+1)+'_'+str(j+1)+'(model_act'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_pad'+str(i+1)+'_'+str(j+1)+'=self.pad'+str(i+1)+'_'+str(j+1)+'(model_conv'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                            exec('model_add'+str(i+1)+'=self.add'+str(i+1)+'(model_convs'+str(i+1)+',model_pad'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='dropout' :
                            exec('model_conv'+str(i+1)+'=self.conv'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                            exec('model_pad'+str(i+1)+'=self.pad'+str(i+1)+'(model_conv'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_convs'+str(i+1)+'=self.convs'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                            for j in range(self.residual_deep):
                                exec('model_norm'+str(i+1)+'_'+str(j+1)+'=self.norm'+str(i+1)+'_'+str(j+1)+'(model_pad'+str(i+1)+')', globals(), self.__dict__)
                                exec('model_act'+str(i+1)+'_'+str(j+1)+'=self.act'+str(i+1)+'_'+str(j+1)+'(model_norm'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_conv'+str(i+1)+'_'+str(j+1)+'=self.conv'+str(i+1)+'_'+str(j+1)+'(model_act'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                                exec('model_pad'+str(i+1)+'_'+str(j+1)+'=self.pad'+str(i+1)+'_'+str(j+1)+'(model_conv'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                            exec('model_add'+str(i+1)+'=self.add'+str(i+1)+'(model_convs'+str(i+1)+',model_pad'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                elif self.model_list[i][0] == 'vit':
                    if self.vit_patch_size==None:
                        self.new_hights,self.new_weights,self.vit_patch_size=find_best_common_divisor(self.hight,self.weight)
                    else:
                        self.new_hights,self.new_weights=find_new_a_b(self.hight,self.weight,self.vit_patch_size)
                    exec('in_channels=self.model_list[i][1]', globals(), self.__dict__)
                    if i==0:
                        exec('x = F.pad(x, (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                        exec('model_vit'+str(i+1)+'=self.vit'+str(i+1)+'(x)', globals(), self.__dict__)
                    else:
                        if self.model_list[i-1][0]=='conv':
                            exec('model_conv'+str(i)+' = F.pad(model_conv'+str(i)+', (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_vit'+str(i+1)+'=self.vit'+str(i+1)+'(model_conv'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='resnet':
                            exec('model_add'+str(i)+' = F.pad(model_add'+str(i)+', (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_vit'+str(i+1)+'=self.vit'+str(i+1)+'(model_add'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='vit':
                            exec('model_vit'+str(i)+' = F.pad(model_vit'+str(i)+', (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_vit'+str(i+1)+'=self.vit'+str(i+1)+'(model_vit'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='swin':
                            exec('model_swin'+str(i)+' = F.pad(model_swin'+str(i)+', (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_vit'+str(i+1)+'=self.vit'+str(i+1)+'(model_swin'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='cbam':
                            exec('model_cbam_out'+str(i)+'_2 = F.pad(model_cbam_out'+str(i)+'_2, (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_vit'+str(i+1)+'=self.vit'+str(i+1)+'(model_cbam_out'+str(i)+'_2)', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='maxpooling' or self.model_list[i-1][0]=='avepooling':
                            exec('model_pool'+str(i)+' = F.pad(model_pool'+str(i)+', (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_vit'+str(i+1)+'=self.vit'+str(i+1)+'(model_pool'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='batchnormalization' or self.model_list[i-1][0]=='layernormalization':
                            exec('model_norm'+str(i)+' = F.pad(model_norm'+str(i)+', (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_vit'+str(i+1)+'=self.vit'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='activation' :
                            exec('model_act'+str(i)+' = F.pad(model_act'+str(i)+', (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_vit'+str(i+1)+'=self.vit'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='dropout' :
                            exec('model_drop'+str(i)+' = F.pad(model_drop'+str(i)+', (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_vit'+str(i+1)+'=self.vit'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                    if self.timestep!=None:
                        exec('model_vit'+str(i+1)+'=model_vit'+str(i+1)+'.reshape(-1,self.timestep,in_channels,self.hight,self.weight)', globals(), self.__dict__)
                    else:
                        exec('model_vit'+str(i+1)+'=model_vit'+str(i+1)+'.reshape(-1,in_channels,self.hight,self.weight)', globals(), self.__dict__)
                elif self.model_list[i][0] == 'swin':
                    if self.vit_patch_size==None:
                        self.new_hights,self.new_weights,self.vit_patch_size=find_best_common_divisor(self.hight,self.weight)
                    else:
                        self.new_hights,self.new_weights=find_new_a_b(self.hight,self.weight,self.vit_patch_size)
                    exec('in_channels=self.model_list[i][1]', globals(), self.__dict__)
                    if i==0:
                        exec('x = F.pad(x, (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                        exec('model_swin'+str(i+1)+'=self.swin'+str(i+1)+'(x)', globals(), self.__dict__)
                    else:
                        if self.model_list[i-1][0]=='conv':
                            exec('model_conv'+str(i)+' = F.pad(model_conv'+str(i)+', (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_swin'+str(i+1)+'=self.swin'+str(i+1)+'(model_conv'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='resnet':
                            exec('model_add'+str(i)+' = F.pad(model_add'+str(i)+', (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_swin'+str(i+1)+'=self.swin'+str(i+1)+'(model_add'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='vit':
                            exec('model_vit'+str(i)+' = F.pad(model_vit'+str(i)+', (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_swin'+str(i+1)+'=self.swin'+str(i+1)+'(model_vit'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='swin':
                            exec('model_swin'+str(i)+' = F.pad(model_swin'+str(i)+', (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_swin'+str(i+1)+'=self.swin'+str(i+1)+'(model_swin'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='cbam':
                            exec('model_cbam_out'+str(i)+'_2 = F.pad(model_cbam_out'+str(i)+'_2, (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_swin'+str(i+1)+'=self.swin'+str(i+1)+'(model_cbam_out'+str(i)+'_2)', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='maxpooling' or self.model_list[i-1][0]=='avepooling':
                            exec('model_pool'+str(i)+' = F.pad(model_pool'+str(i)+', (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_swin'+str(i+1)+'=self.swin'+str(i+1)+'(model_pool'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='batchnormalization' or self.model_list[i-1][0]=='layernormalization':
                            exec('model_norm'+str(i)+' = F.pad(model_norm'+str(i)+', (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_swin'+str(i+1)+'=self.swin'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='activation' :
                            exec('model_act'+str(i)+' = F.pad(model_act'+str(i)+', (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_swin'+str(i+1)+'=self.swin'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='dropout' :
                            exec('model_drop'+str(i)+' = F.pad(model_drop'+str(i)+', (int(np.floor((self.new_weights-self.weight)/2.0)),int(np.ceil((self.new_weights-self.weight)/2.0)),int(np.floor((self.new_hights-self.hight)/2.0)),int(np.ceil((self.new_hights-self.hight)/2.0))), mode="constant", value=0)', globals(), self.__dict__)
                            exec('model_swin'+str(i+1)+'=self.swin'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                    if self.timestep!=None:
                        exec('model_swin'+str(i+1)+'=model_swin'+str(i+1)+'.reshape(-1,self.timestep,in_channels,self.hight,self.weight)', globals(), self.__dict__)
                    else:
                        exec('model_swin'+str(i+1)+'=model_swin'+str(i+1)+'.reshape(-1,in_channels,self.hight,self.weight)', globals(), self.__dict__)
                elif self.model_list[i][0] == 'cbam' :
                    if i==0:
                        exec('model_cbam_max_pool'+str(i+1)+'=self.cbam_max_pool'+str(i+1)+'(x)', globals(), self.__dict__)
                        exec('model_cbam_ave_pool'+str(i+1)+'=self.cbam_ave_pool'+str(i+1)+'(x)', globals(), self.__dict__)
                        exec('model_cbam_conv'+str(i+1)+'_1_max=self.cbam_conv'+str(i+1)+'_1(model_cbam_max_pool'+str(i+1)+')', globals(), self.__dict__)
                        exec('model_cbam_conv'+str(i+1)+'_1_ave=self.cbam_conv'+str(i+1)+'_1(model_cbam_ave_pool'+str(i+1)+')', globals(), self.__dict__)
                        exec('model_cbam_act'+str(i+1)+'_1_max=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_max)', globals(), self.__dict__)
                        exec('model_cbam_act'+str(i+1)+'_1_ave=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                        exec('model_cbam_conv'+str(i+1)+'_2_max=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_max)', globals(), self.__dict__)
                        exec('model_cbam_conv'+str(i+1)+'_2_ave=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                        exec('model_cbam_out'+str(i+1)+'_1=(self.cbam_act'+str(i+1)+'_2(model_cbam_conv'+str(i+1)+'_2_max+model_cbam_conv'+str(i+1)+'_2_ave))*x', globals(), self.__dict__)
                        if self.timestep!=None:
                            exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)[0]', globals(), self.__dict__)
                            exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)', globals(), self.__dict__)
                            exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=2))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                        else:
                            exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)[0]', globals(), self.__dict__)
                            exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)', globals(), self.__dict__)
                            exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=1))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                    else:
                        if self.model_list[i-1][0]=='cov' :
                            exec('model_cbam_max_pool'+str(i+1)+'=self.cbam_max_pool'+str(i+1)+'(model_conv'+str(i)+')', globals(), self.__dict__)
                            exec('model_cbam_ave_pool'+str(i+1)+'=self.cbam_ave_pool'+str(i+1)+'(model_conv'+str(i)+')', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_max=self.cbam_conv'+str(i+1)+'_1(model_cbam_max_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_ave=self.cbam_conv'+str(i+1)+'_1(model_cbam_ave_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_max=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_ave=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_max=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_ave=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_out'+str(i+1)+'_1=(self.cbam_act'+str(i+1)+'_2(model_cbam_conv'+str(i+1)+'_2_max+model_cbam_conv'+str(i+1)+'_2_ave))*model_conv'+str(i), globals(), self.__dict__)
                            if self.timestep!=None:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=2))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                            else:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=1))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='resnet' :
                            exec('model_cbam_max_pool'+str(i+1)+'=self.cbam_max_pool'+str(i+1)+'(model_add'+str(i)+')', globals(), self.__dict__)
                            exec('model_cbam_ave_pool'+str(i+1)+'=self.cbam_ave_pool'+str(i+1)+'(model_add'+str(i)+')', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_max=self.cbam_conv'+str(i+1)+'_1(model_cbam_max_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_ave=self.cbam_conv'+str(i+1)+'_1(model_cbam_ave_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_max=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_ave=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_max=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_ave=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_out'+str(i+1)+'_1=(self.cbam_act'+str(i+1)+'_2(model_cbam_conv'+str(i+1)+'_2_max+model_cbam_conv'+str(i+1)+'_2_ave))*model_add'+str(i), globals(), self.__dict__)
                            if self.timestep!=None:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=2))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                            else:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=1))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='vit' :
                            exec('model_cbam_max_pool'+str(i+1)+'=self.cbam_max_pool'+str(i+1)+'(model_vit'+str(i)+')', globals(), self.__dict__)
                            exec('model_cbam_ave_pool'+str(i+1)+'=self.cbam_ave_pool'+str(i+1)+'(model_vit'+str(i)+')', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_max=self.cbam_conv'+str(i+1)+'_1(model_cbam_max_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_ave=self.cbam_conv'+str(i+1)+'_1(model_cbam_ave_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_max=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_ave=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_max=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_ave=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_out'+str(i+1)+'_1=(self.cbam_act'+str(i+1)+'_2(model_cbam_conv'+str(i+1)+'_2_max+model_cbam_conv'+str(i+1)+'_2_ave))*model_vit'+str(i), globals(), self.__dict__)
                            if self.timestep!=None:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=2))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                            else:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=1))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='swin' :
                            exec('model_cbam_max_pool'+str(i+1)+'=self.cbam_max_pool'+str(i+1)+'(model_swin'+str(i)+')', globals(), self.__dict__)
                            exec('model_cbam_ave_pool'+str(i+1)+'=self.cbam_ave_pool'+str(i+1)+'(model_swin'+str(i)+')', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_max=self.cbam_conv'+str(i+1)+'_1(model_cbam_max_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_ave=self.cbam_conv'+str(i+1)+'_1(model_cbam_ave_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_max=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_ave=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_max=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_ave=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_out'+str(i+1)+'_1=(self.cbam_act'+str(i+1)+'_2(model_cbam_conv'+str(i+1)+'_2_max+model_cbam_conv'+str(i+1)+'_2_ave))*model_swin'+str(i), globals(), self.__dict__)
                            if self.timestep!=None:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=2))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                            else:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=1))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='cbam' :
                            exec('model_cbam_max_pool'+str(i+1)+'=self.cbam_max_pool'+str(i+1)+'(model_cbam_out'+str(i)+'_2)', globals(), self.__dict__)
                            exec('model_cbam_ave_pool'+str(i+1)+'=self.cbam_ave_pool'+str(i+1)+'(model_cbam_out'+str(i)+'_2)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_max=self.cbam_conv'+str(i+1)+'_1(model_cbam_max_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_ave=self.cbam_conv'+str(i+1)+'_1(model_cbam_ave_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_max=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_ave=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_max=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_ave=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_out'+str(i+1)+'_1=(self.cbam_act'+str(i+1)+'_2(model_cbam_conv'+str(i+1)+'_2_max+model_cbam_conv'+str(i+1)+'_2_ave))*model_cbam_out'+str(i)+'_2', globals(), self.__dict__)
                            if self.timestep!=None:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=2))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                            else:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=1))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='maxpooling' or self.model_list[i-1][0]=='avepooling':
                            exec('model_cbam_max_pool'+str(i+1)+'=self.cbam_max_pool'+str(i+1)+'(model_pool'+str(i)+')', globals(), self.__dict__)
                            exec('model_cbam_ave_pool'+str(i+1)+'=self.cbam_ave_pool'+str(i+1)+'(model_pool'+str(i)+')', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_max=self.cbam_conv'+str(i+1)+'_1(model_cbam_max_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_ave=self.cbam_conv'+str(i+1)+'_1(model_cbam_ave_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_max=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_ave=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_max=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_ave=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_out'+str(i+1)+'_1=(self.cbam_act'+str(i+1)+'_2(model_cbam_conv'+str(i+1)+'_2_max+model_cbam_conv'+str(i+1)+'_2_ave))*model_pool'+str(i), globals(), self.__dict__)
                            if self.timestep!=None:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=2))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                            else:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=1))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='batchnormalization' or self.model_list[i-1][0]=='layernormalization':
                            exec('model_cbam_max_pool'+str(i+1)+'=self.cbam_max_pool'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            exec('model_cbam_ave_pool'+str(i+1)+'=self.cbam_ave_pool'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_max=self.cbam_conv'+str(i+1)+'_1(model_cbam_max_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_ave=self.cbam_conv'+str(i+1)+'_1(model_cbam_ave_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_max=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_ave=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_max=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_ave=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_out'+str(i+1)+'_1=(self.cbam_act'+str(i+1)+'_2(model_cbam_conv'+str(i+1)+'_2_max+model_cbam_conv'+str(i+1)+'_2_ave))*model_norm'+str(i), globals(), self.__dict__)
                            if self.timestep!=None:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=2))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                            else:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=1))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='activation' :
                            exec('model_cbam_max_pool'+str(i+1)+'=self.cbam_max_pool'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            exec('model_cbam_ave_pool'+str(i+1)+'=self.cbam_ave_pool'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_max=self.cbam_conv'+str(i+1)+'_1(model_cbam_max_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_ave=self.cbam_conv'+str(i+1)+'_1(model_cbam_ave_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_max=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_ave=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_max=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_ave=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_out'+str(i+1)+'_1=(self.cbam_act'+str(i+1)+'_2(model_cbam_conv'+str(i+1)+'_2_max+model_cbam_conv'+str(i+1)+'_2_ave))*model_act'+str(i), globals(), self.__dict__)
                            if self.timestep!=None:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=2))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                            else:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=1))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='dropout' :
                            exec('model_cbam_max_pool'+str(i+1)+'=self.cbam_max_pool'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                            exec('model_cbam_ave_pool'+str(i+1)+'=self.cbam_ave_pool'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_max=self.cbam_conv'+str(i+1)+'_1(model_cbam_max_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_1_ave=self.cbam_conv'+str(i+1)+'_1(model_cbam_ave_pool'+str(i+1)+')', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_max=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_act'+str(i+1)+'_1_ave=self.cbam_act'+str(i+1)+'_1(model_cbam_conv'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_max=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_max)', globals(), self.__dict__)
                            exec('model_cbam_conv'+str(i+1)+'_2_ave=self.cbam_conv'+str(i+1)+'_2(model_cbam_act'+str(i+1)+'_1_ave)', globals(), self.__dict__)
                            exec('model_cbam_out'+str(i+1)+'_1=(self.cbam_act'+str(i+1)+'_2(model_cbam_conv'+str(i+1)+'_2_max+model_cbam_conv'+str(i+1)+'_2_ave))*model_drop'+str(i), globals(), self.__dict__)
                            if self.timestep!=None:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=2, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=2))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                            else:
                                exec('model_cbam_sa'+str(i+1)+'_max=torch.max(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)[0]', globals(), self.__dict__)
                                exec('model_cbam_sa'+str(i+1)+'_ave=torch.mean(model_cbam_out'+str(i+1)+'_1, dim=1, keepdim=True)', globals(), self.__dict__)
                                exec('model_cbam_out'+str(i+1)+'_2=(self.cbam_act'+str(i+1)+'_2(self.cbam_conv'+str(i+1)+'_3(torch.cat([model_cbam_sa'+str(i+1)+'_max,model_cbam_sa'+str(i+1)+'_ave],dim=1))))*model_cbam_out'+str(i+1)+'_1', globals(), self.__dict__)
                elif self.model_list[i][0] in ['gc', 'ga', 'chebnet', 'gsage']:
                    if i == 0:
                        if i == len(self.model_list) - 1:
                            outputs = eval('self.gc'+str(i+1)+'(x)', globals(), self.__dict__)
                        else:
                            exec('model_gc'+str(i+1)+'=self.gc'+str(i+1)+'(x)', globals(), self.__dict__)
                    else:
                        if self.model_list[i-1][0] == 'cov':
                            prev_tensor = 'model_conv'+str(i)
                        elif self.model_list[i-1][0] == 'resnet':
                            prev_tensor = 'model_add'+str(i)
                        elif self.model_list[i-1][0] == 'vit':
                            prev_tensor = 'model_vit'+str(i)
                        elif self.model_list[i-1][0] == 'swin':
                            prev_tensor = 'model_swin'+str(i)
                        elif self.model_list[i-1][0] == 'cbam':
                            prev_tensor = 'model_cbam_out'+str(i)+'_2'
                        elif self.model_list[i-1][0] == 'maxpooling' or self.model_list[i-1][0] == 'avepooling':
                            prev_tensor = 'model_pool'+str(i)
                        elif self.model_list[i-1][0] == 'batchnormalization' or self.model_list[i-1][0] == 'layernormalization':
                            prev_tensor = 'model_norm'+str(i)
                        elif self.model_list[i-1][0] == 'activation':
                            prev_tensor = 'model_act'+str(i)
                        elif self.model_list[i-1][0] == 'dropout':
                            prev_tensor = 'model_drop'+str(i)
                        elif self.model_list[i-1][0] in ['gc', 'ga', 'chebnet', 'gsage']:
                            prev_tensor = 'model_gc'+str(i)
                        else:
                            raise ValueError("gc/ga/chebnet/gsage 前一层不能是 flatten/fc/transformer/mamba/tcn/lstm/rnn/gru，建议把图层放在 flatten 之前。")

                        if i == len(self.model_list) - 1:
                            outputs = eval('self.gc'+str(i+1)+'('+prev_tensor+')', globals(), self.__dict__)
                        else:
                            exec('model_gc'+str(i+1)+'=self.gc'+str(i+1)+'('+prev_tensor+')', globals(), self.__dict__)
                elif self.model_list[i][0] == 'maxpooling' or self.model_list[i][0] == 'avepooling':
                    if i==0:
                        exec('model_pool'+str(i+1)+'=self.pool'+str(i+1)+'(x)', globals(), self.__dict__)
                    else:
                        if self.model_list[i-1][0]=='cov' :
                            exec('model_pool'+str(i+1)+'=self.pool'+str(i+1)+'(model_conv'+str(i)+')', globals(), self.__dict__)
                            if ifmulti_scale=='yes':
                                exec('model_pool_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_conv'+str(i)+')', globals(), self.__dict__)
                                try:
                                    if self.timestep!=None:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=2)', globals(), self.__dict__)
                                    else:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=1)', globals(), self.__dict__)
                                except:
                                    exec('model_conc=model_pool_fla'+str(i+1), globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='resnet' :
                            exec('model_pool'+str(i+1)+'=self.pool'+str(i+1)+'(model_add'+str(i)+')', globals(), self.__dict__)
                            if ifmulti_scale=='yes':
                                exec('model_pool_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_add'+str(i)+')', globals(), self.__dict__)
                                try:
                                    if self.timestep!=None:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=2)', globals(), self.__dict__)
                                    else:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=1)', globals(), self.__dict__)
                                except:
                                    exec('model_conc=model_pool_fla'+str(i+1), globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='vit' :
                            exec('model_pool'+str(i+1)+'=self.pool'+str(i+1)+'(model_vit'+str(i)+')', globals(), self.__dict__)
                            if ifmulti_scale=='yes':
                                exec('model_pool_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_vit'+str(i)+')', globals(), self.__dict__)
                                try:
                                    if self.timestep!=None:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=2)', globals(), self.__dict__)
                                    else:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=1)', globals(), self.__dict__)
                                except:
                                    exec('model_conc=model_pool_fla'+str(i+1), globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='swin' :
                            exec('model_pool'+str(i+1)+'=self.pool'+str(i+1)+'(model_swin'+str(i)+')', globals(), self.__dict__)
                            if ifmulti_scale=='yes':
                                exec('model_pool_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_swin'+str(i)+')', globals(), self.__dict__)
                                try:
                                    if self.timestep!=None:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=2)', globals(), self.__dict__)
                                    else:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=1)', globals(), self.__dict__)
                                except:
                                    exec('model_conc=model_pool_fla'+str(i+1), globals(), self.__dict__)
                        elif self.model_list[i-1][0] in ['gc', 'ga', 'chebnet', 'gsage']:
                            if i == len(self.model_list) - 1:
                                outputs = eval('self.pool'+str(i+1)+'(model_gc'+str(i)+')', globals(), self.__dict__)
                            else:
                                exec('model_pool'+str(i+1)+'=self.pool'+str(i+1)+'(model_gc'+str(i)+')', globals(), self.__dict__)

                            if ifmulti_scale == 'yes':
                                try:
                                    exec('model_pool_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_gc'+str(i)+')', globals(), self.__dict__)
                                    exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),dim=-1)', globals(), self.__dict__)
                                except:
                                    exec('model_conc=model_pool_fla'+str(i+1), globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='cbam' :
                            exec('model_pool'+str(i+1)+'=self.pool'+str(i+1)+'(model_cbam_out'+str(i)+'_2)', globals(), self.__dict__)
                            if ifmulti_scale=='yes':
                                exec('model_pool_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_cbam_out'+str(i)+'_2)', globals(), self.__dict__)
                                try:
                                    if self.timestep!=None:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=2)', globals(), self.__dict__)
                                    else:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=1)', globals(), self.__dict__)
                                except:
                                    exec('model_conc=model_pool_fla'+str(i+1), globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='maxpooling' or self.model_list[i-1][0]=='avepooling':
                            exec('model_pool'+str(i+1)+'=self.pool'+str(i+1)+'(model_pool'+str(i)+')', globals(), self.__dict__)
                            if ifmulti_scale=='yes':
                                exec('model_pool_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_pool'+str(i)+')', globals(), self.__dict__)
                                try:
                                    if self.timestep!=None:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=2)', globals(), self.__dict__)
                                    else:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=1)', globals(), self.__dict__)
                                except:
                                    exec('model_conc=model_pool_fla'+str(i+1), globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='batchnormalization' or self.model_list[i-1][0]=='layernormalization':
                            exec('model_pool'+str(i+1)+'=self.pool'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            if ifmulti_scale=='yes':
                                exec('model_pool_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                                try:
                                    if self.timestep!=None:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=2)', globals(), self.__dict__)
                                    else:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=1)', globals(), self.__dict__)
                                except:
                                    exec('model_conc=model_pool_fla'+str(i+1), globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='activation' :
                            exec('model_pool'+str(i+1)+'=self.pool'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            if ifmulti_scale=='yes':
                                exec('model_pool_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                                try:
                                    if self.timestep!=None:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=2)', globals(), self.__dict__)
                                    else:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=1)', globals(), self.__dict__)
                                except:
                                    exec('model_conc=model_pool_fla'+str(i+1), globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='dropout' :
                            exec('model_pool'+str(i+1)+'=self.pool'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                            if ifmulti_scale=='yes':
                                exec('model_pool_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                                try:
                                    if self.timestep!=None:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=2)', globals(), self.__dict__)
                                    else:
                                        exec('model_conc=self.conc'+str(i+1)+'((model_conc,model_pool_fla'+str(i+1)+'),axis=1)', globals(), self.__dict__)
                                except:
                                    exec('model_conc=model_pool_fla'+str(i+1), globals(), self.__dict__)
                elif self.model_list[i][0] == 'batchnormalization' or self.model_list[i][0] == 'layernormalization':
                    if i ==0:
                        if i==len(self.model_list)-1:
                            outputs=eval('self.norm'+str(i+1)+'(x)', globals(), self.__dict__)
                        else:
                            exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(x)', globals(), self.__dict__)
                    else:
                        if i==len(self.model_list)-1:
                            if self.model_list[i-1][0]=='cov' :
                                outputs=eval('self.norm'+str(i+1)+'(model_conv'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='resnet' :
                                outputs=eval('self.norm'+str(i+1)+'(model_add'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='vit' :
                                outputs=eval('self.norm'+str(i+1)+'(model_vit'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='swin' :
                                outputs=eval('self.norm'+str(i+1)+'(model_swin'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0] in ['gc', 'ga', 'chebnet', 'gsage']:
                                outputs = eval('self.norm'+str(i+1)+'(model_gc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='cbam' :
                                outputs=eval('self.norm'+str(i+1)+'(model_cbam_out'+str(i)+'_2)', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='transformer' :
                                outputs=eval('self.norm'+str(i+1)+'(model_transformer'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='mamba' or self.model_list[i-1][0]=='mamba2':
                                outputs=eval('self.norm'+str(i+1)+'(model_mamba'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='tcn' :
                                outputs=eval('self.norm'+str(i+1)+'(model_tcn'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='lstm' :
                                outputs=eval('self.norm'+str(i+1)+'(model_lstm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='rnn' :
                                outputs=eval('self.norm'+str(i+1)+'(model_rnn'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='ltc' :
                                outputs=eval('self.norm'+str(i+1)+'(model_ltc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='gru' :
                                outputs=eval('self.norm'+str(i+1)+'(model_gru'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='maxpooling' or self.model_list[i-1][0]=='avepooling':
                                outputs=eval('self.norm'+str(i+1)+'(model_pool'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='batchnormalization' or self.model_list[i-1][0]=='layernormalization':
                                outputs=eval('self.norm'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='activation':
                                outputs=eval('self.norm'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='dropout' :
                                outputs=eval('self.norm'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='fc':
                                outputs=eval('self.norm'+str(i+1)+'(model_fc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='flatten':
                                outputs=eval('self.norm'+str(i+1)+'(model_fla'+str(i)+')', globals(), self.__dict__)
                        else:
                            if self.model_list[i-1][0]=='cov' :
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_conv'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='resnet' :
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_add'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='vit' :
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_vit'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='swin' :
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_swin'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0] in ['gc', 'ga', 'chebnet', 'gsage']:
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_gc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='cbam' :
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_cbam_out'+str(i)+'_2)', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='transformer' :
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_transformer'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='mamba' or self.model_list[i-1][0]=='mamba2':
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_mamba'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='tcn' :
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_tcn'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='lstm' :
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_lstm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='rnn' :
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_rnn'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='ltc' :
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_ltc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='gru' :
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_gru'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='maxpooling' or self.model_list[i-1][0]=='avepooling':
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_pool'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='batchnormalization' or self.model_list[i-1][0]=='layernormalization':
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='activation':
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='dropout' :
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='fc':
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_fc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='flatten':
                                exec('model_norm'+str(i+1)+'=self.norm'+str(i+1)+'(model_fla'+str(i)+')', globals(), self.__dict__)
                elif self.model_list[i][0] == 'activation':
                    if i==0:
                        if i==len(self.model_list)-1:
                            outputs=eval('self.act'+str(i+1)+'(x)', globals(), self.__dict__)
                        else:
                            exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(x)', globals(), self.__dict__)
                    else:
                        if i==len(self.model_list)-1:
                            if self.model_list[i-1][0]=='cov' :
                                outputs=eval('self.act'+str(i+1)+'(model_conv'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='resnet' :
                                outputs=eval('self.act'+str(i+1)+'(model_add'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='vit' :
                                outputs=eval('self.act'+str(i+1)+'(model_vit'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='swin' :
                                outputs=eval('self.act'+str(i+1)+'(model_swin'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0] in ['gc', 'ga', 'chebnet', 'gsage']:
                                outputs = eval('self.act'+str(i+1)+'(model_gc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='cbam' :
                                outputs=eval('self.act'+str(i+1)+'(model_cbam_out'+str(i)+'_2)', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='transformer' :
                                outputs=eval('self.act'+str(i+1)+'(model_transformer'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='mamba' or self.model_list[i-1][0]=='mamba2':
                                outputs=eval('self.act'+str(i+1)+'(model_mamba'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='tcn' :
                                outputs=eval('self.act'+str(i+1)+'(model_tcn'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='lstm' :
                                outputs=eval('self.act'+str(i+1)+'(model_lstm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='rnn' :
                                outputs=eval('self.act'+str(i+1)+'(model_rnn'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='ltc' :
                                outputs=eval('self.act'+str(i+1)+'(model_ltc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='gru' :
                                outputs=eval('self.act'+str(i+1)+'(model_gru'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='maxpooling' or self.model_list[i-1][0]=='avepooling':
                                outputs=eval('self.act'+str(i+1)+'(model_pool'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='batchnormalization' or self.model_list[i-1][0]=='layernormalization':
                                outputs=eval('self.act'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='activation':
                                outputs=eval('self.act'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='dropout' :
                                outputs=eval('self.act'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='fc':
                                outputs=eval('self.act'+str(i+1)+'(model_fc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='flatten':
                                outputs=eval('self.act'+str(i+1)+'(model_fla'+str(i)+')', globals(), self.__dict__)
                        else:
                            if self.model_list[i-1][0]=='cov':
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_conv'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='resnet':
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_add'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='vit':
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_vit'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='swin':
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_swin'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0] in ['gc', 'ga', 'chebnet', 'gsage']:
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_gc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='cbam':
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_cbam_out'+str(i)+'_2)', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='transformer':
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_transformer'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='mamba' or self.model_list[i-1][0]=='mamba2':
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_mamba'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='tcn':
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_tcn'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='lstm':
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_lstm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='rnn':
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_rnn'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='ltc':
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_ltc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='gru':
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_gru'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='maxpooling' or self.model_list[i-1][0]=='avepooling':
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_pool'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='batchnormalization' or self.model_list[i-1][0]=='layernormalization':
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='activation':
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='dropout' :
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='fc':
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_fc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='flatten':
                                exec('model_act'+str(i+1)+'=self.act'+str(i+1)+'(model_fla'+str(i)+')', globals(), self.__dict__)
                elif self.model_list[i][0] == 'flatten':
                    if i ==0:
                        if i==len(self.model_list)-1:
                            outputs=eval('self.fla'+str(i+1)+'(x)', globals(), self.__dict__)
                        else:
                            exec('model_fla'+str(i+1)+'=self.fla'+str(i+1)+'(x)', globals(), self.__dict__)
                    else:
                        if i==len(self.model_list)-1:
                            if self.model_list[i-1][0]=='cov' :
                                outputs=eval('self.fla'+str(i+1)+'(model_conv'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='resnet' :
                               outputs=eval('self.fla'+str(i+1)+'(model_add'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='vit' :
                                outputs=eval('self.fla'+str(i+1)+'(model_vit'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='swin' :
                                outputs=eval('self.fla'+str(i+1)+'(model_swin'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0] in ['gc', 'ga', 'chebnet', 'gsage']:
                                outputs = eval('self.fla'+str(i+1)+'(model_gc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='cbam' :
                                outputs=eval('self.fla'+str(i+1)+'(model_cbam_out'+str(i)+'_2)', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='maxpooling' or self.model_list[i-1][0]=='avepooling':
                                outputs=eval('self.fla'+str(i+1)+'(model_pool'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='batchnormalization' or self.model_list[i-1][0]=='layernormalization':
                                outputs=eval('self.fla'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='activation':
                                outputs=eval('self.fla'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='dropout':
                                outputs=eval('self.fla'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                        else:
                            if self.model_list[i-1][0]=='cov' :
                                exec('model_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_conv'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='resnet' :
                                exec('model_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_add'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='vit' :
                                exec('model_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_vit'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='swin' :
                                exec('model_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_swin'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0] in ['gc', 'ga', 'chebnet', 'gsage']:
                                exec('model_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_gc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='cbam' :
                                exec('model_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_cbam_out'+str(i)+'_2)', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='maxpooling' or self.model_list[i-1][0]=='avepooling':
                                exec('model_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_pool'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='batchnormalization' or self.model_list[i-1][0]=='layernormalization':
                                exec('model_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='activation':
                                exec('model_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='dropout':
                                exec('model_fla'+str(i+1)+'=self.fla'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                        if ifmulti_scale=='yes':
                            if timestep !=None:
                                exec('model_fla'+str(i+1)+'=self.conc'+str(i+1)+'((model_conc,model_fla'+str(i+1)+'),axis=2)', globals(), self.__dict__)
                            else:
                                exec('model_fla'+str(i+1)+'=self.conc'+str(i+1)+'((model_conc,model_fla'+str(i+1)+'),axis=1)', globals(), self.__dict__)
                elif self.model_list[i][0] =='fc':
                    if i==0:
                        if i==len(self.model_list)-1:
                            outputs=eval('self.fc'+str(i+1)+'(x)', globals(), self.__dict__)
                        else:
                            exec('model_fc'+str(i+1)+'=self.fc'+str(i+1)+'(x)', globals(), self.__dict__)
                    else:
                        if i==len(self.model_list)-1:
                            if self.model_list[i-1][0]=='batchnormalization' :
                                outputs=eval('self.fc'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='transformer' :
                                outputs=eval('self.fc'+str(i+1)+'(model_transformer'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='mamba' or self.model_list[i-1][0]=='mamba2':
                                outputs=eval('self.fc'+str(i+1)+'(model_mamba'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='tcn' :
                                outputs=eval('self.fc'+str(i+1)+'(model_tcn'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='lstm' :
                                outputs=eval('self.fc'+str(i+1)+'(model_lstm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='rnn' :
                                outputs=eval('self.fc'+str(i+1)+'(model_rnn'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='ltc' :
                                outputs=eval('self.fc'+str(i+1)+'(model_ltc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='gru' :
                                outputs=eval('self.fc'+str(i+1)+'(model_gru'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='activation':
                                outputs=eval('self.fc'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='dropout':
                                outputs=eval('self.fc'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='fc':
                                outputs=eval('self.fc'+str(i+1)+'(model_fc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='flatten':
                                outputs=eval('self.fc'+str(i+1)+'(model_fla'+str(i)+')', globals(), self.__dict__)
                        else:
                            if self.model_list[i-1][0]=='batchnormalization' :
                                exec('model_fc'+str(i+1)+'=self.fc'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='transformer' :
                                exec('model_fc'+str(i+1)+'=self.fc'+str(i+1)+'(model_transformer'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='mamba' or self.model_list[i-1][0]=='mamba2':
                                exec('model_fc'+str(i+1)+'=self.fc'+str(i+1)+'(model_mamba'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='tcn' :
                                exec('model_fc'+str(i+1)+'=self.fc'+str(i+1)+'(model_tcn'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='lstm' :
                                exec('model_fc'+str(i+1)+'=self.fc'+str(i+1)+'(model_lstm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='rnn' :
                                exec('model_fc'+str(i+1)+'=self.fc'+str(i+1)+'(model_rnn'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='ltc' :
                                exec('model_fc'+str(i+1)+'=self.fc'+str(i+1)+'(model_ltc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='gru' :
                                exec('model_fc'+str(i+1)+'=self.fc'+str(i+1)+'(model_gru'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='activation':
                                exec('model_fc'+str(i+1)+'=self.fc'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='dropout':
                                exec('model_fc'+str(i+1)+'=self.fc'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='fc':
                                exec('model_fc'+str(i+1)+'=self.fc'+str(i+1)+'(model_fc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='flatten':
                                exec('model_fc'+str(i+1)+'=self.fc'+str(i+1)+'(model_fla'+str(i)+')', globals(), self.__dict__)
                elif self.model_list[i][0] == 'dropout':
                    if i==0:
                        if i==len(self.model_list)-1:
                            outputs=eval('self.drop'+str(i+1)+'(x)', globals(), self.__dict__)
                        else:
                            exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(x)', globals(), self.__dict__)
                    else:
                        if i==len(self.model_list)-1:
                            if self.model_list[i-1][0]=='cov' :
                                outputs=eval('self.drop'+str(i+1)+'(model_conv'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='resnet' :
                                outputs=eval('self.drop'+str(i+1)+'(model_add'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='vit' :
                                outputs=eval('self.drop'+str(i+1)+'(model_vit'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='swin' :
                                outputs=eval('self.drop'+str(i+1)+'(model_swin'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0] in ['gc', 'ga', 'chebnet', 'gsage']:
                                outputs = eval('self.drop'+str(i+1)+'(model_gc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='cbam' :
                                outputs=eval('self.drop'+str(i+1)+'(model_cbam_out'+str(i)+'_2)', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='transformer' :
                                outputs=eval('self.drop'+str(i+1)+'(model_transformer'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='mamba' or self.model_list[i-1][0]=='mamba2':
                                outputs=eval('self.drop'+str(i+1)+'(model_mamba'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='tcn' :
                                outputs=eval('self.drop'+str(i+1)+'(model_tcn'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='lstm' :
                                outputs=eval('self.drop'+str(i+1)+'(model_lstm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='rnn' :
                                outputs=eval('self.drop'+str(i+1)+'(model_rnn'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='ltc' :
                                outputs=eval('self.drop'+str(i+1)+'(model_ltc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='gru' :
                                outputs=eval('self.drop'+str(i+1)+'(model_gru'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='maxpooling' or self.model_list[i-1][0]=='avepooling':
                                outputs=eval('self.drop'+str(i+1)+'(model_pool'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='batchnormalization' or self.model_list[i-1][0]=='layernormalization':
                                outputs=eval('self.drop'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='activation':
                                outputs=eval('self.drop'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='dropout' :
                                outputs=eval('self.drop'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='fc':
                                outputs=eval('self.drop'+str(i+1)+'(model_fc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='flatten':
                                outputs=eval('self.drop'+str(i+1)+'(model_fla'+str(i)+')', globals(), self.__dict__)
                        else:
                            if self.model_list[i-1][0]=='cov':
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_conv'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='resnet':
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_add'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='vit':
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_vit'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='swin':
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_swin'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0] in ['gc', 'ga', 'chebnet', 'gsage']:
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_gc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='cbam':
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_cbam_out'+str(i)+'_2)', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='transformer':
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_transformer'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='mamba' or self.model_list[i-1][0]=='mamba2':
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_mamba'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='tcn':
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_tcn'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='lstm':
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_lstm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='rnn':
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_rnn'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='ltc':
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_ltc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='gru':
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_gru'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='maxpooling' or self.model_list[i-1][0]=='avepooling':
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_pool'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='batchnormalization' or self.model_list[i-1][0]=='layernormalization':
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='activation':
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='dropout' :
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='fc':
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_fc'+str(i)+')', globals(), self.__dict__)
                            elif self.model_list[i-1][0]=='flatten':
                                exec('model_drop'+str(i+1)+'=self.drop'+str(i+1)+'(model_fla'+str(i)+')', globals(), self.__dict__)
                elif self.model_list[i][0] == 'transformer':
                    if i ==0:
                        exec('model_embedding'+str(i+1)+'=self.embedding'+str(i+1)+'(position)', globals(), self.__dict__)
                        exec('model_transformer_add'+str(i+1)+'=self.transformer_add'+str(i+1)+'(model_embedding'+str(i+1)+',x)', globals(), self.__dict__)
                        exec('model_transformer'+str(i+1)+'=self.transformer'+str(i+1)+'(model_transformer_add'+str(i+1)+')', globals(), self.__dict__)
                        if (self.outputstep is None) and is_last_sequence_layer(model_list,i):
                            if i==len(self.model_list)-1:
                                outputs=eval('model_transformer'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                            else:
                                exec('model_transformer'+str(i+1)+'=model_transformer'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                    else:
                        if model_list[i-1][0] == 'flatten':
                            exec('model_embedding'+str(i+1)+'=self.embedding'+str(i+1)+'(position)', globals(), self.__dict__)
                            exec('model_transformer_add'+str(i+1)+'=self.transformer_add'+str(i+1)+'(model_embedding'+str(i+1)+',model_fla'+str(i)+')', globals(), self.__dict__)
                            exec('model_transformer'+str(i+1)+'=self.transformer'+str(i+1)+'(model_transformer_add'+str(i+1)+')', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'transformer':
                            exec('model_embedding'+str(i+1)+'=self.embedding'+str(i+1)+'(position)', globals(), self.__dict__)
                            exec('model_transformer_add'+str(i+1)+'=self.transformer_add'+str(i+1)+'(model_embedding'+str(i+1)+',model_transformer'+str(i)+')', globals(), self.__dict__)
                            exec('model_transformer'+str(i+1)+'=self.transformer'+str(i+1)+'(model_transformer_add'+str(i+1)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='mamba' or self.model_list[i-1][0]=='mamba2':
                            exec('model_embedding'+str(i+1)+'=self.embedding'+str(i+1)+'(position)', globals(), self.__dict__)
                            exec('model_transformer_add'+str(i+1)+'=self.transformer_add'+str(i+1)+'(model_embedding'+str(i+1)+',model_mamba'+str(i)+')', globals(), self.__dict__)
                            exec('model_transformer'+str(i+1)+'=self.transformer'+str(i+1)+'(model_transformer_add'+str(i+1)+')', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'tcn':
                            exec('model_embedding'+str(i+1)+'=self.embedding'+str(i+1)+'(position)', globals(), self.__dict__)
                            exec('model_transformer_add'+str(i+1)+'=self.transformer_add'+str(i+1)+'(model_embedding'+str(i+1)+',model_tcn'+str(i)+')', globals(), self.__dict__)
                            exec('model_transformer'+str(i+1)+'=self.transformer'+str(i+1)+'(model_transformer_add'+str(i+1)+')', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'lstm':
                            exec('model_embedding'+str(i+1)+'=self.embedding'+str(i+1)+'(position)', globals(), self.__dict__)
                            exec('model_transformer_add'+str(i+1)+'=self.transformer_add'+str(i+1)+'(model_embedding'+str(i+1)+',model_lstm'+str(i)+')', globals(), self.__dict__)
                            exec('model_transformer'+str(i+1)+'=self.transformer'+str(i+1)+'(model_transformer_add'+str(i+1)+')', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'rnn':
                            exec('model_embedding'+str(i+1)+'=self.embedding'+str(i+1)+'(position)', globals(), self.__dict__)
                            exec('model_transformer_add'+str(i+1)+'=self.transformer_add'+str(i+1)+'(model_embedding'+str(i+1)+',model_rnn'+str(i)+')', globals(), self.__dict__)
                            exec('model_transformer'+str(i+1)+'=self.transformer'+str(i+1)+'(model_transformer_add'+str(i+1)+')', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'ltc':
                            exec('model_embedding'+str(i+1)+'=self.embedding'+str(i+1)+'(position)', globals(), self.__dict__)
                            exec('model_transformer_add'+str(i+1)+'=self.transformer_add'+str(i+1)+'(model_embedding'+str(i+1)+',model_ltc'+str(i)+')', globals(), self.__dict__)
                            exec('model_transformer'+str(i+1)+'=self.transformer'+str(i+1)+'(model_transformer_add'+str(i+1)+')', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'gru':
                            exec('model_embedding'+str(i+1)+'=self.embedding'+str(i+1)+'(position)', globals(), self.__dict__)
                            exec('model_transformer_add'+str(i+1)+'=self.transformer_add'+str(i+1)+'(model_embedding'+str(i+1)+',model_gru'+str(i)+')', globals(), self.__dict__)
                            exec('model_transformer'+str(i+1)+'=self.transformer'+str(i+1)+'(model_transformer_add'+str(i+1)+')', globals(), self.__dict__)
                        elif model_list[i-1][0]=='batchnormalization' or model_list[i-1][0]=='layernormalization':
                            exec('model_embedding'+str(i+1)+'=self.embedding'+str(i+1)+'(position)', globals(), self.__dict__)
                            exec('model_transformer_add'+str(i+1)+'=self.transformer_add'+str(i+1)+'(model_embedding'+str(i+1)+',model_norm'+str(i)+')', globals(), self.__dict__)
                            exec('model_transformer'+str(i+1)+'=self.transformer'+str(i+1)+'(model_transformer_add'+str(i+1)+')', globals(), self.__dict__)
                        elif model_list[i-1][0]=='activation':
                            exec('model_embedding'+str(i+1)+'=self.embedding'+str(i+1)+'(position)', globals(), self.__dict__)
                            exec('model_transformer_add'+str(i+1)+'=self.transformer_add'+str(i+1)+'(model_embedding'+str(i+1)+',model_act'+str(i)+')', globals(), self.__dict__)
                            exec('model_transformer'+str(i+1)+'=self.transformer'+str(i+1)+'(model_transformer_add'+str(i+1)+')', globals(), self.__dict__)
                        elif model_list[i-1][0]=='dropout':
                            exec('model_embedding'+str(i+1)+'=self.embedding'+str(i+1)+'(position)', globals(), self.__dict__)
                            exec('model_transformer_add'+str(i+1)+'=self.transformer_add'+str(i+1)+'(model_embedding'+str(i+1)+',model_drop'+str(i)+')', globals(), self.__dict__)
                            exec('model_transformer'+str(i+1)+'=self.transformer'+str(i+1)+'(model_transformer_add'+str(i+1)+')', globals(), self.__dict__)
                        if (self.outputstep is None) and is_last_sequence_layer(model_list,i):
                            if i==len(self.model_list)-1:
                                outputs=eval('model_transformer'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                            else:
                                exec('model_transformer'+str(i+1)+'=model_transformer'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                elif self.model_list[i][0]=='mamba' or self.model_list[i][0]=='mamba2':
                    if i ==0:
                        tmp = x
                        tmp = tmp.contiguous()
                        exec('model_mamba'+str(i+1)+'=self.mamba'+str(i+1)+'(tmp)', globals(), self.__dict__)
                        if (self.outputstep is None) and is_last_sequence_layer(model_list,i):
                            if i==len(self.model_list)-1:
                                outputs=eval('model_mamba'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                            else:
                                exec('model_mamba'+str(i+1)+'=model_mamba'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                    else:
                        if model_list[i-1][0] == 'flatten':
                            tmp = eval('model_fla'+str(i), globals(), self.__dict__)
                            tmp = tmp.contiguous()
                            self.__dict__['tmp'] = tmp
                            exec('model_mamba'+str(i+1)+'=self.mamba'+str(i+1)+'(tmp)', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'transformer':
                            tmp = eval('model_transformer'+str(i), globals(), self.__dict__)
                            tmp = tmp.contiguous()
                            self.__dict__['tmp'] = tmp
                            exec('model_mamba'+str(i+1)+'=self.mamba'+str(i+1)+'(tmp)', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='mamba' or self.model_list[i-1][0]=='mamba2':
                            tmp = eval('model_mamba'+str(i), globals(), self.__dict__)
                            tmp = tmp.contiguous()
                            self.__dict__['tmp'] = tmp
                            exec('model_mamba'+str(i+1)+'=self.mamba'+str(i+1)+'(tmp)', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'tcn':
                            tmp = eval('model_tcn'+str(i), globals(), self.__dict__)
                            tmp = tmp.contiguous()
                            self.__dict__['tmp'] = tmp
                            exec('model_mamba'+str(i+1)+'=self.mamba'+str(i+1)+'(tmp)', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'lstm':
                            tmp = eval('model_lstm'+str(i), globals(), self.__dict__)
                            tmp = tmp.contiguous()
                            self.__dict__['tmp'] = tmp
                            exec('model_mamba'+str(i+1)+'=self.mamba'+str(i+1)+'(tmp)', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'rnn':
                            tmp = eval('model_rnn'+str(i), globals(), self.__dict__)
                            tmp = tmp.contiguous()
                            self.__dict__['tmp'] = tmp
                            exec('model_mamba'+str(i+1)+'=self.mamba'+str(i+1)+'(tmp)', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'ltc':
                            tmp = eval('model_ltc'+str(i), globals(), self.__dict__)
                            tmp = tmp.contiguous()
                            self.__dict__['tmp'] = tmp
                            exec('model_mamba'+str(i+1)+'=self.mamba'+str(i+1)+'(tmp)', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'gru':
                            tmp = eval('model_gru'+str(i), globals(), self.__dict__)
                            tmp = tmp.contiguous()
                            self.__dict__['tmp'] = tmp
                            exec('model_mamba'+str(i+1)+'=self.mamba'+str(i+1)+'(tmp)', globals(), self.__dict__)
                        elif model_list[i-1][0]=='batchnormalization' or model_list[i-1][0]=='layernormalization':
                            tmp = eval('model_norm'+str(i), globals(), self.__dict__)
                            tmp = tmp.contiguous()
                            self.__dict__['tmp'] = tmp
                            exec('model_mamba'+str(i+1)+'=self.mamba'+str(i+1)+'(tmp)', globals(), self.__dict__)
                        elif model_list[i-1][0]=='activation':
                            tmp = eval('model_act'+str(i), globals(), self.__dict__)
                            tmp = tmp.contiguous()
                            self.__dict__['tmp'] = tmp
                            exec('model_mamba'+str(i+1)+'=self.mamba'+str(i+1)+'(tmp)', globals(), self.__dict__)
                        elif model_list[i-1][0]=='dropout':
                            tmp = eval('model_drop'+str(i), globals(), self.__dict__)
                            tmp = tmp.contiguous()
                            self.__dict__['tmp'] = tmp
                            exec('model_mamba'+str(i+1)+'=self.mamba'+str(i+1)+'(tmp)', globals(), self.__dict__)
                        if (self.outputstep is None) and is_last_sequence_layer(model_list,i):
                            if i==len(self.model_list)-1:
                                outputs=eval('model_mamba'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                            else:
                                exec('model_mamba'+str(i+1)+'=model_mamba'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                elif self.model_list[i][0] == 'tcn':
                    if i ==0:
                        exec('model_tcn'+str(i+1)+'=self.tcn'+str(i+1)+'(x)', globals(), self.__dict__)
                        if (self.outputstep is None) and is_last_sequence_layer(model_list,i):
                            if i==len(self.model_list)-1:
                                outputs=eval('model_tcn'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                            else:
                                exec('model_tcn'+str(i+1)+'=model_tcn'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                    else:
                        if model_list[i-1][0] == 'flatten':
                            exec('model_tcn'+str(i+1)+'=self.tcn'+str(i+1)+'(model_fla'+str(i)+')', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'transformer':
                            exec('model_tcn'+str(i+1)+'=self.tcn'+str(i+1)+'(model_transformer'+str(i)+')', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='mamba' or self.model_list[i-1][0]=='mamba2':
                            exec('model_tcn'+str(i+1)+'=self.tcn'+str(i+1)+'(model_mamba'+str(i)+')', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'tcn':
                            exec('model_tcn'+str(i+1)+'=self.tcn'+str(i+1)+'(model_tcn'+str(i)+')', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'lstm':
                            exec('model_tcn'+str(i+1)+'=self.tcn'+str(i+1)+'(model_lstm'+str(i)+')', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'rnn':
                            exec('model_tcn'+str(i+1)+'=self.tcn'+str(i+1)+'(model_rnn'+str(i)+')', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'ltc':
                            exec('model_tcn'+str(i+1)+'=self.tcn'+str(i+1)+'(model_ltc'+str(i)+')', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'gru':
                            exec('model_tcn'+str(i+1)+'=self.tcn'+str(i+1)+'(model_gru'+str(i)+')', globals(), self.__dict__)
                        elif model_list[i-1][0]=='batchnormalization' or model_list[i-1][0]=='layernormalization':
                            exec('model_tcn'+str(i+1)+'=self.tcn'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                        elif model_list[i-1][0]=='activation':
                            exec('model_tcn'+str(i+1)+'=self.tcn'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                        elif model_list[i-1][0]=='dropout':
                            exec('model_tcn'+str(i+1)+'=self.tcn'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                        if (self.outputstep is None) and is_last_sequence_layer(model_list,i):
                            if i==len(self.model_list)-1:
                                outputs=eval('model_tcn'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                            else:
                                exec('model_tcn'+str(i+1)+'=model_tcn'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                elif self.model_list[i][0] == 'lstm':
                    if i ==0:
                        exec('model_lstm'+str(i+1)+',(h_0,c_0)=self.lstm'+str(i+1)+'(x)', globals(), self.__dict__)
                        exec('self.encoder_lstm_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        exec('self.encoder_lstm_c'+str(i+1)+'=c_0', globals(), self.__dict__)
                        if (self.outputstep is None) and is_last_sequence_layer(model_list,i):
                            if i==len(self.model_list)-1:
                                outputs=eval('model_lstm'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                            else:
                                exec('model_lstm'+str(i+1)+'=model_lstm'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                    else:
                        if model_list[i-1][0] == 'flatten':
                            exec('model_lstm'+str(i+1)+',(h_0,c_0)=self.lstm'+str(i+1)+'(model_fla'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_lstm_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                            exec('self.encoder_lstm_c'+str(i+1)+'=c_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'transformer':
                            exec('model_lstm'+str(i+1)+',(h_0,c_0)=self.lstm'+str(i+1)+'(model_transformer'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_lstm_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                            exec('self.encoder_lstm_c'+str(i+1)+'=c_0', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='mamba' or self.model_list[i-1][0]=='mamba2':
                            exec('model_lstm'+str(i+1)+',(h_0,c_0)=self.lstm'+str(i+1)+'(model_mamba'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_lstm_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                            exec('self.encoder_lstm_c'+str(i+1)+'=c_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'tcn':
                            exec('model_lstm'+str(i+1)+',(h_0,c_0)=self.lstm'+str(i+1)+'(model_tcn'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_lstm_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                            exec('self.encoder_lstm_c'+str(i+1)+'=c_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'lstm':
                            exec('model_lstm'+str(i+1)+',(h_0,c_0)=self.lstm'+str(i+1)+'(model_lstm'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_lstm_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                            exec('self.encoder_lstm_c'+str(i+1)+'=c_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'rnn':
                            exec('model_lstm'+str(i+1)+',(h_0,c_0)=self.lstm'+str(i+1)+'(model_rnn'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_lstm_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                            exec('self.encoder_lstm_c'+str(i+1)+'=c_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'ltc':
                            exec('model_lstm'+str(i+1)+',(h_0,c_0)=self.lstm'+str(i+1)+'(model_ltc'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_lstm_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                            exec('self.encoder_lstm_c'+str(i+1)+'=c_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'gru':
                            exec('model_lstm'+str(i+1)+',(h_0,c_0)=self.lstm'+str(i+1)+'(model_gru'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_lstm_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                            exec('self.encoder_lstm_c'+str(i+1)+'=c_0', globals(), self.__dict__)
                        elif model_list[i-1][0]=='batchnormalization' or model_list[i-1][0]=='layernormalization':
                            exec('model_lstm'+str(i+1)+',(h_0,c_0)=self.lstm'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_lstm_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                            exec('self.encoder_lstm_c'+str(i+1)+'=c_0', globals(), self.__dict__)
                        elif model_list[i-1][0]=='activation':
                            exec('model_lstm'+str(i+1)+',(h_0,c_0)=self.lstm'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_lstm_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                            exec('self.encoder_lstm_c'+str(i+1)+'=c_0', globals(), self.__dict__)
                        elif model_list[i-1][0]=='dropout':
                            exec('model_lstm'+str(i+1)+',(h_0,c_0)=self.lstm'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_lstm_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                            exec('self.encoder_lstm_c'+str(i+1)+'=c_0', globals(), self.__dict__)
                        if (self.outputstep is None) and is_last_sequence_layer(model_list,i):
                            if i==len(self.model_list)-1:
                                outputs=eval('model_lstm'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                            else:
                                exec('model_lstm'+str(i+1)+'=model_lstm'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                elif self.model_list[i][0] == 'rnn':
                    if i ==0:
                        exec('model_rnn'+str(i+1)+',h_0=self.rnn'+str(i+1)+'(x)', globals(), self.__dict__)
                        exec('self.encoder_rnn_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        if (self.outputstep is None) and is_last_sequence_layer(model_list,i):
                            if i==len(self.model_list)-1:
                                outputs=eval('model_rnn'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                            else:
                                exec('model_rnn'+str(i+1)+'=model_rnn'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                    else:
                        if model_list[i-1][0] == 'flatten':
                            exec('model_rnn'+str(i+1)+',h_0=self.rnn'+str(i+1)+'(model_fla'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_rnn_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'transformer':
                            exec('model_rnn'+str(i+1)+',h_0=self.rnn'+str(i+1)+'(model_transformer'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_rnn_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='mamba' or self.model_list[i-1][0]=='mamba2':
                            exec('model_rnn'+str(i+1)+',h_0=self.rnn'+str(i+1)+'(model_mamba'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_rnn_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'tcn':
                            exec('model_rnn'+str(i+1)+',h_0=self.rnn'+str(i+1)+'(model_tcn'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_rnn_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'lstm':
                            exec('model_rnn'+str(i+1)+',h_0=self.rnn'+str(i+1)+'(model_lstm'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_rnn_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'rnn':
                            exec('model_rnn'+str(i+1)+',h_0=self.rnn'+str(i+1)+'(model_rnn'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_rnn_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'ltc':
                            exec('model_ltc'+str(i+1)+',h_0=self.ltc'+str(i+1)+'(model_ltc'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_ltc_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'gru':
                            exec('model_rnn'+str(i+1)+',h_0=self.rnn'+str(i+1)+'(model_gru'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_rnn_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0]=='batchnormalization' or model_list[i-1][0]=='layernormalization':
                            exec('model_rnn'+str(i+1)+',h_0=self.rnn'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_rnn_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0]=='activation':
                            exec('model_rnn'+str(i+1)+',h_0=self.rnn'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_rnn_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0]=='dropout':
                            exec('model_rnn'+str(i+1)+',h_0=self.rnn'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_rnn_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        if (self.outputstep is None) and is_last_sequence_layer(model_list,i):
                            if i==len(self.model_list)-1:
                                outputs=eval('model_rnn'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                            else:
                                exec('model_rnn'+str(i+1)+'=model_rnn'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                elif self.model_list[i][0] == 'ltc':
                    if i ==0:
                        exec('model_ltc'+str(i+1)+',h_0=self.ltc'+str(i+1)+'(x)', globals(), self.__dict__)
                        exec('self.encoder_ltc_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        if (self.outputstep is None) and is_last_sequence_layer(model_list,i):
                            if i==len(self.model_list)-1:
                                outputs=eval('model_ltc'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                            else:
                                exec('model_ltc'+str(i+1)+'=model_ltc'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                    else:
                        if model_list[i-1][0] == 'flatten':
                            exec('model_ltc'+str(i+1)+',h_0=self.ltc'+str(i+1)+'(model_fla'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_ltc_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'transformer':
                            exec('model_ltc'+str(i+1)+',h_0=self.ltc'+str(i+1)+'(model_transformer'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_ltc_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='mamba' or self.model_list[i-1][0]=='mamba2':
                            exec('model_ltc'+str(i+1)+',h_0=self.ltc'+str(i+1)+'(model_mamba'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_ltc_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'tcn':
                            exec('model_ltc'+str(i+1)+',h_0=self.ltc'+str(i+1)+'(model_tcn'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_ltc_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'lstm':
                            exec('model_ltc'+str(i+1)+',h_0=self.ltc'+str(i+1)+'(model_lstm'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_ltc_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'rnn':
                            exec('model_ltc'+str(i+1)+',h_0=self.ltc'+str(i+1)+'(model_rnn'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_ltc_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'ltc':
                            exec('model_ltc'+str(i+1)+',h_0=self.ltc'+str(i+1)+'(model_ltc'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_ltc_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'gru':
                            exec('model_ltc'+str(i+1)+',h_0=self.ltc'+str(i+1)+'(model_gru'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_ltc_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0]=='batchnormalization' or model_list[i-1][0]=='layernormalization':
                            exec('model_ltc'+str(i+1)+',h_0=self.ltc'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_ltc_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0]=='activation':
                            exec('model_ltc'+str(i+1)+',h_0=self.ltc'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_ltc_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0]=='dropout':
                            exec('model_ltc'+str(i+1)+',h_0=self.ltc'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_ltc_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        if (self.outputstep is None) and is_last_sequence_layer(model_list,i):
                            if i==len(self.model_list)-1:
                                outputs=eval('model_ltc'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                            else:
                                exec('model_ltc'+str(i+1)+'=model_ltc'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                elif self.model_list[i][0] == 'gru':
                    if i ==0:
                        exec('model_gru'+str(i+1)+',h_0=self.gru'+str(i+1)+'(x)', globals(), self.__dict__)
                        exec('self.encoder_gru_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        if (self.outputstep is None) and is_last_sequence_layer(model_list,i):
                            if i==len(self.model_list)-1:
                                outputs=eval('model_gru'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                            else:
                                exec('model_gru'+str(i+1)+'=model_gru'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                    else:
                        if model_list[i-1][0] == 'flatten':
                            exec('model_gru'+str(i+1)+',h_0=self.gru'+str(i+1)+'(model_fla'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_gru_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'transformer':
                            exec('model_gru'+str(i+1)+',h_0=self.gru'+str(i+1)+'(model_transformer'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_gru_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif self.model_list[i-1][0]=='mamba' or self.model_list[i-1][0]=='mamba2':
                            exec('model_gru'+str(i+1)+',h_0=self.gru'+str(i+1)+'(model_mamba'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_gru_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'tcn':
                            exec('model_gru'+str(i+1)+',h_0=self.gru'+str(i+1)+'(model_tcn'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_gru_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'lstm':
                            exec('model_gru'+str(i+1)+',h_0=self.gru'+str(i+1)+'(model_lstm'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_gru_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'rnn':
                            exec('model_gru'+str(i+1)+',h_0=self.gru'+str(i+1)+'(model_rnn'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_gru_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'ltc':
                            exec('model_gru'+str(i+1)+',h_0=self.gru'+str(i+1)+'(model_ltc'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_gru_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0] == 'gru':
                            exec('model_gru'+str(i+1)+',h_0=self.gru'+str(i+1)+'(model_gru'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_gru_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0]=='batchnormalization' or model_list[i-1][0]=='layernormalization':
                            exec('model_gru'+str(i+1)+',h_0=self.gru'+str(i+1)+'(model_norm'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_gru_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0]=='activation':
                            exec('model_gru'+str(i+1)+',h_0=self.gru'+str(i+1)+'(model_act'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_gru_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        elif model_list[i-1][0]=='dropout':
                            exec('model_gru'+str(i+1)+',h_0=self.gru'+str(i+1)+'(model_drop'+str(i)+')', globals(), self.__dict__)
                            exec('self.encoder_gru_h'+str(i+1)+'=h_0', globals(), self.__dict__)
                        if (self.outputstep is None) and is_last_sequence_layer(model_list,i):
                            if i==len(self.model_list)-1:
                                outputs=eval('model_gru'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
                            else:
                                exec('model_gru'+str(i+1)+'=model_gru'+str(i+1)+'[:,-1,:]', globals(), self.__dict__)
            if self.outputstep is not None:
                if self.model_list[self.last_nlp_index][0] == 'transformer':
                    exec('seq2seq_encoder=model_transformer'+str(self.last_nlp_index+1), globals(), self.__dict__)
                elif self.model_list[self.last_nlp_index][0] == 'mamba' or self.model_list[self.last_nlp_index][0] == 'mamba2':
                    exec('seq2seq_encoder=model_mamba'+str(self.last_nlp_index+1), globals(), self.__dict__)
                elif self.model_list[self.last_nlp_index][0] == 'tcn':
                    exec('seq2seq_encoder=model_tcn'+str(self.last_nlp_index+1), globals(), self.__dict__)
                elif self.model_list[self.last_nlp_index][0] == 'lstm':
                    exec('seq2seq_encoder=model_lstm'+str(self.last_nlp_index+1), globals(), self.__dict__)
                elif self.model_list[self.last_nlp_index][0] == 'rnn':
                    exec('seq2seq_encoder=model_rnn'+str(self.last_nlp_index+1), globals(), self.__dict__)
                elif self.model_list[self.last_nlp_index][0] == 'ltc':
                    exec('seq2seq_encoder=model_ltc'+str(self.last_nlp_index+1), globals(), self.__dict__)
                elif self.model_list[self.last_nlp_index][0] == 'gru':
                    exec('seq2seq_encoder=model_gru'+str(self.last_nlp_index+1), globals(), self.__dict__)
                else:
                    raise ValueError('Seq2Seq encoder 最后一层必须是 NLP 层。')
                seq2seq_encoder = self.__dict__['seq2seq_encoder']

                if seq2seq_encoder.dim() != 3:
                    raise ValueError('Seq2Seq encoder 输出必须是 3D 张量 (batch, timestep, feature)。')

                seq2seq_memory = seq2seq_encoder

                seq2seq_decoder = seq2seq_encoder[:, -1:, :].repeat(1, self.outputstep, 1).contiguous()
                seq2seq_position = torch.arange(0, self.outputstep, device=seq2seq_decoder.device).long().unsqueeze(0).repeat(seq2seq_decoder.shape[0], 1)
                self.__dict__['seq2seq_memory'] = seq2seq_memory
                self.__dict__['seq2seq_decoder'] = seq2seq_decoder
                self.__dict__['seq2seq_position'] = seq2seq_position

                for decoder_order, nlp_i in enumerate(self.nlp_indices):
                    nlp_name = self.model_list[nlp_i][0]
                    if nlp_name == 'transformer':
                        exec('seq2seq_decoder_embedding=self.decoder_embedding'+str(nlp_i+1)+'(seq2seq_position)', globals(), self.__dict__)
                        exec('seq2seq_decoder=self.decoder_transformer'+str(nlp_i+1)+'(seq2seq_decoder+seq2seq_decoder_embedding, seq2seq_memory)', globals(), self.__dict__)
                    elif nlp_name == 'mamba' or nlp_name == 'mamba2':
                        exec('seq2seq_decoder=self.decoder_mamba'+str(nlp_i+1)+'(seq2seq_decoder.contiguous())', globals(), self.__dict__)
                    elif nlp_name == 'tcn':
                        exec('seq2seq_decoder=self.decoder_tcn'+str(nlp_i+1)+'(seq2seq_decoder)', globals(), self.__dict__)
                    elif nlp_name == 'lstm':
                        exec('seq2seq_lstm_h0=self.encoder_lstm_h'+str(nlp_i+1), globals(), self.__dict__)
                        exec('seq2seq_lstm_c0=self.encoder_lstm_c'+str(nlp_i+1), globals(), self.__dict__)
                        exec('seq2seq_decoder,(seq2seq_h0,seq2seq_c0)=self.decoder_lstm'+str(nlp_i+1)+'(seq2seq_decoder,(seq2seq_lstm_h0.contiguous(),seq2seq_lstm_c0.contiguous()))', globals(), self.__dict__)
                    elif nlp_name == 'rnn':
                        exec('seq2seq_rnn_h0=self.encoder_rnn_h'+str(nlp_i+1), globals(), self.__dict__)
                        exec('seq2seq_decoder,seq2seq_h0=self.decoder_rnn'+str(nlp_i+1)+'(seq2seq_decoder,seq2seq_rnn_h0.contiguous())', globals(), self.__dict__)
                    elif nlp_name == 'ltc':
                        exec('seq2seq_ltc_h0=self.encoder_ltc_h'+str(nlp_i+1), globals(), self.__dict__)
                        exec('seq2seq_decoder,seq2seq_h0=self.decoder_ltc'+str(nlp_i+1)+'(seq2seq_decoder,seq2seq_ltc_h0.contiguous())', globals(), self.__dict__)
                    elif nlp_name == 'gru':
                        exec('seq2seq_gru_h0=self.encoder_gru_h'+str(nlp_i+1), globals(), self.__dict__)
                        exec('seq2seq_decoder,seq2seq_h0=self.decoder_gru'+str(nlp_i+1)+'(seq2seq_decoder,seq2seq_gru_h0.contiguous())', globals(), self.__dict__)
                    seq2seq_decoder = self.__dict__['seq2seq_decoder']
                    self.__dict__['seq2seq_decoder'] = seq2seq_decoder

                seq2seq_head = seq2seq_decoder
                self.__dict__['seq2seq_head'] = seq2seq_head
                for i in range(self.first_head_index, len(self.model_list)):
                    self.__dict__['i'] = i
                    if self.model_list[i][0] == 'fc':
                        exec('seq2seq_head=self.fc'+str(i+1)+'(seq2seq_head)', globals(), self.__dict__)
                    elif self.model_list[i][0] == 'batchnormalization':
                        if seq2seq_head.dim() == 3:
                            seq2seq_head_shape = seq2seq_head.shape
                            self.__dict__['seq2seq_head_shape'] = seq2seq_head_shape
                            exec('seq2seq_head=self.norm'+str(i+1)+'(seq2seq_head.reshape(-1,seq2seq_head_shape[-1])).reshape(seq2seq_head_shape)', globals(), self.__dict__)
                        else:
                            exec('seq2seq_head=self.norm'+str(i+1)+'(seq2seq_head)', globals(), self.__dict__)
                    elif self.model_list[i][0] == 'layernormalization':
                        exec('seq2seq_head=self.norm'+str(i+1)+'(seq2seq_head)', globals(), self.__dict__)
                    elif self.model_list[i][0] == 'activation':
                        exec('seq2seq_head=self.act'+str(i+1)+'(seq2seq_head)', globals(), self.__dict__)
                    elif self.model_list[i][0] == 'dropout':
                        exec('seq2seq_head=self.drop'+str(i+1)+'(seq2seq_head)', globals(), self.__dict__)
                    else:
                        raise ValueError('outputstep 不为 None 时，最后一个 NLP 层之后仅支持 fc、batchnormalization、layernormalization、activation、dropout 作为 decoder head。')
                    seq2seq_head = self.__dict__['seq2seq_head']
                    self.__dict__['seq2seq_head'] = seq2seq_head

                outputs = seq2seq_head
                if self.seq2seq_target_shape is not None and len(self.seq2seq_target_shape) > 1:
                    target_flat = int(np.prod(self.seq2seq_target_shape))
                    if outputs.dim() == 3 and outputs.shape[-1] == target_flat:
                        outputs = outputs.reshape(outputs.shape[0], outputs.shape[1], *self.seq2seq_target_shape)

            return outputs
    if k_fold!=None:
        models=[]
        for i in range(k_fold):
            models.append(Model(model_list,timestep,dilations,ifmulti_scale,num_channels,trainx[0:2],cov_padding,cov_strides,pooling_strides,dropout,residual_deep,residual_activation,residual_norm,vit_patch_size,vit_dim_feedforward,vit_deep,vit_num_heads,swin_deep,swin_num_heads,cbam_reduction,chebnet_k,gsage_aggr,gsage_lstm_units,transformer_encoder_deep,transformer_num_heads,transformer_dim_feedforward,tcn_kernel_size,nlp_units,nlp_num_layers,ltc_solver,ltc_ode_unfolds,ltc_dt,ltc_tau_min,ltc_activation,ltc_dropout,mamba_state,mamba_conv,mamba_expand,activation,adj_prior=adj_prior,outputstep=outputstep,seq2seq_target_shape=(trainy.shape[2:] if outputstep is not None else None)))
    else:
        model=Model(model_list,timestep,dilations,ifmulti_scale,num_channels,trainx[0:2],cov_padding,cov_strides,pooling_strides,dropout,residual_deep,residual_activation,residual_norm,vit_patch_size,vit_dim_feedforward,vit_deep,vit_num_heads,swin_deep,swin_num_heads,cbam_reduction,chebnet_k,gsage_aggr,gsage_lstm_units,transformer_encoder_deep,transformer_num_heads,transformer_dim_feedforward,tcn_kernel_size,nlp_units,nlp_num_layers,ltc_solver,ltc_ode_unfolds,ltc_dt,ltc_tau_min,ltc_activation,ltc_dropout,mamba_state,mamba_conv,mamba_expand,activation,adj_prior=adj_prior,outputstep=outputstep,seq2seq_target_shape=(trainy.shape[2:] if outputstep is not None else None))
    if if_best_mode!='no':
        if k_fold!=None:
            for i in range(k_fold):
                models[i].load_state_dict(torch.load(modelpath+'_'+str(i+1)+'.pth'))
        else:
            model.load_state_dict(torch.load(modelpath+'.pth'))
    if k_fold!=None:
        for i in range(k_fold):
            models[i].to(devices)
    else:
        model.to(devices)
    if k_fold!=None:
        opts=[]
        for i in range(k_fold):
            if optimizer == 'SGD':
                opts.append(SGD(models[i].parameters(), lr=learning_rate))
            elif optimizer == 'Adam':
                opts.append(Adam(models[i].parameters(), lr=learning_rate))
            elif optimizer == 'Nadam':
                opts.append(NAdam(models[i].parameters(), lr=learning_rate))
    else:
        if optimizer == 'SGD':
            opt = SGD(model.parameters(), lr=learning_rate)
        elif optimizer == 'Adam':
            opt = Adam(model.parameters(), lr=learning_rate)
        elif optimizer == 'Nadam':
            opt = NAdam(model.parameters(), lr=learning_rate)
    if if_best_mode!='no':
        if k_fold!=None:
            for i in range(k_fold):
                opts[i].load_state_dict(torch.load(modelpath+'_'+str(i+1)+'_opt.pth'))
        else:
            opt.load_state_dict(torch.load(modelpath+'_opt.pth'))

    if k_fold!=None:
        schedulers=[]
        for i in range(k_fold):
            if lr_decay is None:
                schedulers.append(None)
            else:
                schedulers.append(ExponentialLR(opts[i], gamma=float(lr_decay)))
    else:
        if lr_decay is None:
            scheduler=None
        else:
            scheduler=ExponentialLR(opt, gamma=float(lr_decay))

    if if_print_model=='yes':
        if k_fold!=None:
            print(models[0])
        else:
            print(model)
    if ifrandom_split!='just_model':
        trainx=np.nan_to_num(trainx,nan=0)
        testx=np.nan_to_num(testx,nan=0)
        if ifrandom_split!='all_test':
            if valid_size!=None or k_fold !=None:
                if k_fold!=None:
                    kf = KFold(n_splits=k_fold, shuffle=True, random_state=25)
                    for fold_no, (train_idx, val_idx) in enumerate(kf.split(trainx, trainy)):
                        X_train_fold, y_train_fold,position_train_fold = trainx[train_idx], trainy[train_idx],train_position[train_idx]
                        X_val_fold, y_val_fold,position_val_fold = trainx[val_idx], trainy[val_idx],train_position[val_idx]
                        if fold_no==0:
                            train_loss=np.zeros((k_fold,int(np.ceil(X_train_fold.shape[0]/batch_size))))
                            train_metric=np.zeros((k_fold,int(np.ceil(X_train_fold.shape[0]/batch_size))))
                            test_loss=np.zeros((k_fold,int(np.ceil(X_val_fold.shape[0]/batch_size))))
                            test_metric=np.zeros((k_fold,int(np.ceil(X_val_fold.shape[0]/batch_size))))
                        if if_early_stopping!=None:
                            early_stopping = EarlyStopping(patience=if_early_stopping)
                        for i in range(epochs):
                            start = datetime.datetime.now()
                            for j in range(int(np.ceil(X_train_fold.shape[0]/batch_size))):
                                if j == int(X_train_fold.shape[0]/batch_size) :
                                    train_output = models[fold_no](torch.tensor(X_train_fold[j*batch_size:],dtype=torch.float32,device=devices),torch.tensor(position_train_fold[j*batch_size:],dtype=torch.int64,device=devices))
                                    if task_mode=='multi_classify':
                                        if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                            train_losss = torch.mean(loss(train_output, torch.tensor(y_train_fold[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        elif loss_function=='NLLLoss':
                                            train_losss = torch.mean(loss(torch.log(train_output).contiguous(), torch.tensor(y_train_fold[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                            train_metrics = torch.mean(metric(torch.argmax(train_output, dim=1), torch.tensor(y_train_fold[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        elif metrics=='CrossEntropyLoss':
                                            train_metrics = torch.mean(metric(train_output, torch.tensor(y_train_fold[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        elif metrics=='NLLLoss':
                                            train_metrics = torch.mean(metric(torch.log(train_output).contiguous(), torch.tensor(y_train_fold[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    else:
                                        train_losss = torch.mean(loss(train_output, torch.tensor(y_train_fold[j*batch_size:],dtype=torch.float32,device=devices)))
                                        train_metrics = torch.mean(metric(train_output, torch.tensor(y_train_fold[j*batch_size:],dtype=torch.float32,device=devices)))
                                else:
                                    train_output = models[fold_no](torch.tensor(X_train_fold[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices),torch.tensor(position_train_fold[j*batch_size:(j+1)*batch_size],dtype=torch.int64,device=devices))
                                    if task_mode=='multi_classify':
                                        if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                            train_losss = torch.mean(loss(train_output, torch.tensor(y_train_fold[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        elif loss_function=='NLLLoss':
                                            train_losss = torch.mean(loss(torch.log(train_output).contiguous(), torch.tensor(y_train_fold[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                            train_metrics = torch.mean(metric(torch.argmax(train_output, dim=1), torch.tensor(y_train_fold[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        elif metrics=='CrossEntropyLoss':
                                            train_metrics = torch.mean(metric(train_output, torch.tensor(y_train_fold[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        elif metrics=='NLLLoss':
                                            train_metrics = torch.mean(metric(torch.log(train_output).contiguous(), torch.tensor(y_train_fold[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    else:
                                        train_losss = torch.mean(loss(train_output, torch.tensor(y_train_fold[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices)))
                                        train_metrics = torch.mean(metric(train_output, torch.tensor(y_train_fold[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices)))
                                train_losss = add_regularization_loss(train_losss, models[fold_no])
                                train_loss[fold_no,j]=np.array(train_losss.item())
                                train_metric[fold_no,j]=np.array(train_metrics.item())
                                opts[fold_no].zero_grad()
                                train_losss.backward(retain_graph=True)
                                opts[fold_no].step()
                            with torch.no_grad():
                                for k in range(int(np.ceil(X_val_fold.shape[0]/batch_size))):
                                    if k == int(X_val_fold.shape[0]/batch_size):
                                        test_output = models[fold_no](torch.tensor(X_val_fold[k*batch_size:],dtype=torch.float32,device=devices),torch.tensor(position_val_fold[k*batch_size:],dtype=torch.int64,device=devices))
                                        if task_mode=='multi_classify':
                                            if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                                test_losss= torch.mean(loss(test_output,torch.tensor(y_val_fold[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                            elif loss_function=='NLLLoss':
                                                test_losss= torch.mean(loss(torch.log(test_output).contiguous(),torch.tensor(y_val_fold[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                            if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                                test_metrics=torch.mean(metric(torch.argmax(test_output, dim=1),torch.tensor(y_val_fold[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                            elif metrics=='CrossEntropyLoss':
                                                test_metrics=torch.mean(metric(test_output,torch.tensor(y_val_fold[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                            elif metrics=='NLLLoss':
                                                test_metrics=torch.mean(metric(torch.log(test_output).contiguous(),torch.tensor(y_val_fold[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        else:
                                            test_losss= torch.mean(loss(test_output,torch.tensor(y_val_fold[k*batch_size:],dtype=torch.float32,device=devices)))
                                            test_metrics=torch.mean(metric(test_output,torch.tensor(y_val_fold[k*batch_size:],dtype=torch.float32,device=devices)))
                                    else:
                                        test_output = models[fold_no](torch.tensor(X_val_fold[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices),torch.tensor(position_val_fold[k*batch_size:(k+1)*batch_size],dtype=torch.int64,device=devices))
                                        if task_mode=='multi_classify':
                                            if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                                test_losss= torch.mean(loss(test_output,torch.tensor(y_val_fold[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                            elif loss_function=='NLLLoss':
                                                test_losss= torch.mean(loss(torch.log(test_output).contiguous(),torch.tensor(y_val_fold[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                            if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                                test_metrics=torch.mean(metric(torch.argmax(test_output, dim=1),torch.tensor(y_val_fold[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                            elif metrics=='CrossEntropyLoss':
                                                test_metrics=torch.mean(metric(test_output,torch.tensor(y_val_fold[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                            elif metrics=='NLLLoss':
                                                test_metrics=torch.mean(metric(torch.log(test_output).contiguous(),torch.tensor(y_val_fold[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        else:
                                            test_losss= torch.mean(loss(test_output,torch.tensor(y_val_fold[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices)))
                                            test_metrics=torch.mean(metric(test_output,torch.tensor(y_val_fold[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices)))
                                    test_loss[fold_no,k]=np.array(test_losss.item())
                                    test_metric[fold_no,k]=np.array(test_metrics.item())
                            if if_early_stopping!=None:
                                early_stopping(np.nanmean(test_loss[fold_no,:]), models[fold_no])
                                if early_stopping.early_stop:
                                    early_stopping.load_best_checkpoint(models[fold_no])
                                    break
                            if schedulers[fold_no] is not None:
                                schedulers[fold_no].step()
                            end = datetime.datetime.now()
                            print('第',i+1,'次训练loss:',np.nanmean(train_loss[fold_no,:]),'，metric:',np.nanmean(train_metric[fold_no,:]),'  第',i+1,'次测试loss:',np.nanmean(test_loss[fold_no,:]),'，metric：',np.nanmean(test_metric[fold_no,:]),'训练用时:',end - start)
                else:
                    if if_early_stopping!=None:
                        early_stopping = EarlyStopping(patience=if_early_stopping)
                    for i in range(epochs):
                        start = datetime.datetime.now()
                        if i==0:
                            if ifrandom_split=='yes':
                                trainy,validy,trainx,validx,train_position,valid_position = train_test_split(trainy,trainx,train_position,test_size=valid_size/(1-test_size),random_state=25)
                            else:
                                index=int((1-valid_size/(1-test_size))*trainy.shape[0])
                                validy=trainy[index:]
                                trainy=trainy[:index]
                                validx=trainx[index:]
                                trainx=trainx[:index]
                                valid_position=train_position[index:]
                                train_position=train_position[:index]
                        train_loss=np.zeros((int(np.ceil(trainx.shape[0]/batch_size))))
                        train_metric=np.zeros((int(np.ceil(trainx.shape[0]/batch_size))))
                        test_loss=np.zeros((int(np.ceil(validx.shape[0]/batch_size))))
                        test_metric=np.zeros((int(np.ceil(validx.shape[0]/batch_size))))
                        for j in range(int(np.ceil(trainx.shape[0]/batch_size))):
                            if j == int(trainx.shape[0]/batch_size) :
                                train_output = model(torch.tensor(trainx[j*batch_size:],dtype=torch.float32,device=devices),torch.tensor(train_position[j*batch_size:],dtype=torch.int64,device=devices))
                                if task_mode=='multi_classify':
                                    if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                        train_losss = torch.mean(loss(train_output, torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    elif loss_function=='NLLLoss':
                                        train_losss = torch.mean(loss(torch.log(train_output).contiguous(), torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                        train_metrics = torch.mean(metric(torch.argmax(train_output, dim=1), torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    elif metrics=='CrossEntropyLoss':
                                        train_metrics = torch.mean(metric(train_output, torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    elif metrics=='NLLLoss':
                                        train_metrics = torch.mean(metric(torch.log(train_output).contiguous(), torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                else:
                                    train_losss = torch.mean(loss(train_output, torch.tensor(trainy[j*batch_size:],dtype=torch.float32,device=devices)))
                                    train_metrics = torch.mean(metric(train_output, torch.tensor(trainy[j*batch_size:],dtype=torch.float32,device=devices)))
                            else:
                                train_output = model(torch.tensor(trainx[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices),torch.tensor(train_position[j*batch_size:(j+1)*batch_size],dtype=torch.int64,device=devices))
                                if task_mode=='multi_classify':
                                    if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                        train_losss = torch.mean(loss(train_output, torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    elif loss_function=='NLLLoss':
                                        train_losss = torch.mean(loss(torch.log(train_output).contiguous(), torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                        train_metrics = torch.mean(metric(torch.argmax(train_output, dim=1), torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    elif metrics=='CrossEntropyLoss':
                                        train_metrics = torch.mean(metric(train_output, torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    elif metrics=='NLLLoss':
                                        train_metrics = torch.mean(metric(torch.log(train_output).contiguous(), torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                else:
                                    train_losss = torch.mean(loss(train_output, torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices)))
                                    train_metrics = torch.mean(metric(train_output, torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices)))
                            train_losss = add_regularization_loss(train_losss, model)
                            train_loss[j]=np.array(train_losss.item())
                            train_metric[j]=np.array(train_metrics.item())
                            opt.zero_grad()
                            train_losss.backward(retain_graph=True)
                            opt.step()
                        with torch.no_grad():
                            for k in range(int(np.ceil(validx.shape[0]/batch_size))):
                                if k == int(validx.shape[0]/batch_size) :
                                    test_output = model(torch.tensor(validx[k*batch_size:],dtype=torch.float32,device=devices),torch.tensor(valid_position[k*batch_size:],dtype=torch.int64,device=devices))
                                    if task_mode=='multi_classify':
                                        if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                            test_losss= torch.mean(loss(test_output,torch.tensor(validy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        elif loss_function=='NLLLoss':
                                            test_losss= torch.mean(loss(torch.log(test_output).contiguous(),torch.tensor(validy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                            test_metrics=torch.mean(metric(torch.argmax(test_output, dim=1),torch.tensor(validy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        elif metrics=='CrossEntropyLoss':
                                            test_metrics=torch.mean(metric(test_output,torch.tensor(validy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        elif metrics=='NLLLoss':
                                            test_metrics=torch.mean(metric(torch.log(test_output).contiguous(),torch.tensor(validy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    else:
                                        test_losss= torch.mean(loss(test_output,torch.tensor(validy[k*batch_size:],dtype=torch.float32,device=devices)))
                                        test_metrics=torch.mean(metric(test_output,torch.tensor(validy[k*batch_size:],dtype=torch.float32,device=devices)))
                                else:
                                    test_output = model(torch.tensor(validx[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices),torch.tensor(valid_position[k*batch_size:(k+1)*batch_size],dtype=torch.int64,device=devices))
                                    if task_mode=='multi_classify':
                                        if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                            test_losss= torch.mean(loss(test_output,torch.tensor(validy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        elif loss_function=='NLLLoss':
                                            test_losss= torch.mean(loss(torch.log(test_output).contiguous(),torch.tensor(validy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                            test_metrics=torch.mean(metric(torch.argmax(test_output, dim=1),torch.tensor(validy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        elif metrics=='CrossEntropyLoss':
                                            test_metrics=torch.mean(metric(test_output,torch.tensor(validy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        elif metrics=='NLLLoss':
                                            test_metrics=torch.mean(metric(torch.log(test_output).contiguous(),torch.tensor(validy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    else:
                                        test_losss= torch.mean(loss(test_output,torch.tensor(validy[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices)))
                                        test_metrics=torch.mean(metric(test_output,torch.tensor(validy[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices)))
                                test_loss[k]=np.array(test_losss.item())
                                test_metric[k]=np.array(test_metrics.item())
                        if if_early_stopping!=None:
                            early_stopping(np.nanmean(test_loss), model)
                            if early_stopping.early_stop:
                                early_stopping.load_best_checkpoint(model)
                                break
                        if scheduler is not None:
                            scheduler.step()
                        end = datetime.datetime.now()
                        print('第',i+1,'次训练loss:',np.nanmean(train_loss),'，metric:',np.nanmean(train_metric),'  第',i+1,'次测试loss:',np.nanmean(test_loss),'，metric：',np.nanmean(test_metric),'训练用时:',end - start)
            else:
                if if_early_stopping!=None:
                    early_stopping = EarlyStopping(patience=if_early_stopping)
                for i in range(epochs):
                    start = datetime.datetime.now()
                    train_loss=np.zeros((int(np.ceil(trainx.shape[0]/batch_size))))
                    train_metric=np.zeros((int(np.ceil(trainx.shape[0]/batch_size))))
                    test_loss=np.zeros((int(np.ceil(testx.shape[0]/batch_size))))
                    test_metric=np.zeros((int(np.ceil(testx.shape[0]/batch_size))))
                    for j in range(int(np.ceil(trainx.shape[0]/batch_size))):
                        if j == int(trainx.shape[0]/batch_size):
                            train_output = model(torch.tensor(trainx[j*batch_size:],dtype=torch.float32,device=devices),torch.tensor(train_position[j*batch_size:],dtype=torch.int64,device=devices))
                            if task_mode=='multi_classify':
                                if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                    train_losss = torch.mean(loss(train_output, torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                elif loss_function=='NLLLoss':
                                    train_losss = torch.mean(loss(torch.log(train_output).contiguous(), torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                    train_metrics = torch.mean(metric(torch.argmax(train_output, dim=1), torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                elif metrics=='CrossEntropyLoss':
                                    train_metrics = torch.mean(metric(train_output, torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                elif metrics=='NLLLoss':
                                    train_metrics = torch.mean(metric(torch.log(train_output).contiguous(), torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                            else:
                                train_losss = torch.mean(loss(train_output, torch.tensor(trainy[j*batch_size:],dtype=torch.float32,device=devices)))
                                train_metrics = torch.mean(metric(train_output, torch.tensor(trainy[j*batch_size:],dtype=torch.float32,device=devices)))
                        else:
                            train_output = model(torch.tensor(trainx[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices),torch.tensor(train_position[j*batch_size:(j+1)*batch_size],dtype=torch.int64,device=devices))
                            if task_mode=='multi_classify':
                                if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                    train_losss = torch.mean(loss(train_output, torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                elif loss_function=='NLLLoss':
                                    train_losss = torch.mean(loss(torch.log(train_output).contiguous(), torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                    train_metrics = torch.mean(metric(torch.argmax(train_output, dim=1), torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                elif metrics=='CrossEntropyLoss':
                                    train_metrics = torch.mean(metric(train_output, torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                elif metrics=='NLLLoss':
                                    train_metrics = torch.mean(metric(torch.log(train_output).contiguous(), torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                            else:
                                train_losss = torch.mean(loss(train_output, torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices)))
                                train_metrics = torch.mean(metric(train_output, torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices)))
                        train_losss = add_regularization_loss(train_losss, model)
                        train_loss[j]=np.array(train_losss.item())
                        train_metric[j]=np.array(train_metrics.item())
                        opt.zero_grad()
                        train_losss.backward(retain_graph=True)
                        opt.step()
                    with torch.no_grad():
                        for k in range(int(np.ceil(testx.shape[0]/batch_size))):
                            if k == int(testx.shape[0]/batch_size) :
                                test_output = model(torch.tensor(testx[k*batch_size:],dtype=torch.float32,device=devices),torch.tensor(test_position[k*batch_size:],dtype=torch.int64,device=devices))
                                if task_mode=='multi_classify':
                                    if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                        test_losss= torch.mean(loss(test_output,torch.tensor(testy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    elif loss_function=='NLLLoss':
                                        test_losss= torch.mean(loss(torch.log(test_output).contiguous(),torch.tensor(testy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                        test_metrics=torch.mean(metric(torch.argmax(test_output, dim=1),torch.tensor(testy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    elif metrics=='CrossEntropyLoss':
                                        test_metrics=torch.mean(metric(test_output,torch.tensor(testy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    elif metrics=='NLLLoss':
                                        test_metrics=torch.mean(metric(torch.log(test_output).contiguous(),torch.tensor(testy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                else:
                                    test_losss= torch.mean(loss(test_output,torch.tensor(testy[k*batch_size:],dtype=torch.float32,device=devices)))
                                    test_metrics=torch.mean(metric(test_output,torch.tensor(testy[k*batch_size:],dtype=torch.float32,device=devices)))
                            else:
                                test_output = model(torch.tensor(testx[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices),torch.tensor(test_position[k*batch_size:(k+1)*batch_size],dtype=torch.int64,device=devices))
                                if task_mode=='multi_classify':
                                    if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                        test_losss= torch.mean(loss(test_output,torch.tensor(testy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    elif loss_function=='NLLLoss':
                                        test_losss= torch.mean(loss(torch.log(test_output).contiguous(),torch.tensor(testy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                        test_metrics=torch.mean(metric(torch.argmax(test_output, dim=1),torch.tensor(testy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    elif metrics=='CrossEntropyLoss':
                                        test_metrics=torch.mean(metric(test_output,torch.tensor(testy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    elif metrics=='NLLLoss':
                                        test_metrics=torch.mean(metric(torch.log(test_output).contiguous(),torch.tensor(testy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                else:
                                    test_losss= torch.mean(loss(test_output,torch.tensor(testy[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices)))
                                    test_metrics=torch.mean(metric(test_output,torch.tensor(testy[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices)))
                            test_loss[k]=np.array(test_losss.item())
                            test_metric[k]=np.array(test_metrics.item())
                    if if_early_stopping!=None:
                        early_stopping(np.nanmean(test_loss), model)
                        if early_stopping.early_stop:
                            early_stopping.load_best_checkpoint(model)
                            break
                    if scheduler is not None:
                        scheduler.step()
                    end = datetime.datetime.now()
                    print('第',i+1,'次训练loss:',np.nanmean(train_loss),'，metric:',np.nanmean(train_metric),'  第',i+1,'次测试loss:',np.nanmean(test_loss),'，metric：',np.nanmean(test_metric),'训练用时:',end - start)
        if k_fold!=None:
            if task_mode=='multi_classify':
                predicty=np.zeros((k_fold,testy.shape[0],int(np.max(vy))+1))
            else:
                if outputstep is not None:
                    predicty=np.zeros((k_fold,*testy.shape), dtype=np.float32)
                else:
                    predicty=np.zeros((k_fold,testy.shape[0],testy.shape[1]), dtype=np.float32)
        else:
            if task_mode=='multi_classify':
                predicty=np.zeros((testy.shape[0],int(np.max(vy))+1))
            else:
                if outputstep is not None:
                    predicty=np.zeros_like(testy,dtype=np.float32)
                else:
                    predicty=np.zeros((testy.shape[0],testy.shape[1]), dtype=np.float32)
        if k_fold!=None:
            for i in range(k_fold):
                for k in range(int(np.ceil(testx.shape[0]/batch_size))):
                    if k == int(testx.shape[0]/batch_size) :
                        predicty_batch = models[i](torch.tensor(testx[k*batch_size:],dtype=torch.float32,device=devices),torch.tensor(test_position[k*batch_size:],dtype=torch.int64,device=devices))
                        predicty[i,k*batch_size:]=predicty_batch.cpu().detach().numpy()
                    else:
                        predicty_batch = models[i](torch.tensor(testx[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices),torch.tensor(test_position[k*batch_size:(k+1)*batch_size],dtype=torch.int64,device=devices))
                        predicty[i,k*batch_size:(k+1)*batch_size]=predicty_batch.cpu().detach().numpy()
            predicty=np.nanmean(predicty,axis=0)
        else:
            for k in range(int(np.ceil(testx.shape[0]/batch_size))):
                if k == int(testx.shape[0]/batch_size) :
                    predicty_batch = model(torch.tensor(testx[k*batch_size:],dtype=torch.float32,device=devices),torch.tensor(test_position[k*batch_size:],dtype=torch.int64,device=devices))
                    predicty[k*batch_size:]=predicty_batch.cpu().detach().numpy()
                else:
                    predicty_batch = model(torch.tensor(testx[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices),torch.tensor(test_position[k*batch_size:(k+1)*batch_size],dtype=torch.int64,device=devices))
                    predicty[k*batch_size:(k+1)*batch_size]=predicty_batch.cpu().detach().numpy()
        predicty = np.nan_to_num(predicty,nan=0)

        predicty_return = _adl_restore_y_array(predicty)
        testy_return = _adl_restore_y_array(testy)

        if task_mode=='regression':
            if predicty_return.shape != testy_return.shape:
                raise ValueError(
                    f'预测值与真实值恢复后的形状不一致：predicty={predicty_return.shape}, '
                    f'testy={testy_return.shape}。请检查 model_list 最后一层 fc 的输出节点数。'
                )

            metric_output_shape = tuple(testy_return.shape[1:])
            predicty_for_metric = predicty_return.reshape(predicty_return.shape[0], -1)
            testy_for_metric = testy_return.reshape(testy_return.shape[0], -1)
            r_flat = np.zeros(testy_for_metric.shape[1], dtype=np.float64)
            p_flat = np.ones(testy_for_metric.shape[1], dtype=np.float64)

            for i in range(testy_for_metric.shape[1]):
                pred_i = predicty_for_metric[:, i]
                true_i = testy_for_metric[:, i]
                valid_i = np.isfinite(pred_i) & np.isfinite(true_i)
                if np.count_nonzero(valid_i) < 2:
                    r_flat[i] = 0.0
                    p_flat[i] = 1.0
                    continue
                pred_valid = pred_i[valid_i]
                true_valid = true_i[valid_i]
                if np.nanstd(pred_valid) == 0 or np.nanstd(true_valid) == 0:
                    r_flat[i] = 0.0
                    p_flat[i] = 1.0
                    continue
                r_flat[i], p_flat[i] = pearsonr(pred_valid, true_valid)

            r = np.nan_to_num(r_flat, nan=0.0, posinf=0.0, neginf=0.0).reshape(metric_output_shape)
            p = np.nan_to_num(p_flat, nan=1.0, posinf=1.0, neginf=1.0).reshape(metric_output_shape)
        elif task_mode=='binary_classify':
            accuracy=np.zeros((testy.shape[1]))
            recall=np.zeros((testy.shape[1]))
            precision=np.zeros((testy.shape[1]))
            f1=np.zeros((testy.shape[1]))
            for i in range(predicty.shape[1]):
                predicty[:,i]=[int(round(predicty[j,i],0)) for j in range(predicty.shape[0])]
            r=np.zeros((testy.shape[1]))
            for i in range(testy.shape[1]):
                if metrics=='Recall':
                    r[i]=recall_score(testy[:,i], predicty[:,i])
                elif metrics=='Precision':
                    r[i]=precision_score(testy[:,i], predicty[:,i])
                else:
                    r[i]=accuracy_score(testy[:,i], predicty[:,i])
                recall[i]=recall_score(testy[:,i], predicty[:,i])
                precision[i]=precision_score(testy[:,i], predicty[:,i])
                accuracy[i]=accuracy_score(testy[:,i], predicty[:,i])
                f1[i]=f1_score(testy[:,i], predicty[:,i])
            p=0
        elif task_mode=='multi_classify':
            r=np.zeros((testy.shape[1]))
            for i in range(testy.shape[1]):
                r[i]=accuracy_score(testy[:,i], np.argmax(predicty,axis=1))
            p=0
        if ifmute == 'no':
            if task_mode=='regression':
                print('相关系数',np.nanmean(r))
            elif task_mode=='binary_classify':
                print('召回率+精确率',np.nanmean(f1),'准确率',np.nanmean(accuracy),'召回率',np.nanmean(recall),'精确率',np.nanmean(precision))
            elif task_mode=='multi_classify':
                print('准确率',np.nanmean(r))
        current_testx_data = None
        current_test_position_data = None
        sample_x_shape = None
        sample_pos_shape = None
        num_model_outputs = None
        combined_input_for_shap_data = None

        if (ifheatmap == 'yes' and testx.ndim > 3) or ifweight == 'yes' or ifweight == 'shap':
            current_testx_data = testx
            current_test_position_data = test_position
            if testx.shape[0] >= 100:
                index = np.random.randint(0, testx.shape[0], size=100)
                current_testx_data = testx[index, ]
                current_test_position_data = test_position[index, ]

            sample_x_shape = current_testx_data.shape[1:]
            sample_pos_shape = current_test_position_data.shape[1:]

            flat_current_testx_tensor = torch.tensor(current_testx_data, dtype=torch.float32, device=devices).flatten(start_dim=1)
            flat_current_test_position_tensor = torch.tensor(current_test_position_data, dtype=torch.float32, device=devices).flatten(start_dim=1)
            combined_input_for_shap_data = torch.cat((flat_current_testx_tensor, flat_current_test_position_tensor), dim=1)

            if outputstep is not None:
                num_model_outputs = int(np.prod(testy.shape[1:]))
            else:
                num_model_outputs = testy.shape[1]

        has_mamba_layer = find_string_in_nested_list('mamba', model_list) or find_string_in_nested_list('mamba2', model_list)
        has_transformer_layer = find_string_in_nested_list('transformer', model_list)
        has_rnn_like_layer = find_string_in_nested_list('lstm', model_list) or \
                            find_string_in_nested_list('gru', model_list) or \
                            find_string_in_nested_list('rnn', model_list) or \
                            find_string_in_nested_list('ltc', model_list) or \
                            find_string_in_nested_list('tcn', model_list)
        has_vit_swin_layer = find_string_in_nested_list('vit', model_list) or \
                            find_string_in_nested_list('swin', model_list)
        has_cbam_layer = find_string_in_nested_list('cbam', model_list)

        use_gradient_explainer = has_transformer_layer or has_rnn_like_layer or has_vit_swin_layer or has_mamba_layer or has_cbam_layer
        disable_cudnn_for_rnn_tcn = has_rnn_like_layer

        def _compute_shap_values_for_model(model_instance):
            wrapped_model_combined = ShapWrapperCombinedInput(model_instance, sample_x_shape, sample_pos_shape)

            if use_gradient_explainer:
                explainer = shap.GradientExplainer(wrapped_model_combined, combined_input_for_shap_data)
            else:
                explainer = shap.DeepExplainer(wrapped_model_combined, combined_input_for_shap_data)

            raw_shap_values = explainer.shap_values(combined_input_for_shap_data)

            if isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3 and raw_shap_values.shape[2] == num_model_outputs:
                raw_shap_values = [raw_shap_values[:, :, i] for i in range(num_model_outputs)]
            elif not isinstance(raw_shap_values, list) and num_model_outputs > 1:
                total_flat_size_per_output = wrapped_model_combined.x_flat_size + wrapped_model_combined.pos_flat_size
                if raw_shap_values.shape[1] == num_model_outputs * total_flat_size_per_output:
                    split_shap_values = []
                    for i in range(num_model_outputs):
                        start_idx = i * total_flat_size_per_output
                        end_idx = (i + 1) * total_flat_size_per_output
                        split_shap_values.append(raw_shap_values[:, start_idx:end_idx])
                    raw_shap_values = split_shap_values
                else:
                    raw_shap_values = [raw_shap_values]
            elif not isinstance(raw_shap_values, list) and num_model_outputs == 1:
                raw_shap_values = [raw_shap_values]

            current_model_shap_x_outputs = []
            if isinstance(raw_shap_values, list):
                for output_shap_array in raw_shap_values:
                    shap_x_flat = output_shap_array[:, :wrapped_model_combined.x_flat_size]
                    shap_x_reshaped = shap_x_flat.reshape(output_shap_array.shape[0], *wrapped_model_combined.x_sample_shape)
                    current_model_shap_x_outputs.append(np.abs(shap_x_reshaped))
                return np.stack(current_model_shap_x_outputs, axis=0)
            else:
                shap_x_flat = raw_shap_values[:, :wrapped_model_combined.x_flat_size]
                shap_x_reshaped = shap_x_flat.reshape(raw_shap_values.shape[0], *wrapped_model_combined.x_sample_shape)
                return np.abs(shap_x_reshaped)

        shap_results = None

        should_compute_shap = (ifheatmap == 'yes' and testx.ndim > 3) or (ifweight in ['yes', 'shap'])

        if should_compute_shap:
            def _zero_shap_results_for_current_model():
                return np.zeros((num_model_outputs, current_testx_data.shape[0], *sample_x_shape), dtype=np.float32)
            try:
                if k_fold != None:
                    all_models_shap_x_results = []
                    for model_k_fold in models:
                        model_k_fold.eval()
                        if disable_cudnn_for_rnn_tcn:
                            with torch.backends.cudnn.flags(enabled=False):
                                all_models_shap_x_results.append(_compute_shap_values_for_model(model_k_fold))
                        else:
                            all_models_shap_x_results.append(_compute_shap_values_for_model(model_k_fold))

                    shap_results = np.nanmean(np.stack(all_models_shap_x_results, axis=0), axis=0)
                else:
                    model.eval()
                    if disable_cudnn_for_rnn_tcn:
                        with torch.backends.cudnn.flags(enabled=False):
                            shap_results = _compute_shap_values_for_model(model)
                    else:
                        shap_results = _compute_shap_values_for_model(model)
            except Exception as shap_error:
                if ifmute == 'no':
                    print('SHAP 计算失败，已返回零 heatmap/weights。错误:', repr(shap_error))
                shap_results = _zero_shap_results_for_current_model()

        if ifheatmap == 'yes' and testx.ndim > 3:
            heatmap_more = shap_results

            if timestep is not None:
                heatmap = np.nanmean(heatmap_more, axis=(1, 2))
            else:
                heatmap = np.nanmean(heatmap_more, axis=1)

        if ifweight == 'yes' or ifweight == 'shap':
            weights_more = shap_results

            if timestep is not None:
                feature_axis_in_sample_x_shape = 1
            elif vx_original_ndim == 3 and len(sample_x_shape) == 2:
                feature_axis_in_sample_x_shape = 0
            else:
                feature_axis_in_sample_x_shape = 0

            feature_axis_in_weights_more = 2 + feature_axis_in_sample_x_shape

            axes_to_average = [1]

            for ax_idx in range(len(sample_x_shape)):
                current_axis_in_weights_more = 2 + ax_idx
                if current_axis_in_weights_more != feature_axis_in_weights_more:
                    axes_to_average.append(current_axis_in_weights_more)

            weights = np.nanmean(weights_more, axis=tuple(axes_to_average))

            if weights.ndim == 1:
                weights = weights.reshape(1, -1)

            if timestep != None:
                input_channel_for_weights = trainx.shape[2]
            elif vx_original_ndim == 3 and trainx.ndim == 3:
                input_channel_for_weights = trainx.shape[1]
            else:
                input_channel_for_weights = trainx.shape[1]

            if weights.shape[0] == num_model_outputs and weights.shape[1] == input_channel_for_weights:
                weight_for_loop = weights.transpose(1, 0)
            elif weights.shape[0] == 1 and weights.shape[1] == input_channel_for_weights:
                weight_for_loop = weights.reshape(-1, 1)
            else:
                weight_for_loop = np.zeros((input_channel_for_weights, num_model_outputs))

            weights_final_output = np.zeros((num_model_outputs, input_channel_for_weights))

            for i in range(num_model_outputs):
                for j in range(input_channel_for_weights):
                    if weight_for_loop.shape[0] > j and weight_for_loop.shape[1] > i:
                        sum_for_norm = np.nansum(weight_for_loop[:, i])
                        if sum_for_norm != 0:
                            contribution_val = (weight_for_loop[j, i] / sum_for_norm) * 100
                        else:
                            contribution_val = 0.0
                        weights_final_output[i, j] = contribution_val
                        print('预报因子', j+1, '对预报值', i+1, '的贡献：', np.array(weights_final_output[i, j]), '％')
                    else:
                        pass
                print('\n')

            weights = weights_final_output


        elif ifweight=='oob':
            if task_mode=='multi_classify':
                oob_output_number = int(np.max(vy)) + 1
                weights=np.zeros((oob_output_number,input_channel))
                weight_more=np.zeros((oob_output_number,input_channel))
                testy_oob = testy
                predicty_oob = predicty
            else:
                testy_oob = testy.reshape(testy.shape[0], -1)
                predicty_oob = predicty.reshape(predicty.shape[0], -1)
                oob_output_number = testy_oob.shape[1]
                weights=np.zeros((oob_output_number,input_channel))
                weight_more=np.zeros((oob_output_number,input_channel))

            for i in tqdm(range(oob_output_number)):
                for j in range(input_channel):
                    testx_new=copy.deepcopy(testx)
                    weight=[]
                    for k in range(10):
                        per=np.random.permutation(testx.shape[0])
                        if timestep is not None:
                            testx_new[:, :, j, ...] = testx[per, :, j, ...]
                        elif vx_original_ndim == 3 and testx_new.ndim == 3:
                            testx_new[:, j, :] = testx[per, j, :]
                        elif testx_new.ndim == 2:
                            testx_new[:, j] = testx[per, j]
                        else:
                            testx_new[:, j, ...] = testx[per, j, ...]

                        if k_fold!=None:
                            if task_mode=='multi_classify':
                                predicty_new=np.zeros((k_fold,testy.shape[0],int(np.max(vy))+1))
                            else:
                                predicty_new=np.zeros((k_fold,*predicty.shape), dtype=np.float32)
                            for n in range(k_fold):
                                for l in range(int(np.ceil(testx.shape[0]/batch_size))):
                                    if l == int(testx.shape[0]/batch_size):
                                        predicty_new_batch = models[n](torch.tensor(testx_new[l*batch_size:],dtype=torch.float32,device=devices),torch.tensor(test_position[l*batch_size:],dtype=torch.int64,device=devices))
                                        predicty_new[n,l*batch_size:]=predicty_new_batch.cpu().detach().numpy()
                                    else:
                                        predicty_new_batch = models[n](torch.tensor(testx_new[l*batch_size:(l+1)*batch_size],dtype=torch.float32,device=devices),torch.tensor(test_position[l*batch_size:(l+1)*batch_size],dtype=torch.int64,device=devices))
                                        predicty_new[n,l*batch_size:(l+1)*batch_size]=predicty_new_batch.cpu().detach().numpy()
                            predicty_new=np.nanmean(predicty_new,axis=0)
                        else:
                            if task_mode=='multi_classify':
                                predicty_new=np.zeros((testy.shape[0],int(np.max(vy))+1))
                            else:
                                predicty_new=np.zeros_like(predicty,dtype=np.float32)
                            for l in range(int(np.ceil(testx.shape[0]/batch_size))):
                                if l == int(testx.shape[0]/batch_size) :
                                    predicty_new_batch = model(torch.tensor(testx_new[l*batch_size:],dtype=torch.float32,device=devices),torch.tensor(test_position[l*batch_size:],dtype=torch.int64,device=devices))
                                    predicty_new[l*batch_size:]=predicty_new_batch.cpu().detach().numpy()
                                else:
                                    predicty_new_batch = model(torch.tensor(testx_new[l*batch_size:(l+1)*batch_size],dtype=torch.float32,device=devices),torch.tensor(test_position[l*batch_size:(l+1)*batch_size],dtype=torch.int64,device=devices))
                                    predicty_new[l*batch_size:(l+1)*batch_size]=predicty_new_batch.cpu().detach().numpy()

                        if task_mode=='regression':
                            predicty_new_oob = predicty_new.reshape(predicty_new.shape[0], -1)
                            weight.append(
                                sklearn.metrics.mean_squared_error(testy_oob[:,i],predicty_new_oob[:,i])
                                - sklearn.metrics.mean_squared_error(testy_oob[:,i],predicty_oob[:,i])
                            )
                        elif task_mode=='multi_classify':
                            weight.append(sklearn.metrics.log_loss(testy[:,i],predicty_new[:,:])-sklearn.metrics.log_loss(testy[:,i],predicty[:,:]))
                        else:
                            predicty_new_oob = predicty_new.reshape(predicty_new.shape[0], -1)
                            weight.append(
                                sklearn.metrics.log_loss(testy_oob[:,i],predicty_new_oob[:,i])
                                - sklearn.metrics.log_loss(testy_oob[:,i],predicty_oob[:,i])
                            )
                    weight_more[i,j]=np.nanmean(weight)
            for i in range(oob_output_number):
                for j in range(input_channel):
                    denom = np.nansum(weight_more[i,:])
                    weights[i,j]=(weight_more[i,j]/denom)*100 if denom != 0 else 0.0
                    print('预报因子',j+1,'对预报值',i+1,'的贡献：',np.array(weights[i,j]),'％')
                print('\n')
    if ifsave=='yes':
        if k_fold!=None:
            for i in range(k_fold):
                torch.save(models[i].state_dict(),savepath+'_'+str(i+1)+'.pth')
                torch.save(opts[i].state_dict(),savepath+'_'+str(i+1)+'_opt.pth')
        else:
            torch.save(model.state_dict(),savepath+'.pth')
            torch.save(opt.state_dict(),savepath+'_opt.pth')
    if k_fold!=None:
        return models,predicty_return,testy_return,r,p,heatmap,weights
    else:
        return model,predicty_return,testy_return,r,p,heatmap,weights
