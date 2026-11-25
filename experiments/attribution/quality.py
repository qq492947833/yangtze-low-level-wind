"""Numerical criteria of the stagewise expected-gradients experiment."""
import numpy as np
from expected_gradients import gpu_expected_gradients
from stagewise_wrapper import aggregate_features


def convergence(previous,current,share_tolerance=.05,l1_tolerance=.2,tie=.02):
    old=np.asarray(previous,dtype=np.float64);new=np.asarray(current,dtype=np.float64)
    if old.shape!=new.shape or old.ndim!=5 or old.shape[1]!=6 or old.shape[2] not in (3,12):raise ValueError('same full stage element arrays required')
    if not np.isfinite(old).all() or not np.isfinite(new).all():raise ValueError('nonfinite attribution')
    flat=lambda z:z.reshape(len(z),-1)
    l1=np.abs(flat(new-old)).sum(1)/np.maximum(np.abs(flat(new)).sum(1),1e-12)
    f0=aggregate_features(old);f1=aggregate_features(new)
    sh0=np.abs(f0)/np.maximum(np.abs(f0).sum(1,keepdims=True),1e-12);sh1=np.abs(f1)/np.maximum(np.abs(f1).sum(1,keepdims=True),1e-12)
    share=np.abs(sh1-sh0).sum(1);C=new.shape[2]
    order=np.asarray([all(abs(a[i]-a[j])<=tie or (a[i]-a[j])*(b[i]-b[j])>=0 for i in range(C) for j in range(i)) for a,b in zip(sh0,sh1)])
    return {'convergence_l1':l1,'convergence_feature_share_l1':share,'convergence_robust_order':order,'signed_feature_convergence_l1':np.abs(f1-f0).sum(1)/np.maximum(np.abs(f1).sum(1),1e-12),'convergence_pass':(l1<=l1_tolerance)&(share<=share_tolerance)&order}


def integrate_outputs(model, x, backgrounds, prediction, expected, scale, stage, lead, sample, microbatch, maximum):
    """Retain the first prefix passing both completeness and full-element convergence for each output."""
    remaining = np.arange(9)
    fields = {}
    previous = None
    resume_acc = None
    offset = 0
    seed = 25 + 7919 * sample
    extended = {(1, 2, 95, 5), (1, 5, 92, 5), (1, 6, 97, 2)}
    for n in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
        if n > maximum:
            break
        if n == 262144 and any((stage, lead, sample, int(c)) not in extended for c in remaining):
            raise RuntimeError(f"Regular integration budget exhausted for outputs {remaining}")
        def observe(count, phi, acc):
            nonlocal previous, resume_acc
            if count == n // 2:
                previous = phi.copy()
            if count == n:
                resume_acc = acc.copy()
        elements, _ = gpu_expected_gradients(
            model, x, backgrounds, [(int(c), 0, 0) for c in remaining],
            seed, n, microbatch=microbatch, prefix_observer=observe,
            resume_acc=resume_acc, resume_offset=offset)
        if previous is None:
            raise RuntimeError("Missing half-length prefix")
        cv = convergence(previous, elements)
        features = aggregate_features(elements)
        total = elements.reshape(len(remaining), -1).sum(1)
        error = total - (prediction[remaining] - expected[remaining])
        limit = .02 * scale[remaining] + .05 * np.maximum(
            np.abs(prediction[remaining] - expected[remaining]), .1 * scale[remaining])
        passing = (np.abs(error) <= limit) & cv["convergence_pass"]
        for j in np.flatnonzero(passing):
            c = int(remaining[j])
            fields[c] = dict(phi_feature=features[j], prediction=prediction[c], expected_value=expected[c],
                             attribution_sum=total[j], efficiency_error=error[j], efficiency_tolerance=limit[j],
                             nsamples=n, previous_nsamples=n // 2, quality_pass=passing[j],
                             **{key: val[j] for key, val in cv.items()})
        if passing.all():
            return {k: np.asarray([fields[c][k] for c in range(9)]) for k in fields[0]}
        remaining = remaining[~passing]
        previous = elements[~passing].copy()
        resume_acc = resume_acc[~passing].copy()
        offset = n
    raise RuntimeError(f"Attribution did not converge for stage {stage}, lead {lead}, sample {sample}, outputs {remaining}")
