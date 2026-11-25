"""Float64 QR/SVD core for the isolated alpha-boundary diagnostic.

No I/O, project imports, normalization fitting, masking, or thread control.
Inputs have already been standardized with the original saved training scales.
Compressed constant/X/Y columns MUST share the same QR coordinate system.
The caller records the actual (uncompressed) sample count and provenance.
"""
from __future__ import annotations
import numpy as np

EPS = np.finfo(np.float64).eps
BACKWARD_RESIDUAL_LIMIT = 1e-10


def _matrix(value, label):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(f'{label} must be a nonempty matrix')
    if not np.isfinite(result).all():
        raise ValueError(f'{label} contains nonfinite values')
    return result


def _count(n, rows):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < rows:
        raise ValueError('n must be the integer original sample count, at least compressed rows')
    return int(n)


def compress_rows(array):
    """Return R with A.T@A == R.T@R to QR roundoff; no explicit Gram is formed."""
    a = _matrix(array, 'array')
    r = np.linalg.qr(a, mode='r')
    # Canonical row signs help deterministic replay for an unchanged block order.
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    r *= signs[:, None]
    if not np.isfinite(r).all():
        raise ValueError('QR returned nonfinite values')
    return r


def merge_qr(r, newrows):
    """Merge existing R (or None) with raw rows or another compatible R."""
    a = _matrix(newrows, 'newrows')
    if r is None:
        return compress_rows(a)
    previous = _matrix(r, 'r')
    if previous.shape[1] != a.shape[1]:
        raise ValueError('QR merge column counts differ')
    return compress_rows(np.vstack((previous, a)))


def _constant(value, rows, n, p):
    if value is None:
        raise ValueError('compressed constant column is required')
    z = np.asarray(value, dtype=np.float64)
    if z.shape != (rows,) or not np.isfinite(z).all():
        raise ValueError('constant must be a finite vector in the shared QR coordinates')
    zz = float(z @ z)
    relative_bound = 512.0 * EPS * max(p + 1, int(np.ceil(np.log2(max(n, 2)))))
    if abs(zz - n) > relative_bound * n:
        raise ValueError('compressed constant norm does not match original sample count')
    return z, zz


def solve_small(x_compressed, y_compressed, n, alphas, *, constant=None, intercept='fit'):
    """Solve mean-sample squared loss + alpha*||W||_F**2.

    Positive alpha uses every singular value (no OLS cutoff). At alpha==0,
    minimum-norm OLS retains s > eps64*max(original_n,p)*s_max. This fixed,
    precision-derived policy never examines validation or test data.

    intercept='fit' projects the shared constant column and returns an
    unpenalized b. 'fixed_zero' keeps the original saved-model execution's
    exact zero standardized intercept; no recentering occurs in that mode.
    Neither mode estimates or modifies feature/target normalization scales.
    """
    x, y = _matrix(x_compressed, 'X'), _matrix(y_compressed, 'Y')
    if x.shape[0] != y.shape[0]:
        raise ValueError('X/Y compressed row counts differ')
    n = _count(n, len(x))
    p, q = x.shape[1], y.shape[1]
    values = np.asarray(alphas, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('alphas must be a nonempty finite nonnegative vector')
    if len(np.unique(values)) != len(values):
        raise ValueError('duplicate alpha candidates are not allowed')
    if intercept not in ('fit', 'fixed_zero'):
        raise ValueError('intercept must be fit or fixed_zero')
    if intercept == 'fit':
        z, zz = _constant(constant, len(x), n, p)
        mx, my = (z @ x) / zz, (z @ y) / zz
        xc, yc = x - z[:, None] * mx, y - z[:, None] * my
    else:
        if constant is not None:
            _constant(constant, len(x), n, p)
        mx, my = np.zeros(p), np.zeros(q)
        xc, yc = x, y
    xn, yn = xc / np.sqrt(float(n)), yc / np.sqrt(float(n))
    u, singular, vt = np.linalg.svd(xn, full_matrices=False)
    smax = float(singular[0])
    relative_cutoff = EPS * max(n, p)
    cutoff = relative_cutoff * smax
    retained = singular > cutoff
    projected_y = u.T @ yn
    coef, offsets, checks = [], [], []
    cross_norm = float(np.linalg.norm(xn.T @ yn))
    for alpha in values:
        if alpha == 0.0:
            filt = np.zeros_like(singular)
            filt[retained] = 1.0 / singular[retained]
        else:
            filt = singular / (singular * singular + alpha)
        w = vt.T @ (filt[:, None] * projected_y)
        b = my - mx @ w
        residual = xn @ w - yn
        gradient = xn.T @ residual + alpha * w
        denominator = (smax*smax + alpha) * np.linalg.norm(w) + cross_norm
        backward = float(np.linalg.norm(gradient) / max(float(denominator), np.finfo(float).tiny))
        if not np.isfinite(w).all() or not np.isfinite(b).all() or not np.isfinite(backward):
            raise ValueError('SVD solution or numerical diagnostics are nonfinite')
        if backward > BACKWARD_RESIDUAL_LIMIT:
            raise ValueError(f'alpha {alpha:g} backward residual {backward:g} exceeds {BACKWARD_RESIDUAL_LIMIT:g}')
        coef.append(w)
        offsets.append(b)
        smin = float(singular[-1]) if len(singular) == p else 0.0
        checks.append({'alpha': float(alpha), 'normal_equation_relative_backward_residual': backward,
                       'ridge_normal_matrix_condition': float((smax*smax+alpha)/(smin*smin+alpha)) if alpha > 0 else None,
                       'ols_cutoff_applied': bool(alpha == 0.0)})
    diagnostics = {
        'solver': 'float64_qr_compression_then_svd', 'numpy_version': np.__version__,
        'n_original_samples': n, 'compressed_rows': len(x), 'n_features': p, 'n_outputs': q,
        'intercept_mode': intercept, 'original_training_standardization_preserved': True,
        'ols_rank': int(np.count_nonzero(retained)), 'ols_relative_cutoff': float(relative_cutoff),
        'ols_absolute_cutoff': float(cutoff), 'singular_values': singular.tolist(),
        'relative_backward_residual_limit': BACKWARD_RESIDUAL_LIMIT,
        'positive_alpha_rank_truncation': False, 'train_standardized_x_mean': mx.tolist(),
        'train_standardized_y_mean': my.tolist(), 'alpha_diagnostics': checks,
    }
    return {'coef': np.stack(coef), 'intercept': np.stack(offsets),
            'alphas': values.copy(), 'diagnostics': diagnostics}


def score_small(x_compressed, y_compressed, constant, coef, intercepts, n):
    """Per-alpha/output MSE from ||Yc-XcW-zb||**2/n; no moment subtraction.

    X/Y are normalized using the original TRAIN means/scales even if these
    compressed rows represent validation. The constant belongs to that split.
    """
    x, y = _matrix(x_compressed, 'X'), _matrix(y_compressed, 'Y')
    if x.shape[0] != y.shape[0]:
        raise ValueError('X/Y compressed row counts differ')
    n = _count(n, len(x))
    z, _ = _constant(constant, len(x), n, x.shape[1])
    w, b = np.asarray(coef, dtype=np.float64), np.asarray(intercepts, dtype=np.float64)
    if w.ndim != 3 or w.shape[1:] != (x.shape[1], y.shape[1]) or b.shape != (w.shape[0], y.shape[1]) or not len(w):
        raise ValueError('coefficient/intercept shape mismatch')
    if not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite coefficient/intercept')
    scores = []
    for wi, bi in zip(w, b):
        residual = y - x @ wi - z[:, None] * bi
        scores.append(np.sum(residual * residual, axis=0, dtype=np.float64) / n)
    result = np.stack(scores)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite residual MSE')
    return result
