"""Float64, per-point multi-output ridge regression from streaming moments.

No file I/O, GPU access, project data access, or implicit missing-value filtering.
Each point and lead is fitted using training data alone. The exact objective is

    mean_n(sum_q((Ystd - Xstd @ W - b)**2)) + alpha * sum(W**2).

Thus (Xstd.T @ Xstd / n + alpha I) W = Xstd.T @ Ystd / n.
The intercept b is unpenalized. X/Y population means and standard deviations
(ddof=0) come from training only; constant columns use scale 1. All outputs
are retained, including constant outputs. Equal point/output weighting is
used ONLY for global validation alpha selection, not inside the ridge penalty.

Statistics retain raw sum/XX/XY/YY for audit and stable centered second moments
for all calculations. Restoring only raw moments is deliberately unsupported:
large physical offsets can make raw second-moment centering inaccurate.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


DEFAULT_ALPHAS = np.power(10.0, np.arange(-6, 4, dtype=np.float64))


class NumericalError(ValueError):
    """Numerical conditioning or a mathematical consistency check failed."""


def _matrix(value, columns, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape [n,{columns}], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values; caller must define valid samples")
    return array


def _alphas(value):
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("alphas must be a nonempty finite positive vector")
    if np.any(np.diff(array) <= 0):
        raise ValueError("alphas must be strictly increasing, enabling deterministic smallest-alpha ties")
    return array


@dataclass
class SufficientStatistics:
    n: int
    sum_x: np.ndarray
    sum_y: np.ndarray
    xx: np.ndarray
    xy: np.ndarray
    yy: np.ndarray
    mean_x: np.ndarray
    mean_y: np.ndarray
    m2_xx: np.ndarray
    m2_xy: np.ndarray
    m2_yy: np.ndarray
    min_x: np.ndarray
    max_x: np.ndarray
    min_y: np.ndarray
    max_y: np.ndarray

    @classmethod
    def empty(cls, p, q):
        if not isinstance(p, (int, np.integer)) or not isinstance(q, (int, np.integer)) or p <= 0 or q <= 0:
            raise ValueError("p and q must be positive integers")
        return cls(0, np.zeros(p), np.zeros(q), np.zeros((p, p)), np.zeros((p, q)),
                   np.zeros((q, q)), np.zeros(p), np.zeros(q), np.zeros((p, p)),
                   np.zeros((p, q)), np.zeros((q, q)), np.full(p, np.inf),
                   np.full(p, -np.inf), np.full(q, np.inf), np.full(q, -np.inf))

    @property
    def p(self):
        return self.sum_x.size

    @property
    def q(self):
        return self.sum_y.size

    def update(self, x, y):
        """Accumulate a finite raw batch; empty batches are a no-op.

        A single centered Gram calculation supplies both raw and stable moments.
        All numerical checks occur before mutating this accumulator.
        """
        x, y = _matrix(x, self.p, "X"), _matrix(y, self.q, "Y")
        if len(x) != len(y):
            raise ValueError("X and Y sample counts differ")
        if not len(x):
            return self
        batch = self.empty(self.p, self.q)
        batch.n = len(x)
        batch.mean_x, batch.mean_y = x.mean(axis=0), y.mean(axis=0)
        cx, cy = x - batch.mean_x, y - batch.mean_y
        batch.m2_xx, batch.m2_xy, batch.m2_yy = cx.T @ cx, cx.T @ cy, cy.T @ cy
        batch.sum_x, batch.sum_y = x.sum(axis=0), y.sum(axis=0)
        batch.xx = batch.m2_xx + batch.n * np.outer(batch.mean_x, batch.mean_x)
        batch.xy = batch.m2_xy + batch.n * np.outer(batch.mean_x, batch.mean_y)
        batch.yy = batch.m2_yy + batch.n * np.outer(batch.mean_y, batch.mean_y)
        batch.min_x, batch.max_x = x.min(axis=0), x.max(axis=0)
        batch.min_y, batch.max_y = y.min(axis=0), y.max(axis=0)
        batch._validate()
        return self.merge(batch)

    def _validate(self):
        if not isinstance(self.n, (int, np.integer)) or self.n < 0:
            raise ValueError("invalid statistics sample count")
        shapes = {"sum_x": (self.p,), "sum_y": (self.q,), "mean_x": (self.p,), "mean_y": (self.q,),
                  "xx": (self.p, self.p), "xy": (self.p, self.q), "yy": (self.q, self.q),
                  "m2_xx": (self.p, self.p), "m2_xy": (self.p, self.q), "m2_yy": (self.q, self.q),
                  "min_x": (self.p,), "max_x": (self.p,), "min_y": (self.q,), "max_y": (self.q,)}
        for name, shape in shapes.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != np.float64:
                raise ValueError(f"{name} must be float64 with shape {shape}")
            if self.n and not np.all(np.isfinite(value)):
                raise NumericalError(f"{name} contains nonfinite sufficient statistics")

    def merge(self, other):
        """Merge independent accumulators with the Chan centered-moment formula."""
        if (self.p, self.q) != (other.p, other.q):
            raise ValueError("cannot merge statistics with different feature/output counts")
        self._validate()
        other._validate()
        if not other.n:
            return self
        if not self.n:
            for name in self.__dataclass_fields__:
                value = getattr(other, name)
                setattr(self, name, value.copy() if isinstance(value, np.ndarray) else int(value))
            return self
        count = self.n + other.n
        weight = float(self.n) * (float(other.n) / float(count))
        dx, dy = other.mean_x - self.mean_x, other.mean_y - self.mean_y
        updates = {
            "mean_x": self.mean_x + dx * (other.n / count),
            "mean_y": self.mean_y + dy * (other.n / count),
            "m2_xx": self.m2_xx + other.m2_xx + weight * np.outer(dx, dx),
            "m2_xy": self.m2_xy + other.m2_xy + weight * np.outer(dx, dy),
            "m2_yy": self.m2_yy + other.m2_yy + weight * np.outer(dy, dy),
            "sum_x": self.sum_x + other.sum_x, "sum_y": self.sum_y + other.sum_y,
            "xx": self.xx + other.xx, "xy": self.xy + other.xy, "yy": self.yy + other.yy,
            "min_x": np.minimum(self.min_x, other.min_x), "max_x": np.maximum(self.max_x, other.max_x),
            "min_y": np.minimum(self.min_y, other.min_y), "max_y": np.maximum(self.max_y, other.max_y),
        }
        if any(not np.all(np.isfinite(value)) for value in updates.values()):
            raise NumericalError("merging sufficient statistics overflowed")
        for name, value in updates.items():
            setattr(self, name, value)
        self.n = count
        return self


@dataclass
class RidgePath:
    alphas: np.ndarray
    coef_std: np.ndarray  # [alpha, feature, output]
    intercept_std: np.ndarray  # [alpha, output], unpenalized and zero after train centering
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    constant_x: np.ndarray
    constant_y: np.ndarray
    diagnostics: dict


def fit_ridge_path(stats, alphas=DEFAULT_ALPHAS, *, max_condition=1e12, residual_tolerance=1e-10):
    """Fit every candidate alpha with one symmetric eigendecomposition.

    Any nonfinite result, materially indefinite covariance, excessive condition
    number, or failed backward residual raises NumericalError. No pseudo-inverse
    fallback or silently accepted failure is permitted. Small negative eigenvalues
    attributable to roundoff are clipped and explicitly counted in diagnostics.
    """
    stats._validate()
    if stats.n < 2:
        raise ValueError("at least two training samples are required")
    values = _alphas(alphas)
    if not np.isfinite(max_condition) or max_condition <= 1 or not np.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError("invalid numerical acceptance thresholds")
    constant_x, constant_y = stats.min_x == stats.max_x, stats.min_y == stats.max_y
    vx, vy = np.diag(stats.m2_xx).copy() / stats.n, np.diag(stats.m2_yy).copy() / stats.n
    if np.any(vx < 0) or np.any(vy < 0):
        raise NumericalError("negative training variance")
    vx[constant_x], vy[constant_y] = 0.0, 0.0
    if np.any((vx == 0) & ~constant_x) or np.any((vy == 0) & ~constant_y):
        raise NumericalError("nonconstant columns have numerically zero variance")
    sx, sy = np.sqrt(vx), np.sqrt(vy)
    sx[constant_x], sy[constant_y] = 1.0, 1.0
    covariance = stats.m2_xx / stats.n / sx[:, None] / sx[None, :]
    cross = stats.m2_xy / stats.n / sx[:, None] / sy[None, :]
    covariance[constant_x, :], covariance[:, constant_x] = 0.0, 0.0
    cross[constant_x, :], cross[:, constant_y] = 0.0, 0.0
    covariance = (covariance + covariance.T) * 0.5
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(cross)):
        raise NumericalError("standardized moments are nonfinite")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    negative_tolerance = 64 * np.finfo(np.float64).eps * max(stats.p, 1) * spectral_scale
    if eigenvalues[0] < -negative_tolerance:
        raise NumericalError(f"training covariance is indefinite: min eigenvalue {eigenvalues[0]}")
    clipped = int(np.count_nonzero(eigenvalues < 0))
    effective = np.maximum(eigenvalues, 0.0)
    transformed = eigenvectors.T @ cross
    coefficients, alpha_diagnostics = [], []
    for alpha in values:
        condition = float((effective[-1] + alpha) / (effective[0] + alpha))
        if condition > max_condition:
            raise NumericalError(f"alpha {alpha:g} condition {condition:g} exceeds {max_condition:g}")
        coefficient = eigenvectors @ (transformed / (effective[:, None] + alpha))
        coefficient[constant_x, :], coefficient[:, constant_y] = 0.0, 0.0
        residual = covariance @ coefficient + alpha * coefficient - cross
        denominator = (spectral_scale + alpha) * np.linalg.norm(coefficient) + np.linalg.norm(cross)
        relative_residual = float(np.linalg.norm(residual) / max(float(denominator), np.finfo(np.float64).tiny))
        if not np.all(np.isfinite(coefficient)) or not np.isfinite(relative_residual) or relative_residual > residual_tolerance:
            raise NumericalError(f"alpha {alpha:g} failed solve residual: {relative_residual:g}")
        coefficients.append(coefficient)
        alpha_diagnostics.append({"alpha": float(alpha), "condition_number": condition,
                                  "relative_backward_residual": relative_residual,
                                  "condition_times_float64_epsilon": condition * np.finfo(np.float64).eps})
    diagnostics = {"solver": "symmetric_eigh_shared_over_alpha", "dtype": "float64", "n_train": stats.n,
                   "n_features": stats.p, "n_outputs": stats.q, "standardization_ddof": 0,
                   "objective": "mean_n(sum_q(residual_std**2)) + alpha*sum(W**2); intercept unpenalized",
                   "constant_feature_indices": np.flatnonzero(constant_x).tolist(),
                   "constant_output_indices": np.flatnonzero(constant_y).tolist(),
                   "constant_policy": "scale=1; constant feature coefficients and constant output coefficients=0",
                   "min_unclipped_covariance_eigenvalue": float(eigenvalues[0]),
                   "max_covariance_eigenvalue": float(eigenvalues[-1]),
                   "roundoff_negative_eigenvalue_count": clipped, "negative_eigenvalue_tolerance": negative_tolerance,
                   "max_allowed_condition": float(max_condition), "residual_tolerance": float(residual_tolerance),
                   "stable_centered_statistics": True, "alpha_diagnostics": alpha_diagnostics}
    return RidgePath(values, np.stack(coefficients), np.zeros((len(values), stats.q)),
                     stats.mean_x.copy(), sx, stats.mean_y.copy(), sy, constant_x, constant_y, diagnostics)


def predict(path, x, alpha_index, *, standardized=False):
    """Predict using train-fitted scales; return float64 in raw Y units by default."""
    if not isinstance(alpha_index, (int, np.integer)) or not 0 <= alpha_index < len(path.alphas):
        raise ValueError("alpha_index is outside the fitted path")
    x = _matrix(x, len(path.x_mean), "X")
    result = ((x - path.x_mean) / path.x_scale) @ path.coef_std[alpha_index] + path.intercept_std[alpha_index]
    if not standardized:
        result = result * path.y_scale + path.y_mean
    if not np.all(np.isfinite(result)):
        raise NumericalError("prediction is nonfinite")
    return result


def validation_mse(path, stats, *, return_diagnostics=False):
    """Per-alpha/per-output standardized MSE from stable validation moments.

    Validation means are NOT used to fit or change model standardization. The
    centered covariance identity plus the squared residual mean includes every
    intercept/cross term. Tiny negative roundoff is clipped with a reported bound;
    materially negative values fail. Samples share one finite X/Y support.
    """
    stats._validate()
    if not stats.n or (stats.p, stats.q) != (len(path.x_mean), len(path.y_mean)):
        raise ValueError("validation statistics are empty or have incompatible dimensions")
    cxx = stats.m2_xx / stats.n / path.x_scale[:, None] / path.x_scale[None, :]
    cxy = stats.m2_xy / stats.n / path.x_scale[:, None] / path.y_scale[None, :]
    cyy = np.diag(stats.m2_yy) / stats.n / path.y_scale ** 2
    dx, dy = (stats.mean_x - path.x_mean) / path.x_scale, (stats.mean_y - path.y_mean) / path.y_scale
    w = path.coef_std
    quadratic = np.einsum("apq,pr,arq->aq", w, cxx, w, optimize=True)
    cross = 2 * np.einsum("apq,pq->aq", w, cxy, optimize=True)
    residual_mean = dy[None, :] - np.einsum("p,apq->aq", dx, w) - path.intercept_std
    mean_square = residual_mean ** 2
    result = cyy[None, :] + quadratic - cross + mean_square
    tolerance = 128 * np.finfo(np.float64).eps * np.maximum(
        np.abs(cyy)[None, :] + np.abs(quadratic) + np.abs(cross) + mean_square, 1.0)
    if not np.all(np.isfinite(result)) or np.any(result < -tolerance):
        raise NumericalError("validation MSE is nonfinite or materially negative")
    negatives = result < 0
    result = np.maximum(result, 0.0)
    diagnostics = {"n_validation": stats.n, "negative_roundoff_clipped_count": int(np.count_nonzero(negatives)),
                   "max_absolute_roundoff_bound": float(np.max(tolerance)),
                   "formula": "centered validation residual variance + squared residual mean",
                   "train_scaling_only": True}
    return (result, diagnostics) if return_diagnostics else result


def select_global_alpha(point_mse, alphas=DEFAULT_ALPHAS):
    """Select one lead-global alpha with equal weights across valid points/outputs.

    Input [point, alpha, output]. A wholly NaN point is an explicit excluded
    point; partial missing scores, infinities, negative scores, and no valid
    points fail. Every alpha is compared on the exact same point/output support.
    Ties choose the first (smallest) alpha. Point sample counts are not weights.
    """
    values = _alphas(alphas)
    scores = np.asarray(point_mse, dtype=np.float64)
    if scores.ndim != 3 or scores.shape[1] != len(values) or not scores.shape[0] or not scores.shape[2]:
        raise ValueError("point_mse must have nonempty shape [point,alpha,output]")
    excluded = np.isnan(scores).all(axis=(1, 2))
    selected = scores[~excluded]
    if not len(selected) or not np.all(np.isfinite(selected)) or np.any(selected < 0):
        raise ValueError("valid points need finite nonnegative scores for every alpha/output")
    means = selected.mean(axis=(0, 2))
    index = int(np.argmin(means))
    return {"alpha_index": index, "alpha": float(values[index]), "mean_validation_mse": means.tolist(),
            "valid_point_count": int(np.count_nonzero(~excluded)), "n_outputs": scores.shape[2],
            "excluded_point_indices": np.flatnonzero(excluded).tolist(),
            "weighting": "equal valid points and equal outputs; fixed support across alphas",
            "tie_policy": "smallest alpha", "training_refit_with_validation": False}
