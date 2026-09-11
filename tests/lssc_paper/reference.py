"""Fixed-dictionary SC -> pilot matching -> SSC research reference.

This is NOT complete LSSC: no image-adaptive/grouped dictionary learning yet.
See README.md for paper equations, explicit choices, and verification boundaries.
No author implementation code or dictionary assets are included here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import chi2


@dataclass
class Pursuit:
    support: np.ndarray
    coefficients: np.ndarray
    reconstruction: np.ndarray
    residual_squared: float
    residual_history: np.ndarray
    epsilon: float
    stop_reason: str


@dataclass
class Stage:
    patches: np.ndarray
    groups: list[np.ndarray]
    pursuits: list[Pursuit]


@dataclass
class Result:
    output: np.ndarray
    pilot: np.ndarray
    coverage: np.ndarray
    positions: np.ndarray
    labels: np.ndarray
    pilot_stage: Stage
    final_stage: Stage
    matching_threshold: float


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 2 or min(value.shape) < 1 or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f'{name} must be a nonempty real matrix')
    if np.iscomplexobj(value) or not np.isfinite(value).all():
        raise ValueError(f'{name} must be finite and real')
    return np.asarray(value, dtype=np.float64)


def validate_dictionary(dictionary: np.ndarray) -> np.ndarray:
    dictionary = _matrix(dictionary, 'dictionary')
    norms = np.linalg.norm(dictionary, axis=0)
    if not np.allclose(norms, 1, rtol=1e-8, atol=1e-10):
        raise ValueError('dictionary columns must have unit norm; no implicit normalization')
    return dictionary


def noise_budget(m: int, group_size: int, sigma: float, tau: float = .8) -> float:
    """Mairal Eq. (7): sigma**2 * chi2.ppf(tau, m * |S|).

    Keep the paper's stated m*|S| degrees of freedom after mean subtraction;
    do not silently substitute (m-1)*|S| or a hand-tuned lambda.
    """
    if not isinstance(m, (int, np.integer)) or not isinstance(group_size, (int, np.integer)):
        raise ValueError('dimensions must be integers')
    if m < 1 or group_size < 1 or not np.isfinite(sigma) or sigma < 0 or not 0 < tau < 1:
        raise ValueError('invalid noise-budget parameters')
    value = float(sigma**2 * chi2.ppf(tau, int(m) * int(group_size)))
    if not np.isfinite(value):
        raise ValueError('nonfinite noise budget')
    return value


def somp(dictionary: np.ndarray, signals: np.ndarray, epsilon: float) -> Pursuit:
    """Tropp S-OMP: sum of absolute correlations, shared support, LS refit.

    Stop on squared Frobenius residual <= epsilon, never a fixed atom budget.
    A rank-limited dictionary may fail this constraint; expose that failure.
    Float64 SVD least squares is an intentionally transparent reference solve.
    """
    dictionary = validate_dictionary(dictionary)
    signals = _matrix(signals, 'signals')
    if dictionary.shape[0] != signals.shape[0] or not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError('invalid signals or epsilon')
    m, atoms = dictionary.shape
    support: list[int] = []
    excluded = np.zeros(atoms, dtype=bool)
    residual = signals.copy()
    reconstruction = np.zeros_like(signals)
    coefficients = np.empty((0, signals.shape[1]), dtype=np.float64)
    energy = float(np.sum(residual * residual))
    history = [energy]
    # Numerical feasibility tolerance, not an extra denoising/noise allowance.
    tolerance = 64 * np.finfo(np.float64).eps * max(1., energy)
    reason = 'span_exhausted'
    while True:
        if energy <= epsilon + tolerance:
            reason = 'noise_budget'
            break
        if len(support) >= min(m, atoms) or excluded.all():
            break
        correlations = dictionary.T @ residual
        scores = np.sum(np.abs(correlations), axis=1)
        scores[excluded] = -np.inf
        atom = int(np.argmax(scores))  # Stable first-index tie rule.
        if scores[atom] <= np.finfo(np.float64).eps * max(1., np.linalg.norm(signals)):
            break
        excluded[atom] = True
        trial = support + [atom]
        active = dictionary[:, trial]
        fit, _, rank, _ = np.linalg.lstsq(active, signals, rcond=None)
        if rank != len(trial):
            continue  # Never divide by a tiny Cholesky pivot or hide rank failure.
        reconstructed = active @ fit
        next_residual = signals - reconstructed
        next_energy = float(np.sum(next_residual * next_residual))
        if not np.isfinite(next_energy) or next_energy > energy + tolerance:
            raise ArithmeticError('least-squares residual did not decrease')
        support = trial
        coefficients = fit
        reconstruction = reconstructed
        residual = next_residual
        energy = next_energy
        history.append(energy)
    return Pursuit(np.asarray(support, dtype=np.int64), coefficients, reconstruction,
                   energy, np.asarray(history), float(epsilon), reason)


def extract_patches(image: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense VALID grid. Rows are patches; pixels within a patch use MATLAB/F order."""
    image = _matrix(image, 'image')
    if not isinstance(block, (int, np.integer)) or block < 1 or block > min(image.shape):
        raise ValueError('block must be a positive integer fitting the image')
    height, width = image.shape
    ny, nx = height - block + 1, width - block + 1
    windows = np.lib.stride_tricks.sliding_window_view(image, (block, block))
    # Swap patch y/x only, not the raster order of patch locations.
    patches = windows.transpose(0, 1, 3, 2).reshape(ny * nx, block * block).copy()
    ys, xs = np.indices((ny, nx))
    positions = np.column_stack((ys.ravel(), xs.ravel())).astype(np.int64)
    return patches, positions


def aggregate(patches: np.ndarray, positions: np.ndarray, shape: tuple[int, int],
              block: int) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (9) counting aggregation: no window, sharpen, clipping, or source blend."""
    patches = _matrix(patches, 'patches')
    positions = np.asarray(positions)
    if (positions.shape != (len(patches), 2) or patches.shape[1] != block * block
            or not np.issubdtype(positions.dtype, np.integer)):
        raise ValueError('invalid patch geometry')
    if len(shape) != 2 or min(shape) < block or block < 1:
        raise ValueError('invalid image geometry')
    ys, xs = positions.T
    if np.any(ys < 0) or np.any(xs < 0) or np.any(ys + block > shape[0]) or np.any(xs + block > shape[1]):
        raise ValueError('patch outside image')
    numerator = np.zeros(shape, dtype=np.float64)
    coverage = np.zeros(shape, dtype=np.int64)
    for dx in range(block):
        for dy in range(block):
            np.add.at(numerator, (ys + dy, xs + dx), patches[:, dy + block * dx])
            np.add.at(coverage, (ys + dy, xs + dx), 1)
    if np.any(coverage == 0):
        raise ValueError('uncovered pixels; no silent noisy-source fallback')
    return numerator / coverage, coverage


def similarity_threshold(sigma: float, patch_area: int) -> float:
    """Literal Eq. (6) + Sec. 4.1: raw SSD <= (32*sigma)**2 / m.

    Author-wrapper/internal SSD normalization has NOT been established.
    Do not tune this value against clean ground truth or silently drop /m.
    """
    if not np.isfinite(sigma) or sigma < 0 or patch_area < 1:
        raise ValueError('invalid matching parameters')
    return float((32 * sigma)**2 / patch_area)


def greedy_groups(pilot_patches: np.ndarray, grid_shape: tuple[int, int], window: int,
                  threshold: float) -> tuple[list[np.ndarray], np.ndarray]:
    """One explicit semi-local, disjoint-cover realization of Sec. 3.3.

    Raster first-unassigned seed; include unassigned patches inside a window
    whose raw SSD to the seed is <= threshold. No fixed cluster/group count.
    Members share a support; pairwise member distances need not be <= threshold.
    This exact grouping schedule is NOT specified by the paper or verified MEX.
    """
    pilot_patches = _matrix(pilot_patches, 'pilot_patches')
    if (not isinstance(window, (int, np.integer)) or window < 1 or len(grid_shape) != 2
            or min(grid_shape) < 1 or int(np.prod(grid_shape)) != len(pilot_patches)
            or not np.isfinite(threshold) or threshold < 0):
        raise ValueError('invalid grouping parameters')
    ny, nx = grid_shape
    labels = np.full(len(pilot_patches), -1, dtype=np.int64)
    groups = []
    left, right = window // 2, (window - 1) // 2
    grid = np.arange(len(labels)).reshape(ny, nx)
    for seed in range(len(labels)):
        if labels[seed] >= 0:
            continue
        y, x = divmod(seed, nx)
        candidates = grid[max(0, y-left):min(ny, y+right+1),
                          max(0, x-left):min(nx, x+right+1)].ravel()
        candidates = candidates[labels[candidates] < 0]
        difference = pilot_patches[candidates] - pilot_patches[seed]
        members = candidates[np.sum(difference * difference, axis=1) <= threshold]
        if seed not in members:
            raise ArithmeticError('group does not include its seed')
        labels[members] = len(groups)
        groups.append(members)
    return groups, labels


def encode_groups(patches: np.ndarray, dictionary: np.ndarray, groups: list[np.ndarray],
                  sigma: float, tau: float, progress: Callable | None = None) -> Stage:
    """Fit original noisy patch data, never the pilot or the clean image."""
    patches = _matrix(patches, 'patches')
    dictionary = validate_dictionary(dictionary)
    if patches.shape[1] != dictionary.shape[0]:
        raise ValueError('patch/dictionary mismatch')
    visits = np.zeros(len(patches), dtype=np.int64)
    for group in groups:
        group = np.asarray(group)
        if (group.ndim != 1 or len(group) == 0 or not np.issubdtype(group.dtype, np.integer)
                or np.any(group < 0) or np.any(group >= len(patches))):
            raise ValueError('invalid group membership')
        np.add.at(visits, group, 1)
    if not np.all(visits == 1):
        raise ValueError('groups must form a disjoint complete cover')
    reconstructed = np.empty_like(patches)
    pursuits = []
    for index, group in enumerate(groups):
        signals = patches[group].T
        means = np.mean(signals, axis=0, keepdims=True)
        epsilon = noise_budget(signals.shape[0], signals.shape[1], sigma, tau)
        fit = somp(dictionary, signals-means, epsilon)
        if fit.stop_reason != 'noise_budget':
            raise ArithmeticError(f'group {index} could not meet noise budget: {fit.stop_reason}')
        reconstructed[group] = (fit.reconstruction + means).T
        pursuits.append(fit)
        if progress is not None and (index % 256 == 0 or index + 1 == len(groups)):
            progress(index + 1, len(groups))
    return Stage(reconstructed, groups, pursuits)


def denoise(image: np.ndarray, dictionary: np.ndarray, sigma: float, *, tau: float = .8,
            window: int = 32, stage: str = 'ssc', progress: Callable | None = None) -> Result:
    """Stage P1 only: fixed dictionary, initial SC, optional pilot-guided SSC."""
    image = _matrix(image, 'image')
    dictionary = validate_dictionary(dictionary)
    block = int(np.sqrt(dictionary.shape[0]))
    if block * block != dictionary.shape[0] or stage not in ('sc', 'ssc'):
        raise ValueError('square grayscale dictionary and stage sc/ssc required')
    noise_budget(block * block, 1, sigma, tau)
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError('window must be a positive integer')
    patches, positions = extract_patches(image, block)
    singleton = [np.asarray([i], dtype=np.int64) for i in range(len(patches))]
    pilot_stage = encode_groups(patches, dictionary, singleton, sigma, tau,
                                (lambda n, total: progress('sc', n, total)) if progress else None)
    pilot, coverage = aggregate(pilot_stage.patches, positions, image.shape, block)
    threshold = similarity_threshold(sigma, block * block)
    if stage == 'sc':
        return Result(pilot.copy(), pilot, coverage, positions, np.arange(len(patches)),
                      pilot_stage, pilot_stage, threshold)
    pilot_patches, _ = extract_patches(pilot, block)
    groups, labels = greedy_groups(pilot_patches, (image.shape[0]-block+1, image.shape[1]-block+1),
                                  window, threshold)
    final_stage = encode_groups(patches, dictionary, groups, sigma, tau,
                                (lambda n, total: progress('ssc', n, total)) if progress else None)
    output, final_coverage = aggregate(final_stage.patches, positions, image.shape, block)
    if not np.array_equal(coverage, final_coverage):
        raise ArithmeticError('final grouping changed patch coverage')
    return Result(output, pilot, coverage, positions, labels, pilot_stage, final_stage, threshold)
