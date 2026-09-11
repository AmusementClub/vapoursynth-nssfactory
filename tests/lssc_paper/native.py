"""Typed adapter for the opt-in native LSSC exploration library.

P1/P2 stay frozen. This module does not register a filter or contain author code.
Learning is an explicitly penalized, sampled, alternating approximation.
"""
from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import chi2

from reference import _matrix, noise_budget, validate_dictionary


INTEGER_OPTIONS = [
    'precision', 'correlation', 'solver', 'grouping', 'match_precision', 'window',
    'max_group', 'max_support', 'batch', 'image_passes', 'group_passes',
    'learning_samples', 'learning_groups', 'learning_iterations', 'learning_group_cap', 'reserved',
]
REAL_OPTIONS = ['sigma', 'threshold_multiplier', 'epsilon_scale', 'learning_lambda', 'learning_tolerance']
INTEGER_STATS = [
    'groups', 'occurrences', 'pilot_patches', 'support_sum', 'support_max', 'capped_groups',
    'rank_rejections', 'refined_candidates', 'correlation_refreshes', 'dictionary_versions',
    'tracked_peak_bytes', 'learning_codes', 'learning_converged', 'learning_steps',
    'learning_backtracks', 'dictionary_updates', 'last_rank', 'last_history_size', 'group_max',
]
REAL_STATS = [
    'last_residual', 'worst_budget_ratio', 'dictionary_change', 'learning_stationarity_max',
    'learning_fixed_code_drop', 'prepare_seconds', 'image_learning_seconds', 'pilot_seconds',
    'matching_seconds', 'group_learning_seconds', 'final_seconds', 'total_seconds',
]


class COptions(ct.Structure):
    _fields_ = ([(name, ct.c_int32) for name in INTEGER_OPTIONS] +
                [(name, ct.c_double) for name in REAL_OPTIONS] + [('memory_limit_bytes', ct.c_uint64)])


class CStats(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in INTEGER_STATS] + [(name, ct.c_double) for name in REAL_STATS]


DEFAULTS = dict(zip(INTEGER_OPTIONS, [0, 0, 1, 1, 0, 32, 0, 0, 128, 0, 0, 2048, 64, 48, 64, 0]))
DEFAULTS.update(sigma=25 / 255, threshold_multiplier=1., epsilon_scale=1.,
                learning_lambda=.75, learning_tolerance=1e-4, memory_limit_bytes=1024**3)


def options(**values):
    unknown = values.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f'unknown native options: {sorted(unknown)}')
    merged = DEFAULTS | values
    for name in INTEGER_OPTIONS + ['memory_limit_bytes']:
        if not isinstance(merged[name], (int, np.integer)):
            raise ValueError(f'{name} must be an integer')
        limit = 2**64 if name == 'memory_limit_bytes' else 2**31
        if not 0 <= merged[name] < limit:
            raise ValueError(f'{name} outside ABI range')
    return COptions(**merged)


def stats_dict(stats):
    return {name: getattr(stats, name) for name in INTEGER_STATS + REAL_STATS}


def default_library():
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    default = Path(__file__).resolve().parents[2] / 'artifacts/lssc-native-build/tests/lssc_paper/native'
    return Path(os.environ.get('NSS_LSSC_NATIVE_LIBRARY', default / ('libnss_lssc_explore' + suffix)))


def budget_table(area, maximum_group, sigma, tau=.8):
    # Reuse the reference's validation and exact formula. Vectorizing ppf avoids
    # repeating Python/scipy setup once for every encountered group.
    first = noise_budget(area, 1, sigma, tau)
    if not isinstance(maximum_group, int) or maximum_group < 1 or maximum_group > 16384:
        raise ValueError('invalid budget table size')
    result = np.zeros(maximum_group + 1, dtype=np.float64)
    result[1:] = sigma**2 * chi2.ppf(tau, area * np.arange(1, maximum_group + 1))
    if result[1] != first or not np.isfinite(result).all():
        raise ArithmeticError('budget formula mismatch/nonfinite quantile')
    return result


class NativeError(RuntimeError):
    def __init__(self, message, stats):
        super().__init__(message)
        self.stats = stats


@dataclass
class NativePursuit:
    support: np.ndarray
    coefficients: np.ndarray
    reconstruction: np.ndarray
    residual_history: np.ndarray
    stats: dict


@dataclass
class NativeResult:
    output: np.ndarray
    pilot: np.ndarray
    dictionary: np.ndarray
    coverage: np.ndarray
    stats: dict


PD = ct.POINTER(ct.c_double)
PI = ct.POINTER(ct.c_int32)
PU32 = ct.POINTER(ct.c_uint32)
PU64 = ct.POINTER(ct.c_uint64)
PO = ct.POINTER(COptions)
PS = ct.POINTER(CStats)


def ptr(array, pointer_type=PD):
    return array.ctypes.data_as(pointer_type)


class Native:
    def __init__(self, library=None):
        self.path = Path(library) if library else default_library()
        self.lib = ct.CDLL(str(self.path.resolve()))
        self.lib.nss_explore_abi.restype = ct.c_uint32
        self.lib.nss_explore_options_size.restype = ct.c_uint64
        self.lib.nss_explore_stats_size.restype = ct.c_uint64
        self.lib.nss_explore_error.restype = ct.c_char_p
        if (self.lib.nss_explore_abi() != 1 or self.lib.nss_explore_options_size() != ct.sizeof(COptions)
                or self.lib.nss_explore_stats_size() != ct.sizeof(CStats)):
            raise RuntimeError('native exploration ABI mismatch; rebuild library and adapter together')
        self.lib.nss_explore_solve.argtypes = [PD, PD, ct.c_int, ct.c_int, ct.c_int, ct.c_double,
                                               PO, PD, PD, PI, PD, PS]
        self.lib.nss_explore_denoise.argtypes = [PD, ct.c_int, ct.c_int, PD, ct.c_int, ct.c_int,
                                                 PD, ct.c_int, PO, PD, PD, PD, PU32, PS]
        self.lib.nss_explore_groups.argtypes = [PD, ct.c_int, ct.c_int, ct.c_int, PO,
                                                PU64, PI, ct.c_uint64, PS]
        if hasattr(self.lib, 'nss_explore_code'):
            self.lib.nss_explore_code.argtypes = [PD, PD, ct.c_int, ct.c_int, ct.c_int,
                                                  ct.c_double, ct.c_int, PO, PD, PS]
        if hasattr(self.lib, 'nss_explore_backend_counts'):
            self.lib.nss_explore_backend_counts.argtypes = [PU64, ct.c_int]

    def result_stats(self, stats):
        result = stats_dict(stats)
        if hasattr(self.lib, 'nss_explore_backend_counts'):
            counts = (ct.c_uint64 * 4)()
            if self.lib.nss_explore_backend_counts(counts, 4):
                raise RuntimeError('backend diagnostic ABI failure')
            result.update(zip(['correlation_tn64', 'correlation_tn32', 'correlation_nn32', 'correlation_sme32'], counts))
            result['sme_available'] = bool(self.lib.nss_explore_sme_available())
        return result

    def check(self, code, stats):
        if code:
            raise NativeError(self.lib.nss_explore_error().decode('utf8'), self.result_stats(stats))

    def solve(self, dictionary, signals, epsilon, **policy):
        dictionary = np.asfortranarray(validate_dictionary(dictionary))
        signals = np.asfortranarray(_matrix(signals, 'signals'))
        m, k = dictionary.shape
        if signals.shape[0] != m:
            raise ValueError('dictionary/signal dimensions differ')
        g = signals.shape[1]
        o, stats = options(**policy), CStats()
        output = np.empty_like(signals, order='F')
        coefficients = np.empty((k, g), dtype=np.float64, order='F')
        support = np.empty(min(m, k), dtype=np.int32)
        history = np.empty(min(m, k) + 1, dtype=np.float64)
        code = self.lib.nss_explore_solve(ptr(dictionary), ptr(signals), m, k, g, epsilon,
            ct.byref(o), ptr(output), ptr(coefficients), ptr(support, PI), ptr(history), ct.byref(stats))
        self.check(code, stats)
        return NativePursuit(support[:stats.last_rank].copy(), coefficients, output,
                             history[:stats.last_history_size].copy(), self.result_stats(stats))

    def denoise(self, image, dictionary, sigma, *, tau=.8, **policy):
        image = np.ascontiguousarray(_matrix(image, 'image'))
        dictionary = np.asfortranarray(validate_dictionary(dictionary))
        m, atoms = dictionary.shape
        block = int(np.sqrt(m))
        if block * block != m:
            raise ValueError('dictionary patch area must be square')
        o, stats = options(sigma=sigma, **policy), CStats()
        maximum = o.max_group or o.window**2
        epsilons = budget_table(m, maximum, sigma, tau)
        output, pilot = np.empty_like(image), np.empty_like(image)
        learned = np.empty_like(dictionary, order='F')
        coverage = np.empty(image.shape, dtype=np.uint32)
        height, width = image.shape
        code = self.lib.nss_explore_denoise(ptr(image), width, height, ptr(dictionary), block, atoms,
            ptr(epsilons), len(epsilons), ct.byref(o), ptr(output), ptr(pilot), ptr(learned),
            ptr(coverage, PU32), ct.byref(stats))
        self.check(code, stats)
        return NativeResult(output, pilot, learned, coverage, self.result_stats(stats))

    def groups(self, pilot, block, sigma, **policy):
        pilot = np.ascontiguousarray(_matrix(pilot, 'pilot'))
        o, stats = options(sigma=sigma, **policy), CStats()
        height, width = pilot.shape
        code = self.lib.nss_explore_groups(ptr(pilot), width, height, block, ct.byref(o),
                                          None, None, 0, ct.byref(stats))
        self.check(code, stats)
        offsets = np.empty((height - block + 1) * (width - block + 1) + 1, dtype=np.uint64)
        members = np.empty(stats.occurrences, dtype=np.int32)
        code = self.lib.nss_explore_groups(ptr(pilot), width, height, block, ct.byref(o),
            ptr(offsets, PU64), ptr(members, PI), len(members), ct.byref(stats))
        self.check(code, stats)
        return [members[offsets[i]:offsets[i + 1]].copy() for i in range(stats.groups)]

    def code(self, dictionary, signals, penalty, *, grouped=False, **policy):
        dictionary = np.asfortranarray(validate_dictionary(dictionary))
        signals = np.asfortranarray(_matrix(signals, 'signals'))
        m, k = dictionary.shape
        if signals.shape[0] != m:
            raise ValueError('dictionary/signal dimensions differ')
        o, stats = options(**policy), CStats()
        coefficients = np.empty((k, signals.shape[1]), dtype=np.float64, order='F')
        code = self.lib.nss_explore_code(ptr(dictionary), ptr(signals), m, k, signals.shape[1],
            penalty, int(grouped), ct.byref(o), ptr(coefficients), ct.byref(stats))
        self.check(code, stats)
        return coefficients, self.result_stats(stats)
