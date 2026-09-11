from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import native
import p2_reference as p2
import reference as ref


@pytest.fixture(scope='module')
def engine():
    if not native.default_library().is_file():
        pytest.skip('opt-in native exploration library is not built')
    return native.Native()


def dictionary(m, k, seed=7):
    values = np.random.default_rng(seed).normal(size=(m, k))
    return values / np.linalg.norm(values, axis=0)


@pytest.mark.parametrize('solver', [0, 1])
@pytest.mark.parametrize('correlation', [0, 1])
@pytest.mark.parametrize('g', [1, 7])
def test_fp64_pursuit_against_frozen_svd(engine, solver, correlation, g):
    d = dictionary(16, 40)
    y = np.random.default_rng(33).normal(size=(16, g))
    epsilon = np.sum(y**2) * .12
    expected = (ref.somp if solver == 0 else p2.simultaneous_ols)(d, y, epsilon)
    actual = engine.solve(d, y, epsilon, solver=solver, correlation=correlation)
    np.testing.assert_array_equal(actual.support, expected.support)
    np.testing.assert_allclose(actual.reconstruction, expected.reconstruction, atol=3e-12, rtol=2e-12)
    np.testing.assert_allclose(d @ actual.coefficients, actual.reconstruction, atol=2e-12)
    np.testing.assert_allclose(actual.residual_history, expected.residual_history, atol=4e-12, rtol=1e-12)


@pytest.mark.parametrize('precision', [0, 1, 2])
def test_512_atoms_shared_support_beyond_32(engine, precision):
    d = np.concatenate([np.eye(64), dictionary(64, 448)], axis=1)
    y = np.eye(64)[:, :40]
    result = engine.solve(d, y, 1e-12, precision=precision, correlation=1)
    np.testing.assert_array_equal(result.support, np.arange(40))
    np.testing.assert_allclose(result.reconstruction, y, atol=1e-12)
    assert result.stats['support_max'] == 40


def test_near_tie_refinement(engine):
    d = np.eye(9)
    d[:, 1] = 0
    d[0, 1], d[1, 1] = np.cos(1e-4), np.sin(1e-4)
    y = d[:, 1:2]
    result = engine.solve(d, y, 1e-10, precision=2, correlation=1)
    np.testing.assert_array_equal(result.support, [1])
    assert result.stats['refined_candidates'] >= 2


def test_mixed_precision_is_not_a_universal_exactness_claim(engine):
    d = np.eye(9)
    d[:, 1] = 0
    d[0, 1], d[1, 1] = 1 - 2.0**-29, 2.0**-14
    y = d[:, 1:2]
    epsilon = 2.0**-27
    exact = engine.solve(d, y, epsilon, precision=0, correlation=1)
    mixed = engine.solve(d, y, epsilon, precision=1, correlation=1)
    refined = engine.solve(d, y, epsilon, precision=2, correlation=1)
    np.testing.assert_array_equal(exact.support, [1])
    np.testing.assert_array_equal(mixed.support, [0])
    np.testing.assert_array_equal(refined.support, exact.support)
    assert np.max(np.abs(mixed.reconstruction - exact.reconstruction)) > 5e-5
    assert mixed.stats['last_residual'] <= epsilon
    np.testing.assert_array_equal(refined.reconstruction, exact.reconstruction)


def test_rank_failure_is_not_mean_only_success(engine):
    d = np.ones((9, 12)) / 3
    y = np.arange(9, dtype=float)[:, None]
    y -= y.mean()
    with pytest.raises(native.NativeError, match='rank exhausted'):
        engine.solve(d, y, 1e-12, correlation=1)


def test_support_cap_is_explicit(engine):
    d = dictionary(16, 40)
    y = np.random.default_rng(91).normal(size=(16, 5))
    result = engine.solve(d, y, 1e-12, max_support=2, correlation=1, precision=1)
    assert len(result.support) == 2
    assert result.stats['capped_groups'] == 1
    assert result.stats['last_residual'] > 1e-12


@pytest.mark.parametrize('grouping', [0, 1])
@pytest.mark.parametrize('match_precision', [0, 1])
def test_grouping_reuses_distance_not_topk_semantics(engine, grouping, match_precision):
    pilot = np.random.default_rng(19).normal(scale=.02, size=(13, 15))
    patches, _ = ref.extract_patches(pilot, 3)
    threshold = ref.similarity_threshold(.04, 9)
    expected = (p2.overlapping_groups(patches, (11, 13), 6, threshold) if grouping else
                ref.greedy_groups(patches, (11, 13), 6, threshold)[0])
    actual = engine.groups(pilot, 3, .04, window=6, grouping=grouping, match_precision=match_precision)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_capped_equal_distance_groups_keep_seed_and_complete_cover(engine):
    groups = engine.groups(np.ones((13, 15)), 3, .04, window=6, max_group=2)
    assert max(map(len, groups)) == 2
    np.testing.assert_array_equal(np.unique(np.concatenate(groups)), np.arange(11 * 13))


@pytest.mark.parametrize('solver,grouping', [(0, 0), (1, 1)])
def test_small_image_against_frozen_reference(engine, solver, grouping):
    d = dictionary(9, 24)
    image = np.random.default_rng(37).normal(loc=.5, scale=.08, size=(11, 13))
    expected = p2.denoise_diagnostic(image, d, .04, window=6,
                                     solver='tropp_l1' if solver == 0 else 'energy_gain',
                                     grouping='overlap' if grouping else 'disjoint')
    actual = engine.denoise(image, d, .04, window=6, solver=solver, grouping=grouping, correlation=1)
    np.testing.assert_allclose(actual.pilot, expected.pilot, atol=2e-12, rtol=1e-12)
    np.testing.assert_allclose(actual.output, expected.output, atol=2e-12, rtol=1e-12)
    np.testing.assert_array_equal(actual.coverage, expected.coverage)
    assert actual.stats['capped_groups'] == 0
    assert actual.stats['worst_budget_ratio'] <= 1 + 1e-9


@pytest.mark.parametrize('block', [9, 12, 16])
def test_nonlegacy_geometry_and_zero_support(engine, block):
    image = np.full((block + 1, block + 2), .314159)
    d = dictionary(block**2, 512)
    result = engine.denoise(image, d, .1, window=3, correlation=1, precision=1)
    np.testing.assert_allclose(result.output, image, atol=2e-15)
    assert result.stats['support_max'] == 0
    assert result.coverage.min() > 0


def test_packing_batch_is_not_an_algorithm_change(engine):
    image = np.random.default_rng(5).normal(loc=.5, scale=.1, size=(13, 15))
    d = dictionary(9, 24)
    results = [engine.denoise(image, d, .05, window=6, correlation=1, batch=batch)
               for batch in [1, 7, 128]]
    for result in results[1:]:
        np.testing.assert_array_equal(result.output, results[0].output)
        np.testing.assert_array_equal(result.pilot, results[0].pilot)


def test_memory_limit_failure_and_next_request(engine):
    d = dictionary(81, 512)
    image = np.ones((13, 15))
    with pytest.raises(native.NativeError, match='memory_limit_mb exceeded') as error:
        engine.denoise(image, d, .05, memory_limit_bytes=4096)
    assert error.value.stats['tracked_peak_bytes'] <= 4096
    result = engine.denoise(image, d, .05, correlation=1)
    assert np.isfinite(result.output).all()
    np.testing.assert_array_equal(image, np.ones((13, 15)))


def test_requests_do_not_share_dictionary_or_workspace(engine):
    d = dictionary(9, 24)
    images = [np.random.default_rng(i).normal(loc=.5, scale=.1, size=(11, 13)) for i in [41, 42]]
    def run(index):
        return engine.denoise(images[index], d, .05, window=6, correlation=1, precision=2).output
    expected = [run(0), run(1)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(run, [1, 0, 1, 0]))
    for result, index in zip(actual, [1, 0, 1, 0]):
        np.testing.assert_array_equal(result, expected[index])


@pytest.mark.parametrize('image_passes,group_passes', [(1, 0), (0, 1), (1, 1)])
def test_learning_is_deterministic_and_invalidates_preparation(engine, image_passes, group_passes):
    image = np.random.default_rng(6).normal(loc=.5, scale=.12, size=(13, 15))
    d = np.eye(9)
    policy = dict(image_passes=image_passes, group_passes=group_passes, learning_samples=24,
                  learning_groups=4, learning_group_cap=8, learning_iterations=60, correlation=1,
                  precision=1, window=6, batch=8)
    first = engine.denoise(image, d, .08, **policy)
    second = engine.denoise(image, d, .08, **policy)
    np.testing.assert_array_equal(first.dictionary, second.dictionary)
    np.testing.assert_array_equal(first.output, second.output)
    assert first.stats['dictionary_versions'] == 1 + image_passes + group_passes
    assert first.stats['dictionary_change'] > 0
    assert first.stats['learning_fixed_code_drop'] >= -1e-9
    assert first.stats['learning_codes'] > 0
    assert first.stats['learning_converged'] <= first.stats['learning_codes']
    assert np.max(np.linalg.norm(first.dictionary, axis=0)) <= 1 + 1e-12
    np.testing.assert_array_equal(d, np.eye(9))


@pytest.mark.parametrize('area,maximum', [(9, 36), (81, 1024), (256, 128)])
def test_cached_budget_table_matches_reference(area, maximum):
    table = native.budget_table(area, maximum, 25 / 255)
    for g in [1, 2, maximum // 2, maximum]:
        assert table[g] == ref.noise_budget(area, g, 25 / 255)


def test_unknown_and_overflow_options_are_rejected():
    with pytest.raises(ValueError, match='unknown'):
        native.options(precison=1)
    with pytest.raises(ValueError, match='ABI range'):
        native.options(max_group=2**32)


@pytest.mark.parametrize('grouped', [False, True])
def test_penalized_code_against_identity_closed_form(engine, grouped):
    y = np.random.default_rng(83).normal(size=(9, 7))
    penalty = .3
    if grouped:
        expected = y * np.maximum(0, 1 - penalty / np.linalg.norm(y, axis=1))[:, None]
    else:
        expected = np.sign(y) * np.maximum(0, np.abs(y) - penalty)
    actual, stats = engine.code(np.eye(9), y, penalty, grouped=grouped, learning_tolerance=1e-5)
    np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=3e-7)
    assert stats['learning_converged'] == 1


@pytest.mark.parametrize('grouped', [False, True])
def test_penalized_code_independent_kkt_check(engine, grouped):
    d = dictionary(9, 12, seed=86)
    y = np.random.default_rng(91).normal(size=(9, 4))
    penalty = .35
    actual, stats = engine.code(d, y, penalty, grouped=grouped,
                                 learning_iterations=3000, learning_tolerance=2e-5)
    gradient = d.T @ (d @ actual - y)
    if grouped:
        norms = np.linalg.norm(actual, axis=1)
        active = norms > 1e-7
        violation = np.zeros(len(actual))
        violation[active] = np.linalg.norm(gradient[active] + penalty * actual[active] / norms[active, None], axis=1)
        violation[~active] = np.maximum(0, np.linalg.norm(gradient[~active], axis=1) - penalty)
    else:
        active = np.abs(actual) > 1e-7
        violation = np.maximum(0, np.abs(gradient) - penalty)
        violation[active] = np.abs(gradient[active] + penalty * np.sign(actual[active]))
    assert np.max(violation) < 2e-4
    assert stats['learning_converged'] == 1


@pytest.mark.parametrize('rows', [81, 144, 256])
@pytest.mark.parametrize('mode', [2, 3, 6, 7])
def test_transposed_and_sme_products_keep_logical_rows_and_shared_group(engine, rows, mode):
    d = dictionary(rows, 512, seed=73)
    amplitudes = np.random.default_rng(14).normal(size=(3, 129)) * np.array([1, .3, .1])[:, None]
    y = d[:, [11, 70, 321]] @ amplitudes
    expected = engine.solve(d, y, 1e-10, correlation=1)
    actual = engine.solve(d, y, 1e-10, correlation=mode, precision=1)
    np.testing.assert_array_equal(actual.support, expected.support)
    np.testing.assert_allclose(actual.reconstruction, expected.reconstruction, atol=3e-12)
    assert actual.stats['last_rank'] == 3
    assert actual.stats['worst_budget_ratio'] <= 1 + 1e-9
    if mode & 4 and actual.stats['sme_available']:
        assert actual.stats['correlation_sme32'] >= 2
    else:
        assert actual.stats['correlation_nn32'] > 0


def test_sme_path_and_neon_path_are_request_local(engine):
    d = dictionary(81, 512, seed=18)
    image = np.random.default_rng(67).normal(loc=.5, scale=.05, size=(21, 23))
    policies = [dict(precision=1, correlation=3), dict(precision=1, correlation=7)]
    expected = [engine.denoise(image, d, .04, **policy) for policy in policies]
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(lambda i: engine.denoise(image, d, .04, **policies[i]), [1, 0]))
    for result, index in zip(actual, [1, 0]):
        np.testing.assert_array_equal(result.output, expected[index].output)
        assert result.stats['correlation_sme32'] == expected[index].stats['correlation_sme32']


@pytest.mark.parametrize('mode', [1, 3, 7])
@pytest.mark.parametrize('grouped', [False, True])
def test_shared_learning_gradient_backend_has_closed_form_oracle(engine, mode, grouped):
    d = np.tile(np.eye(81), (1, 7))[:, :512]
    y = np.random.default_rng(38).normal(size=(81, 129))
    penalty = .3
    if grouped:
        expected = y * np.maximum(0, 1 - penalty / np.linalg.norm(y, axis=1))[:, None]
    else:
        expected = np.sign(y) * np.maximum(0, np.abs(y) - penalty)
    coefficients, stats = engine.code(d, y, penalty, grouped=grouped, correlation=mode,
                                       learning_iterations=96, learning_tolerance=1e-5)
    np.testing.assert_allclose(d @ coefficients, expected, atol=4e-6, rtol=4e-6)
    assert stats['learning_converged'] == 1
    if mode == 7 and stats['sme_available']:
        assert stats['correlation_sme32'] > 0
