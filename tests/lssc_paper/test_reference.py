import numpy as np
import pytest

from reference import (aggregate, denoise, encode_groups, extract_patches, greedy_groups,
                       noise_budget, similarity_threshold, somp, validate_dictionary)


def test_noise_budget_uses_group_chi_square_not_linear_lambda():
    assert noise_budget(2, 1, .1) == pytest.approx(.01 * -2 * np.log(.2), rel=1e-12)
    assert noise_budget(1, 2, .1) == noise_budget(2, 1, .1)
    assert noise_budget(2, 2, .1) != pytest.approx(2 * noise_budget(2, 1, .1))
    assert noise_budget(81, 7, 0) == 0


@pytest.mark.parametrize('m,n,sigma,tau', [(0,1,.1,.8), (1,0,.1,.8),
    (1,1,-1,.8), (1,1,np.nan,.8), (1,1,.1,0), (1,1,.1,1), (1,1,.1,np.nan)])
def test_invalid_noise_budget(m, n, sigma, tau):
    with pytest.raises(ValueError):
        noise_budget(m, n, sigma, tau)


def test_somp_uses_l1_correlations_not_l2():
    signals = np.array([[1., 1.], [1.5, 0.]])
    result = somp(np.eye(2), signals, 2.3)
    assert result.support.tolist() == [0]
    assert result.coefficients.tolist() == [[1., 1.]]
    assert result.residual_squared == 2.25
    assert result.stop_reason == 'noise_budget'


def test_shared_support_different_amplitudes_and_signs_no_soft_bias():
    signals = np.array([[3., -1., .5], [0., 0., 0.], [1., 2., -4.]])
    result = somp(np.eye(3), signals, 0)
    assert set(result.support) == {0, 2}
    np.testing.assert_allclose(result.reconstruction, signals, atol=1e-14)
    np.testing.assert_allclose(result.coefficients, signals[result.support], atol=1e-14)
    assert np.all(np.diff(result.residual_history) <= 0)


def test_nonorthogonal_support_is_refit_with_least_squares():
    dictionary = np.array([[1., .8, 0.], [0., .6, 0.], [0., 0., 1.]])
    signals = dictionary[:, :2] @ np.array([[2., -1.], [.5, 3.]])
    result = somp(dictionary, signals, 1e-20)
    assert set(result.support) == {0, 1}
    residual = signals - result.reconstruction
    np.testing.assert_allclose(dictionary[:, result.support].T @ residual, 0, atol=1e-13)
    np.testing.assert_allclose(result.reconstruction, signals, atol=1e-13)


def test_rank_deficiency_is_visible_not_silently_accepted():
    dictionary = np.array([[1., 1.], [0., 0.]])
    result = somp(dictionary, np.array([[0.], [1.]]), .01)
    assert result.stop_reason == 'span_exhausted'
    assert result.residual_squared == 1
    assert len(result.support) == 0


def test_zero_support_is_valid_when_noise_budget_already_met():
    result = somp(np.eye(2), np.ones((2, 3)) * .01, 1)
    assert result.support.size == 0
    assert result.coefficients.shape == (0, 3)
    np.testing.assert_array_equal(result.reconstruction, 0)


def test_no_fixed_eight_atom_or_sixteen_iteration_limit():
    signals = np.arange(1., 25.).reshape(24, 1)
    result = somp(np.eye(24), signals, 0)
    assert len(result.support) == 24
    assert result.stop_reason == 'noise_budget'


def test_duplicate_atoms_do_not_create_duplicate_support():
    dictionary = np.column_stack((np.eye(3), np.eye(3)))
    result = somp(dictionary, np.array([[3., 1.], [2., -1.], [1., .5]]), 0)
    assert len(result.support) == 3
    assert np.all(result.support < 3)
    assert result.residual_squared < 1e-20


@pytest.mark.parametrize('dictionary', [np.ones((3,2)), np.zeros((3,2)),
    np.array([[np.nan]]), np.array([[1j]]), np.zeros((0,2))])
def test_dictionary_validation_does_not_normalize_or_hide_bad_data(dictionary):
    with pytest.raises(ValueError):
        validate_dictionary(dictionary)


def test_column_major_patch_orientation_and_raster_locations():
    image = np.arange(12.).reshape(3, 4)
    patches, positions = extract_patches(image, 2)
    assert patches[0].tolist() == [0, 4, 1, 5]
    assert patches[1].tolist() == [1, 5, 2, 6]
    assert positions.tolist() == [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]


@pytest.mark.parametrize('block', [1, 2, 3, 4, 9, 12, 16])
def test_dense_patch_roundtrip_covers_odd_sizes_and_non_power_of_two_blocks(block):
    image = np.random.default_rng(9).normal(size=(block+5, block+7))
    patches, positions = extract_patches(image, block)
    reconstructed, coverage = aggregate(patches, positions, image.shape, block)
    np.testing.assert_allclose(reconstructed, image, atol=2e-14)
    assert coverage.min() == 1
    assert coverage.max() == min(block, 6) * min(block, 8)


def test_missing_patch_coverage_is_an_error_not_noisy_fallback():
    with pytest.raises(ValueError, match='uncovered'):
        aggregate(np.ones((1, 4)), np.array([[0, 0]]), (4,4), 2)


def test_aggregation_is_count_normalized_not_fixed_divide_by_patch_area():
    patches, positions = extract_patches(np.ones((5,6)) * .37, 3)
    result, coverage = aggregate(patches, positions, (5,6), 3)
    np.testing.assert_allclose(result, .37, atol=1e-15)
    assert coverage[0,0] == 1 and coverage[2,2] == 9


def test_grouping_covers_once_and_respects_local_window():
    pilot = np.zeros((9, 4))
    groups, labels = greedy_groups(pilot, (3,3), 3, 0)
    assert [g.tolist() for g in groups] == [[0,1,3,4], [2,5], [6,7], [8]]
    np.testing.assert_array_equal(np.sort(np.concatenate(groups)), np.arange(9))
    for label, group in enumerate(groups):
        assert np.all(labels[group] == label)


def test_threshold_is_inclusive_and_does_not_force_outliers_into_a_group():
    groups, _ = greedy_groups(np.array([[0.], [1.], [10.]]), (1,3), 7, 1)
    assert [g.tolist() for g in groups] == [[0,1], [2]]


def test_no_fixed_cluster_count_or_group_size_limit():
    separated = np.arange(90.).reshape(1,90).T
    groups, _ = greedy_groups(separated, (1,90), 32, 0)
    assert len(groups) == 90
    groups, _ = greedy_groups(np.zeros((90,1)), (1,90), 181, 0)
    assert len(groups) == 1 and len(groups[0]) == 90


def test_literal_ssd_threshold_has_explicit_patch_area_scaling():
    assert similarity_threshold(25/255, 81) == pytest.approx((32*25/255)**2/81)


def test_encode_groups_uses_original_data_and_restores_each_patch_mean():
    original = np.array([[.1,.2,.3,.4], [.5,.6,.7,.8]])
    stage = encode_groups(original, np.eye(4), [np.array([0,1])], 0, .8)
    np.testing.assert_allclose(stage.patches, original, atol=1e-14)
    np.testing.assert_allclose(stage.patches.mean(axis=1), original.mean(axis=1), atol=1e-14)


@pytest.mark.parametrize('groups', [[np.array([0])], [np.array([0,0,1])],
                                  [np.array([0,1]), np.array([1])]])
def test_group_cover_is_checked(groups):
    with pytest.raises(ValueError, match='disjoint complete'):
        encode_groups(np.ones((2,4)), np.eye(4), groups, .1, .8)


def test_full_pipeline_is_deterministic_and_does_not_mutate_inputs():
    image = .4 + np.random.default_rng(5).normal(0, .04, (9,10))
    dictionary = np.eye(9)
    before, before_dictionary = image.copy(), dictionary.copy()
    first = denoise(image, dictionary, .04, window=5)
    second = denoise(image, dictionary, .04, window=5)
    np.testing.assert_array_equal(first.output, second.output)
    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(dictionary, before_dictionary)
    assert np.mean((first.output-.4)**2) < np.mean((image-.4)**2)
    assert len(first.pilot_stage.groups) == 56
    assert all(r.stop_reason == 'noise_budget' for r in first.final_stage.pursuits)


def test_zero_noise_roundtrip_with_spanning_dictionary():
    image = np.random.default_rng(11).normal(size=(7,8))
    result = denoise(image, np.eye(9), 0, window=5)
    np.testing.assert_allclose(result.output, image, atol=1e-13)


def test_sc_stage_is_an_explicit_non_grouped_baseline():
    image = np.random.default_rng(11).normal(size=(7,8))
    result = denoise(image, np.eye(9), .1, stage='sc')
    np.testing.assert_array_equal(result.output, result.pilot)
    assert all(len(group) == 1 for group in result.final_stage.groups)
