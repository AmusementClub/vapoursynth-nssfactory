import numpy as np
import pytest

import p2_reference
import reference
from preparation_probe import invoke, prepared


@pytest.mark.parametrize('variant', ['p1', 'p2'])
@pytest.mark.parametrize('sigma', [.01, .1])
def test_exact_preparation_reuse_preserves_arrays_and_inputs(variant, sigma):
    image = .4 + np.random.default_rng(4).normal(0, sigma, (12, 13))
    dictionary = np.eye(9)
    originals = (reference.validate_dictionary, reference.noise_budget,
                 p2_reference.validate_dictionary, p2_reference.noise_budget)
    baseline, _ = invoke(image, dictionary, sigma, variant)
    reused, counters = invoke(image, dictionary, sigma, variant, True)
    np.testing.assert_array_equal(reused.output, baseline.output)
    np.testing.assert_array_equal(reused.pilot, baseline.pilot)
    np.testing.assert_array_equal(reused.coverage, baseline.coverage)
    np.testing.assert_array_equal(dictionary, np.eye(9))
    assert dictionary.flags.writeable
    assert counters['validation_reuses'] > 1
    assert counters['budget_cache']['hits'] > counters['budget_cache']['misses']
    assert originals == (reference.validate_dictionary, reference.noise_budget,
                         p2_reference.validate_dictionary, p2_reference.noise_budget)


def test_invalid_dictionary_and_exception_restore_checks():
    with pytest.raises(ValueError, match='unit norm'):
        invoke(np.ones((4, 4)), np.ones((9, 9)), .1, 'p1', True)
    original = reference.validate_dictionary
    with pytest.raises(RuntimeError, match='nested'):
        with prepared(np.eye(9), {}):
            with prepared(np.eye(9), {}):
                pass
    assert reference.validate_dictionary is original


def test_foreign_dictionary_is_never_implicitly_trusted():
    with prepared(np.eye(9), {}) as dictionary:
        assert not dictionary.flags.writeable
        with pytest.raises(ValueError, match='unit norm'):
            reference.validate_dictionary(np.ones((9, 9)))
