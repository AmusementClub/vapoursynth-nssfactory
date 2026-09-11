import numpy as np
import pytest

from native_campaign import digest64, selected_dictionary, summarize


def test_dictionary_budget_is_a_stratified_subset_not_first_columns():
    data = np.arange(3 * 512).reshape(3, 512)
    selected = selected_dictionary(data, 256)
    np.testing.assert_array_equal(selected, data[:, 1::2])
    with pytest.raises(ValueError):
        selected_dictionary(data, 513)


def test_paired_summary_uses_common_pairs_and_keeps_failures():
    rows = [dict(variant='a', pair=0, seconds=10, native_seconds=9, repeat_fp64_exact=True),
            dict(variant='a', pair=1, seconds=20, native_seconds=19, repeat_fp64_exact=True),
            dict(variant='b', pair=0, seconds=5, native_seconds=4, repeat_fp64_exact=True),
            dict(variant='b', pair=1, failed=True, failure='rank')]
    result = summarize(rows, ['a', 'b'])
    assert result['b']['paired_speedup_median'] == 2
    assert result['b']['complete_pairs'] == [0]
    assert result['b']['failures'][0]['failure'] == 'rank'


def test_repeat_hash_is_float64_not_float32():
    first = np.array([.5], dtype=np.float64)
    second = np.nextafter(first, np.inf)
    assert digest64(first) != digest64(second)
    np.testing.assert_array_equal(first.astype(np.float32), second.astype(np.float32))


def test_c4_rejects_the_whole_pair_when_one_variant_has_sibling_activity():
    rows = [dict(variant=name, pair=pair, seconds=1, native_seconds=.9, repeat_fp64_exact=True,
                 eligible_idle_sample=not (name == 'b' and pair == 1))
            for pair in [0, 1] for name in ['a', 'b']]
    result = summarize(rows, ['a', 'b'], require_c4=True)
    assert result['a']['complete_pairs'] == result['b']['complete_pairs'] == [0]
    assert result['a']['excluded_pairs'] == result['b']['excluded_pairs'] == [1]
