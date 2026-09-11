import copy

import pytest

from author_summary import paired_summary


def sample():
    return {'policies': [('fixed', 0, 0), ('full', 20, 5)], 'pairs': 3,
            'rows': [{'pair': pair, 'variant': variant, 'seconds': seconds,
                      'eligible_idle_sample': True, 'output_matches_warm': True,
                      'dictionary_matches_warm': True}
                     for pair in range(3) for variant, seconds in [('fixed', 1 + pair), ('full', 10 + pair)]]}


def test_common_pair_rejection_never_compares_different_sample_subsets():
    report = sample()
    report['rows'][-1]['eligible_idle_sample'] = False
    before = copy.deepcopy(report)
    result = paired_summary(report)
    assert report == before
    assert result['accepted_pairs'] == [0, 1]
    assert result['median_seconds'] == {'fixed': 1.5, 'full': 10.5}
    assert result['paired_runtime_ratios_to_baseline']['full'] == [10., 5.5]
    assert result['rejected_pairs'][0]['pair'] == 2


def test_missing_or_changed_output_is_not_a_successful_pair():
    report = sample()
    report['rows'].pop()
    report['rows'][0]['output_matches_warm'] = False
    report['rows'][2]['dictionary_matches_warm'] = False
    result = paired_summary(report)
    assert result['median_seconds'] is None and result['accepted_pairs'] == []


def test_duplicate_identity_rejected():
    report = sample()
    report['rows'].append(report['rows'][0])
    with pytest.raises(ValueError, match='duplicate'):
        paired_summary(report)
