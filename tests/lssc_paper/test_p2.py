import numpy as np
import pytest

from p2_reference import denoise_diagnostic, overlapping_groups, simultaneous_ols
from reference import denoise, somp


def test_energy_gain_and_tropp_l1_are_observably_different():
    dictionary=np.eye(2)
    signals=np.array([[1.,1.],[1.5,0.]])
    assert somp(dictionary,signals,2.3).support.tolist()==[0]
    fit=simultaneous_ols(dictionary,signals,2.3)
    assert fit.support.tolist()==[1]
    assert fit.residual_squared==2


def test_energy_gain_matches_exhaustive_one_atom_ls_improvement():
    rng=np.random.default_rng(8)
    dictionary=rng.normal(size=(7,12))
    dictionary/=np.linalg.norm(dictionary,axis=0)
    signals=rng.normal(size=(7,4))
    residuals=[]
    for atom in range(dictionary.shape[1]):
        active=dictionary[:,[atom]]
        fitted=active@np.linalg.lstsq(active,signals,rcond=None)[0]
        residuals.append(np.sum((signals-fitted)**2))
    epsilon=(min(residuals)+np.sum(signals**2))/2
    fit=simultaneous_ols(dictionary,signals,epsilon)
    assert fit.support.tolist()==[int(np.argmin(residuals))]
    assert fit.residual_squared==pytest.approx(min(residuals))


def test_later_energy_gain_refits_nonorthogonal_support():
    rng=np.random.default_rng(31)
    dictionary=rng.normal(size=(8,16))
    dictionary/=np.linalg.norm(dictionary,axis=0)
    signals=rng.normal(size=(8,3))
    fit=simultaneous_ols(dictionary,signals,0)
    support=[]
    for selected in fit.support:
        choices=[]
        for atom in range(dictionary.shape[1]):
            if atom in support:
                continue
            active=dictionary[:,support+[atom]]
            coef,_,rank,_=np.linalg.lstsq(active,signals,rcond=None)
            if rank==len(support)+1:
                choices.append((np.sum((signals-active@coef)**2),atom))
        # At the final full-rank step several atoms fit exactly; compare gain,
        # not an arbitrary machine-rounding tie among equivalent spans.
        best=min(value for value,_ in choices)
        actual=next(value for value,atom in choices if atom==selected)
        assert actual<=best+1e-10
        support.append(int(selected))
    assert fit.stop_reason=='noise_budget'
    np.testing.assert_allclose(fit.reconstruction,signals,atol=1e-12)


def test_overlap_has_duplicate_occurrences_but_does_not_drop_patches():
    groups=overlapping_groups(np.zeros((9,4)),(3,3),3,0)
    assert [g.tolist() for g in groups]==[[0,1,3,4],[1,2,4,5],[3,4,6,7],[4,5,7,8]]
    sources=np.concatenate(groups)
    assert len(sources)==16
    np.testing.assert_array_equal(np.unique(sources),np.arange(9))


def test_diagnostic_control_preserves_frozen_p1_exactly():
    image=.4+np.random.default_rng(9).normal(0,.05,(10,11))
    dictionary=np.eye(9)
    original=denoise(image,dictionary,.05,window=5)
    diagnostic=denoise_diagnostic(image,dictionary,.05,window=5)
    np.testing.assert_array_equal(diagnostic.pilot,original.pilot)
    np.testing.assert_array_equal(diagnostic.output,original.output)
    np.testing.assert_array_equal(diagnostic.coverage,original.coverage)


def test_overlap_pipeline_uses_occurrence_counts_and_remains_constant():
    image=np.ones((10,11))*.4
    result=denoise_diagnostic(image,np.eye(9),.1,grouping='overlap',window=5)
    np.testing.assert_allclose(result.output,.4,atol=1e-15)
    assert len(result.occurrence_sources)>len(result.positions)
    assert result.coverage.min()>0


def test_energy_rank_failure_is_exposed():
    result=simultaneous_ols(np.array([[1.,1.],[0.,0.]]),np.array([[0.],[1.]]),.1)
    assert result.stop_reason=='span_exhausted'


def test_invalid_p2_policy_does_not_silently_choose_an_alternative():
    with pytest.raises(ValueError):
        denoise_diagnostic(np.ones((5,5)),np.eye(9),.1,solver='unknown')
