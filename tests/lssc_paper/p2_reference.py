"""P2 diagnostic alternatives. P1 reference.py remains frozen and unchanged.

The energy-gain rule is defined mathematically here, not copied from SPAMS code.
It is not yet established as the old ICCV MEX's exact solver or grouping policy.
"""
from dataclasses import dataclass

import numpy as np

from reference import (Pursuit, Stage, _matrix, aggregate, encode_groups, extract_patches,
                       greedy_groups, noise_budget, similarity_threshold, somp, validate_dictionary)


def simultaneous_ols(dictionary, signals, epsilon):
    """Greedy exact LS-error reduction over a shared support.

    gain(j) = ||d_j.T R||_2^2 / ||(I-P_support)d_j||_2^2.
    Unlike Tropp's L1 correlation rule, this chooses the largest actual reduction
    in group squared residual after adding an atom and refitting every column.
    """
    dictionary = validate_dictionary(dictionary)
    signals = _matrix(signals,'signals')
    if dictionary.shape[0] != signals.shape[0] or not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError('invalid signals or residual budget')
    m, atoms = dictionary.shape
    support = []
    reconstruction = np.zeros_like(signals)
    coefficients = np.empty((0,signals.shape[1]))
    residual = signals.copy()
    energy = float(np.sum(residual**2))
    history = [energy]
    tolerance = 64*np.finfo(float).eps*max(1.,energy)
    reason = 'span_exhausted'
    while energy > epsilon+tolerance and len(support) < min(m,atoms):
        if support:
            basis, _ = np.linalg.qr(dictionary[:,support],mode='reduced')
            projected = dictionary-basis@(basis.T@dictionary)
        else:
            projected = dictionary
        norm_squared = np.sum(projected**2,axis=0)
        allowed = norm_squared > 128*np.finfo(float).eps
        allowed[support] = False
        if not np.any(allowed):
            break
        scores = np.full(atoms,-np.inf)
        correlations = dictionary.T@residual
        scores[allowed] = np.sum(correlations[allowed]**2,axis=1)/norm_squared[allowed]
        atom = int(np.argmax(scores))
        if scores[atom] <= tolerance:
            break
        trial = support+[atom]
        active = dictionary[:,trial]
        fitted, _, rank, _ = np.linalg.lstsq(active,signals,rcond=None)
        if rank != len(trial):
            raise ArithmeticError('energy-gain candidate is numerically rank deficient')
        next_reconstruction = active@fitted
        next_residual = signals-next_reconstruction
        next_energy = float(np.sum(next_residual**2))
        if not np.isfinite(next_energy) or next_energy > energy+tolerance:
            raise ArithmeticError('LS refit increased residual')
        support, coefficients = trial, fitted
        reconstruction, residual, energy = next_reconstruction, next_residual, next_energy
        history.append(energy)
    if energy <= epsilon+tolerance:
        reason='noise_budget'
    return Pursuit(np.asarray(support,dtype=np.int64),coefficients,reconstruction,energy,
                   np.asarray(history),float(epsilon),reason)


def overlapping_groups(pilot_patches, grid_shape, window, threshold):
    """Skip covered seeds, but allow already-covered members in later groups.

    This is an explicit diagnostic hypothesis motivated by author log counters,
    NOT a claim that those counters alone establish author group membership.
    """
    pilot_patches = _matrix(pilot_patches,'pilot_patches')
    if (len(grid_shape)!=2 or min(grid_shape)<1 or np.prod(grid_shape)!=len(pilot_patches)
            or not isinstance(window,(int,np.integer)) or window<1
            or not np.isfinite(threshold) or threshold<0):
        raise ValueError('invalid grouping parameters')
    ny,nx=grid_shape
    covered=np.zeros(len(pilot_patches),dtype=bool)
    grid=np.arange(len(covered)).reshape(grid_shape)
    groups=[]
    for seed in range(len(covered)):
        if covered[seed]:
            continue
        y,x=divmod(seed,nx)
        candidates=grid[max(0,y-window//2):min(ny,y+(window-1)//2+1),
                        max(0,x-window//2):min(nx,x+(window-1)//2+1)].ravel()
        difference=pilot_patches[candidates]-pilot_patches[seed]
        members=candidates[np.sum(difference**2,axis=1)<=threshold]
        if seed not in members:
            raise ArithmeticError('seed absent from its own group')
        groups.append(members)
        covered[members]=True
    return groups


@dataclass
class DiagnosticResult:
    output: np.ndarray
    pilot: np.ndarray
    coverage: np.ndarray
    groups: list
    pursuits: list
    reconstructed_occurrences: np.ndarray
    occurrence_sources: np.ndarray
    positions: np.ndarray
    solver: str
    grouping: str
    threshold: float


def denoise_diagnostic(image, dictionary, sigma, *, solver='tropp_l1', grouping='disjoint',
                       threshold_multiplier=1., tau=.8, window=32):
    """Single-factor diagnostic configurations, never automatic default promotion."""
    image=_matrix(image,'image')
    dictionary=validate_dictionary(dictionary)
    block=int(np.sqrt(dictionary.shape[0]))
    if block*block!=dictionary.shape[0] or solver not in ('tropp_l1','energy_gain'):
        raise ValueError('invalid dictionary geometry or solver')
    if grouping not in ('disjoint','overlap') or not np.isfinite(threshold_multiplier) or threshold_multiplier<=0:
        raise ValueError('invalid diagnostic grouping')
    solve=somp if solver=='tropp_l1' else simultaneous_ols
    patches,positions=extract_patches(image,block)
    # For the P1 control this preserves the identical call and aggregation path.
    if solver=='tropp_l1':
        pilot_stage=encode_groups(patches,dictionary,[np.array([i]) for i in range(len(patches))],sigma,tau)
        pilot,_=aggregate(pilot_stage.patches,positions,image.shape,block)
    else:
        initial=np.empty_like(patches)
        for index,patch in enumerate(patches):
            mean=patch.mean()
            fit=solve(dictionary,(patch-mean)[:,None],noise_budget(block*block,1,sigma,tau))
            if fit.stop_reason!='noise_budget':
                raise ArithmeticError('pilot cannot meet noise constraint')
            initial[index]=fit.reconstruction[:,0]+mean
        pilot,_=aggregate(initial,positions,image.shape,block)
    matching,_=extract_patches(pilot,block)
    grid_shape=(image.shape[0]-block+1,image.shape[1]-block+1)
    threshold=similarity_threshold(sigma,block*block)*threshold_multiplier
    if grouping=='disjoint':
        groups,_=greedy_groups(matching,grid_shape,window,threshold)
    else:
        groups=overlapping_groups(matching,grid_shape,window,threshold)
    occurrences=[]
    sources=[]
    pursuits=[]
    for members in groups:
        signals=patches[members].T
        mean=signals.mean(axis=0,keepdims=True)
        fit=solve(dictionary,signals-mean,noise_budget(block*block,len(members),sigma,tau))
        if fit.stop_reason!='noise_budget':
            raise ArithmeticError('final group cannot meet noise constraint')
        occurrences.append((fit.reconstruction+mean).T)
        sources.append(members)
        pursuits.append(fit)
    occurrences=np.concatenate(occurrences)
    sources=np.concatenate(sources)
    # Restore source raster order for the P1 control's exact aggregation order.
    if grouping=='disjoint':
        order=np.argsort(sources)
        occurrences,sources=occurrences[order],sources[order]
    output,coverage=aggregate(occurrences,positions[sources],image.shape,block)
    return DiagnosticResult(output,pilot,coverage,groups,pursuits,occurrences,sources,positions,
                            solver,grouping,threshold)
