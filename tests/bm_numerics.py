"""Numerical triage. Above-threshold output is never admitted by PSNR alone."""
import numpy as np

def psnr(a,b):
    mse=float(np.mean((np.asarray(a,dtype=np.float64)-b)**2))
    return float('inf') if mse==0 else -10*np.log10(mse)

def ssim(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    def box(v):
        k=min(7,v.shape[-2],v.shape[-1]);s=np.pad(v,[(0,0)]*(v.ndim-2)+[(1,0),(1,0)]).cumsum(-1).cumsum(-2)
        return (s[...,k:,k:]-s[...,:-k,k:]-s[...,k:,:-k]+s[...,:-k,:-k])/(k*k)
    ma=box(a);mb=box(b);va=np.maximum(0,box(a*a)-ma*ma);vb=np.maximum(0,box(b*b)-mb*mb);cov=box(a*b)-ma*mb
    return float(np.mean((2*ma*mb+.01**2)*(2*cov+.03**2)/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))))

def compare(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape:return dict(passed=False,reason='shape_mismatch',baseline_shape=a.shape,candidate_shape=b.shape)
    if not np.isfinite(a).all() or not np.isfinite(b).all():return dict(passed=False,reason='nonfinite')
    e=np.abs(a-b);i=np.unravel_index(int(e.argmax()),e.shape);rmse=float(np.sqrt(np.mean(e*e)));maximum=float(e[i]);passed=maximum<=1e-5 and rmse<=1e-6
    return dict(passed=passed,reason='within_triage_band' if passed else 'requires_stage_replay',max_abs=maximum,rmse=rmse,
                max_index=list(map(int,i)),baseline_at_max=float(a[i]),candidate_at_max=float(b[i]),
                differing=int(np.count_nonzero(e)),psnr_between=psnr(a,b),ssim_between=ssim(a,b))
