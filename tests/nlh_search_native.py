#!/usr/bin/env python3
"""ctypes owner for the experiment-only preparation cache (no per-image tuning)."""
import ctypes as ct
import numpy as np


class Parameters(ct.Structure):
    _fields_=[(name,ct.c_int*2) for name in ('block','step','group','q','window')]+[
        ('basic_iterations',ct.c_int),('wiener_iterations',ct.c_int),('real_noise',ct.c_int),
        ('mix',ct.c_double),('hard_strength',ct.c_double),('wiener_scale',ct.c_double)]


class Native:
    def __init__(self,path):
        lib=self.lib=ct.CDLL(str(path))
        lib.nlh_search_error.restype=ct.c_char_p
        lib.nlh_search_parameters_size.restype=ct.c_size_t
        if lib.nlh_search_parameters_size()!=ct.sizeof(Parameters):raise RuntimeError('native parameter ABI mismatch')
        fp=ct.POINTER(ct.c_float);dp=ct.POINTER(ct.c_double)
        lib.nlh_search_prepare.argtypes=[fp,ct.c_int,ct.c_int,ct.c_int,ct.c_double,ct.c_int]
        lib.nlh_search_prepare.restype=ct.c_void_p
        lib.nlh_search_free.argtypes=[ct.c_void_p]
        lib.nlh_search_prepare_seconds.argtypes=[ct.c_void_p];lib.nlh_search_prepare_seconds.restype=ct.c_double
        lib.nlh_search_sigma.argtypes=[ct.c_void_p,dp]
        lib.nlh_search_run.argtypes=[ct.c_void_p,ct.POINTER(Parameters),fp,dp,ct.POINTER(ct.c_uint64)]
        lib.nlh_search_run.restype=ct.c_int

    def prepare(self,noisy,sigma):
        noisy=np.ascontiguousarray(noisy,dtype=np.float32)
        c,h,w=noisy.shape
        handle=self.lib.nlh_search_prepare(noisy.ctypes.data_as(ct.POINTER(ct.c_float)),w,h,c,
                                           0 if sigma is None else sigma,int(sigma is None))
        if not handle:raise RuntimeError(self.lib.nlh_search_error().decode())
        units=np.empty(c,np.float64)
        self.lib.nlh_search_sigma(handle,units.ctypes.data_as(ct.POINTER(ct.c_double)))
        return dict(handle=handle,shape=noisy.shape,sigma=units.tolist(),
                    prepare_seconds=self.lib.nlh_search_prepare_seconds(handle))

    def run(self,source,p):
        required={'block_size','block_step','group_size','q','search_window','basic_iters','wiener_iters',
                  'noise_model','lambda_basic','hard_strength','wiener_sigma_scale'}
        if set(p)!=required:raise ValueError('native search requires a complete explicit profile without sigma')
        if p['noise_model'] not in ('auto','awgn','real'):raise ValueError('invalid native noise model')
        options=Parameters()
        for field,key in (('block','block_size'),('step','block_step'),('group','group_size'),('q','q'),('window','search_window')):
            setattr(options,field,(ct.c_int*2)(*p[key]))
        options.basic_iterations=p['basic_iters'];options.wiener_iterations=p['wiener_iters']
        options.real_noise=p['noise_model']=='real' or (p['noise_model']=='auto' and source['shape'][0]==3)
        options.mix=p['lambda_basic'];options.hard_strength=p['hard_strength'];options.wiener_scale=p['wiener_sigma_scale']
        output=np.empty(source['shape'],np.float32);seconds=ct.c_double();groups=ct.c_uint64()
        code=self.lib.nlh_search_run(source['handle'],ct.byref(options),output.ctypes.data_as(ct.POINTER(ct.c_float)),
                                      ct.byref(seconds),ct.byref(groups))
        if code:raise RuntimeError(self.lib.nlh_search_error().decode())
        return output,seconds.value,groups.value

    def free(self,source):
        if source.get('handle'):
            self.lib.nlh_search_free(source['handle']);source['handle']=None
