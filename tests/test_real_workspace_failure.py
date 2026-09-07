#!/usr/bin/env python3
"""Real VS failure translation, using the narrowly scoped Linux interposer."""
import ctypes
import gc
import json
import os
import vapoursynth as vs
from test_plan01_plugin import clip, evaluate, options
core = vs.core
core.num_threads = 8
core.std.LoadPlugin(path=os.environ['NSS_SO'])
process = ctypes.CDLL(None)
process.nss_test_arm_workspace_failure.argtypes = [ctypes.c_long]
process.nss_test_arm_workspace_failure.restype = None
process.nss_test_workspace_injections.restype = ctypes.c_long
rows = []
for name in ('BM3D','WNNM','TWSC','MCWNNM','NCSR','NLH','LSSC','NLM'):
    source=clip(core,width=64,height=48,frames=5,fmt=vs.RGBS if name=='MCWNNM' else vs.GRAYS)
    kw=options(name,radius=0 if name=='LSSC' else 1)
    for repeat in range(3):
        node=getattr(core.nss,name)(source,**kw)
        process.nss_test_arm_workspace_failure(0)
        try:
            node.get_frame(2)
        except vs.Error as error:
            if not any(message in str(error) for message in
                       ('resource allocation failed', 'workspace allocation failed')):
                raise
        else:raise AssertionError(f'{name}: workspace ENOMEM escaped rejection')
        if process.nss_test_workspace_injections()!=1:
            raise AssertionError(f'{name}: wrong allocator interception')
        process.nss_test_arm_workspace_failure(-1)
        del node
        gc.collect()
        node=getattr(core.nss,name)(source,**kw)
        evaluate(node,2)
        del node
        rows.append(dict(algorithm=name,repeat=repeat,rejected=True,recovered=True))
print(json.dumps(dict(passed=True,cases=rows)))
