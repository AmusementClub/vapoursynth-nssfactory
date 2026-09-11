#!/usr/bin/env python3
"""Count actual GEMM shapes; output pixels must equal the uninstrumented worker."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import time
import types
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import c4_integration as integration
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plugin',required=True)
parser.add_argument('--config',required=True)
parser.add_argument('--out',required=True)
args=parser.parse_args()
process=ctypes.CDLL(None)
process.nss_gemm_trace_enable.argtypes=[ctypes.c_int]
process.nss_gemm_trace_save.argtypes=[ctypes.c_char_p]
calls=0
def boundary():
    global calls
    calls+=1
    process.nss_gemm_trace_enable(int(calls==1))
    return time.perf_counter()
integration.time=types.SimpleNamespace(perf_counter=boundary)
result=integration.worker(args.plugin,json.loads(args.config))
assert calls==2
process.nss_gemm_trace_save(args.out.encode())
result['timing_is_instrumented']=True
print(json.dumps(result))
