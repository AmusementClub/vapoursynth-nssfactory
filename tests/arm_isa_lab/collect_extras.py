#!/usr/bin/env python3
"""Opcode PMU controls and non-timed GEMM shape tracing on native C4A."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from c4a_profile import event,parse_stat

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plugin',type=Path,required=True)
parser.add_argument('--profile',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=False)
source=Path(__file__).resolve().parent
report=dict(passed=False,opcode_controls={},shapes=[])
def command(name,argv,env=None):
    p=subprocess.run(argv,capture_output=True,text=True,env=env,timeout=180)
    (args.out/(name+'.log')).write_text(p.stdout+p.stderr)
    (args.out/(name+'.command.json')).write_text(json.dumps(argv,indent=2))
    p.check_returncode()
    return p
try:
    command('build-counters',['g++-15','-std=c++20','-O2',str(source/'counter_probe.cpp'),str(source/'instruction_loops.S'),'-o',str(args.out/'counter-probe')])
    command('build-tracer',['g++-15','-std=c++20','-O2','-fPIC','-shared',str(source/'gemm_trace.cpp'),'-ldl','-pthread','-o',str(args.out/'gemm-trace.so')])
    sme=subprocess.run(['g++-15','-std=c++20','-O3','-march=armv9.2-a+sme','-c',str(source/'sme_gemm.cpp'),'-o',str(args.out/'sme-gemm-compile.o')],capture_output=True,text=True)
    (args.out/'sme-cxx-compile.log').write_text(sme.stdout+sme.stderr)
    report['sme_cpp_compile_returncode']=sme.returncode
    names=('cycles','instructions','neon_spec','sve_spec','fp32_spec')
    for isa in ('neon','sve'):
        path=args.out/(isa+'-counter.csv')
        command(isa+'-counter',['taskset','-c','0','perf','stat','-x',';','--no-big-num','-e','{'+','.join(event(n) for n in names)+'}','-o',str(path),'--',str(args.out/'counter-probe'),isa])
        report['opcode_controls'][isa]=parse_stat(path,names)
    # Each assembly loop contains exactly four vector FMLAs per iteration.
    # Dynamic loader/runtime instructions are also counted in these controls.
    for isa,other in [('neon','sve'),('sve','neon')]:
        counts=report['opcode_controls'][isa]
        if counts[isa+'_spec']['count']<40000000 or counts[other+'_spec']['count']>100000:
            raise RuntimeError('opcode counter control contradicts the instruction stream')
    profile=json.loads((args.profile/'summary.json').read_text())
    assert profile['passed']
    for row in profile['results']:
        if row['config']['name'] not in ('lssc_1080','wnnm_1080','twsc_1080','ncsr_1080','mcwnnm_1080'):
            continue
        config=dict(row['config'],frames=1)
        name=config['name'];path=args.out/(name+'-shapes.csv')
        env=dict(os.environ,LD_PRELOAD=str(args.out/'gemm-trace.so'),NSS_GEMM_TRACE_PLUGIN=str(args.plugin))
        result=command(name+'-trace',['taskset','-c','0',sys.executable,str(source/'trace_worker.py'),'--plugin',str(args.plugin),'--config',json.dumps(config),'--out',str(path)],env)
        value=json.loads(result.stdout)
        if value['sha256']!=row['calibration']['sha256'] or value['input_sha256']!=row['calibration']['input_sha256']:
            raise RuntimeError('tracing changed pixels or inputs')
        shapes=list(csv.DictReader(path.open()))
        if name=='lssc_1080' and not shapes:raise RuntimeError('GEMM interception was not reached')
        report['shapes'].append(dict(name=name,output_exact=True,shapes=shapes,timing_is_instrumented=True))
    report['passed']=True
finally:
    (args.out/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(passed=report['passed'],opcode_controls=report['opcode_controls'],shape_workloads=len(report['shapes']))))
