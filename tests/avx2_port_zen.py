#!/usr/bin/env python3
"""Portable C4 AVX2 artifacts rechecked on an idle physical AVX2 host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from avx2_host_probe import probe

def run(a):
    out=a.out;out.mkdir(parents=True,exist_ok=False)
    host=probe(5)
    (out/'host-idle.json').write_text(json.dumps(host,indent=2))
    selected=host['selected']
    if not selected:
        (out/'status').write_text('host_busy\n');return 2
    cpu,sibling=selected['cpu'],selected['sibling_cpu']
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.baseline,a.candidate,a.configs,Path(__file__))}
    (out/'inputs.json').write_text(json.dumps(manifest,indent=2))
    here=Path(__file__).resolve().parent
    def command(name,args,env=None):
        with (out/(name+'.log')).open('w') as stream:
            rc=subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,env=env).returncode
        if rc:
            (out/'status').write_text(name+'_failed\n');return False
        return True
    if not command('numerics',[sys.executable,str(here/'c4_bm_numeric_matrix.py'),'--baseline',str(a.baseline),'--candidate',str(a.candidate),'--out',str(out/'numerics'),'--cpu',str(cpu)]):return 1
    env=dict(os.environ,NSS_SO=str(a.candidate))
    if not command('semantics',['taskset','-c',str(cpu),sys.executable,str(here/'test_bm3d_semantics.py')],env):return 1
    rows=json.loads(a.configs.read_text())
    small=json.loads(a.configs.read_text())
    for row in small:
        row.pop('sample',None);row['size']=[75,49];row['frames']=2
        if 'kwargs' in row:row['kwargs']['bm_range']=3
    small_path=out/'extra-configs.json';small_path.write_text(json.dumps(small,indent=2))
    if not command('extra-numerics',[sys.executable,str(here/'c4_bm_numeric_matrix.py'),'--baseline',str(a.baseline),
        '--candidate',str(a.candidate),'--configs',str(small_path),'--out',str(out/'extra-numerics'),'--cpu',str(cpu)]):return 1
    # Preserve data bytes and shape; translate only host-local sample paths.
    for row in rows:
        if row.get('sample'):
            src=Path(row['sample'])
            row['sample']=str(a.samples/src.name)
    configs=out/'configs.json';configs.write_text(json.dumps(rows,indent=2))
    if not command('bench',[sys.executable,str(here/'c4_paired_bm.py'),'run','--baseline',str(a.baseline),
        '--candidate',str(a.candidate),'--configs',str(configs),'--out',str(out/'bench'),
        '--cpu',str(cpu),'--sibling-cpu',str(sibling),'--pairs','7','--group-seconds','45','--selection-threshold','1.02'] + (['--continue-numerical-triage'] if a.continue_numerical_triage else [])):return 1
    (out/'status').write_text('completed\n');return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True,type=Path);p.add_argument('--candidate',required=True,type=Path)
    p.add_argument('--configs',required=True,type=Path);p.add_argument('--samples',type=Path,default=Path('/tmp/nss-avx2-samples'))
    p.add_argument('--out',required=True,type=Path);p.add_argument('--continue-numerical-triage',action='store_true');sys.exit(run(p.parse_args()))
