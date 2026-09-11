#!/usr/bin/env python3
"""GCP-only TWSC Top-down/symbol diagnostics, gated at frame requests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def run(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    worker=Path(__file__).with_name('twsc_speed.py')
    rows=[]
    for name in ('TopdownL1','TopdownL2','record'):
        root=out/name;root.mkdir();ctl=root/'control.fifo';ack=root/'ack.fifo'
        os.mkfifo(ctl);os.mkfifo(ack)
        control=['--delay=-1',f'--control=fifo:{ctl},{ack}']
        command=[sys.executable,str(worker),'profile','--plugin',str(Path(args.plugin).resolve()),
                 '--input',str(Path(args.input).resolve()),'--params',args.params,
                 '--control',str(ctl),'--ack',str(ack),'--output',str(root/'output.npy')]
        if name=='record':
            perf=['perf','record','-q',*control,'-e','cycles:u','-F','499','--call-graph','dwarf,8192','-o',str(root/'perf.data')]
        else:perf=['perf','stat',*control,'--no-big-num','-M',name,'-o',str(root/'metrics.txt')]
        result=subprocess.run([*perf,'--','taskset','-c','0',*command],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                              env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        (root/'stdout.log').write_text(result.stdout);(root/'stderr.log').write_text(result.stderr)
        row=dict(kind=name,returncode=result.returncode,command=[*perf,'--','taskset','-c','0',*command]);rows.append(row)
        if result.returncode:raise RuntimeError(f'{name} failed: {result.stderr[-1000:]}')
        row['worker']=json.loads(result.stdout.splitlines()[-1])
        if name=='record':
            for suffix,extra in [('symbols',['--no-children','--sort','dso,symbol']),('header',['--header-only'])]:
                text=subprocess.check_output(['perf','report','--stdio','-i',str(root/'perf.data'),*extra],text=True,stderr=subprocess.STDOUT)
                (root/(suffix+'.txt')).write_text(text)
        ctl.unlink();ack.unlink();print(name,'complete',flush=True)
    (out/'summary.json').write_text(json.dumps(dict(scope='GCP diagnostics only; no acceptance timing or algorithm-quality gate',
        rows=rows,worker_source_sha256=hashlib.sha256(worker.read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plugin','input','params','out'):parser.add_argument('--'+key,required=True)
    run(parser.parse_args())
