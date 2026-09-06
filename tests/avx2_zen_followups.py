#!/usr/bin/env python3
"""Serial causal review and two-real-image corroboration on the physical AVX2 host."""
import argparse,json,os,subprocess,sys,hashlib
from pathlib import Path
from avx2_host_probe import probe
from avx2_port_followup import followup
from avx2_review_evidence import review_phase

def run(a):
    here=Path(__file__).resolve().parent
    base=a.root/'baseline.so'
    base.write_bytes(a.baseline.read_bytes())
    def command(name,cmd):
        with name.open('w') as stream:
            return subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT).returncode
    def replay(phase,screen,mask,cpu):
        p=screen/'summary.json'
        if not p.exists():return
        for index,row in enumerate(json.loads(p.read_text())):
            if row['numerical']['passed']:continue
            out=phase/f'replay-{screen.name}-{index}'
            if out.exists():continue
            common=['--baseline',str(base),'--candidate',str(phase/'candidate.so'),'--screen',str(screen),'--out',str(out),'--index',str(index),'--cpu',str(cpu)]
            if row['config']['algorithm']=='bm3d' and row['config']['stage']=='basic':
                common=['--baseline',str(a.probes/'baseline-probe'),'--candidate',str(a.probes/'candidate-probe'),*common[4:]]
                cmd=[sys.executable,str(here/'avx2_dct_replay.py'),*common]
            elif row['config']['stage']=='two_stage':
                cmd=[sys.executable,str(here/'avx2_nlh_replay.py'),'run',*common]
                if mask==128:cmd+=['--baseline-probe',str(a.probes/'baseline-probe'),'--candidate-probe',str(a.probes/'candidate-probe')]
            else:continue
            command(out.with_suffix('.log'),cmd)
    for mask in a.masks:
        phase=a.root/f'mask-{mask}'
        if not (phase/'bench/summary.json').exists():continue
        (phase/'candidate.so').write_bytes((a.binaries/f'nss-avx2-mask{mask}.so').read_bytes())
        host=probe(5);(phase/'followup-host.json').write_text(json.dumps(host,indent=2))
        selected=host['selected']
        if not selected:
            (a.root/'followup-status').write_text(f'host_busy_before_{mask}\n');return 2
        cpu,sibling=selected['cpu'],selected['sibling_cpu']
        replay(phase,phase/'bench',mask,cpu)
        reviewed=review_phase(phase)
        if not reviewed['bench']['selected']:continue
        configs=followup(phase/'bench',a.samples,phase/'followup-images')
        if not configs:continue
        p=phase/'followup-configs.json';p.write_text(json.dumps(configs,indent=2))
        cmd=[sys.executable,str(here/'c4_paired_bm.py'),'run','--baseline',str(base),'--candidate',str(phase/'candidate.so'),
             '--configs',str(p),'--out',str(phase/'followup'),'--cpu',str(cpu),'--sibling-cpu',str(sibling),
             '--pairs','7','--group-seconds','45','--selection-threshold','1.02','--continue-numerical-triage']
        rc=command(phase/'followup.log',cmd)
        if rc:
            (phase/'followup-status').write_text(f'runner_failed_{rc}\n');continue
        replay(phase,phase/'followup',mask,cpu)
        review_phase(phase)
    (a.root/'followup-status').write_text('completed\n');return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--probes',type=Path,required=True);p.add_argument('--binaries',type=Path,default=Path('/tmp'))
    p.add_argument('--samples',type=Path,default=Path('/tmp/nss-avx2-samples'));p.add_argument('--masks',type=int,nargs='+',default=[1,4,8,16,32,512,1024,64,96,128]);sys.exit(run(p.parse_args()))
