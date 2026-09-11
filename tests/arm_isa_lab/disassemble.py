#!/usr/bin/env python3
"""Static ISA inventory and sampled hot-symbol annotations; not timing proof."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
builds={'generic':'/tmp/nss-sve-generic-build','neon_tuned':'/tmp/nss-sve-neon-tuned-build',
        'sve_auto':'/tmp/nss-sve-auto-build','sve_fixed128':'/tmp/nss-sve-fixed128-build'}
report={'kind':'static instruction inventory plus sampled annotations','variants':{},'annotations':[]}
for name,build in builds.items():
    binary=Path(build)/'libnss.so'
    text=subprocess.check_output(['objdump','-d','-C',str(binary)],text=True)
    (args.out/(name+'.asm')).write_text(text)
    symbols={};symbol='unknown'
    for line in text.splitlines():
        m=re.match(r'^[0-9a-f]+ <(.*)>:',line)
        if m:symbol=m[1];continue
        m=re.match(r'^\s*[0-9a-f]+:\s+[0-9a-f]+\s+(\S+)\s*(.*)',line)
        if not m:continue
        op,operands=m.groups()
        sve=bool(re.search(r'\bz\d+\.[bhsdq]|\bp\d+(?:\.[bhsd]|/[mz])',operands)) or op in ('cntb','cnth','cntw','cntd','rdvl','addvl','addpl')
        if sve:symbols[symbol]=symbols.get(symbol,0)+1
    report['variants'][name]={'binary':str(binary),'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
                              'sve_static_instructions':sum(symbols.values()),'symbols':symbols}
for case in ('bm3d_1080','lssc_1080','wnnm_1080','mcwnnm_1080','nlh_1080'):
    directory=Path('/tmp/nss-c4a-profile-baseline')/case
    choices=[]
    for line in (directory/'hot.txt').read_text().splitlines():
        m=re.match(r'\s*([0-9.]+)%\s+\[.\]\s+(.*?)\s{2,}(\S+)',line)
        if m and m[3]=='libnss.so':choices.append((float(m[1]),m[2]))
    if not choices:raise RuntimeError(f'no plugin samples: {case}')
    percent,symbol=max(choices)
    command=['perf','annotate','--stdio','--percent-type','local-period','--symbol',symbol,'-i',str(directory/'perf.data')]
    proc=subprocess.run(command,capture_output=True,text=True,timeout=60)
    (args.out/(case+'-annotate.txt')).write_text(proc.stdout+proc.stderr)
    report['annotations'].append({'case':case,'symbol':symbol,'sample_percent':percent,'command':command,'returncode':proc.returncode})
(args.out/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
