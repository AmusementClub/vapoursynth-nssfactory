#!/usr/bin/env python3
"""Static project ISA inventory; runtime selection and samples are separate evidence."""
import argparse,hashlib,json,platform,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plugin',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
cmd=['otool','-tvV',str(a.plugin)] if platform.system()=='Darwin' else ['objdump','-d','-C',str(a.plugin)]
proc=subprocess.run(cmd,capture_output=True,text=True);proc.check_returncode();(a.out/'plugin.asm').write_text(proc.stdout)
counts={'neon_float':0,'sve':0,'sme':0};examples={k:[] for k in counts}
function='';detection_only=[];unexpected_sve=[]
for line in proc.stdout.splitlines():
 label=re.match(r'^\s*[0-9a-f]+ <(.+)>:$',line)
 if label:function=label[1];continue
 m=re.match(r'^\s*[0-9a-f]+:?\s+(?:[0-9a-f]{8}\s+)?([a-z][a-z0-9.]*)\s*(.*)',line)
 if not m:continue
 op,operands=m.groups();kinds=[]
 if op.startswith('f') and (re.search(r'\bv\d+\.(?:4s|2s|2d)',operands) or re.search(r'\.(?:4s|2s|2d)$',op)):kinds.append('neon_float')
 if re.search(r'\bz\d+\b|\bp\d+(?:\.[bhsd]|/[zm])',operands) or op in ('ptrue','whilelt','whilelo','rdvl','addvl','addpl','cntb','cnth','cntw','cntd'):kinds.append('sve')
 if re.search(r'\bza\b|\bza\d+',operands) or op in ('smstart','smstop','rdsvl'):kinds.append('sme')
 for k in kinds:
  counts[k]+=1
  if len(examples[k])<12:examples[k].append(line)
 if 'sve' in kinds:
  record=dict(function=function,instruction=line)
  # Highway 1.4.0 targets.cc calls this vector-length probe only after the
  # OS HWCAP check reports SVE. It is not a denoising kernel or an SVE route.
  if function=='hwy::arm::DetectAdditionalSveTargets(long)' and op=='cntb':detection_only.append(record)
  else:unexpected_sve.append(record)
report=dict(passed=counts['neon_float']>0 and not unexpected_sve and counts['sme']==0,plugin_sha256=hashlib.sha256(a.plugin.read_bytes()).hexdigest(),command=cmd,counts=counts,examples=examples,
 detection_only_sve=detection_only,unexpected_sve=unexpected_sve,
 scope='Project denoising kernels exclude SVE/SME. A statically linked Highway HWCAP-guarded CNTB detection probe is listed separately; dependency libraries may select their own ISA. Native Backend() and frame-boundary PMU samples establish executed routes.')
(a.out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='examples'}))
