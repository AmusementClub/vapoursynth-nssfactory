import argparse,json,subprocess,sys,os,shutil
from pathlib import Path
p=argparse.ArgumentParser()
for n in ('drivers','configs','baseline','candidate','baseline-probe','candidate-probe','out'):p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--zen',action='store_true');a=p.parse_args();sys.path.insert(0,str(a.drivers))
from avx2_review_evidence import review_phase
if a.zen:
 from avx2_host_probe import probe
 host=probe(5);selected=host['selected']
 if not selected:raise RuntimeError('host busy')
 cpu,sibling=selected['cpu'],selected['sibling_cpu']
else:cpu,sibling=0,1;host={}
a.out.mkdir(parents=True,exist_ok=False)
(a.out/'host.json').write_text(json.dumps(host,indent=2))
for side in ('baseline','candidate'):shutil.copy2(getattr(a,side),a.out/(side+'.so'))
def command(log,args):
 with log.open('w') as stream:return subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT).returncode
for run in ('main','fallback','real-images'):
 cfg=json.loads((a.configs/(run+'.json')).read_text())
 if a.zen:
  for row in cfg:
   if row.get('sample'):
    path=Path(row['sample']);row['sample']=str((a.configs/'images'/path.name) if 'images' in path.parts else Path('/tmp/nss-avx2-samples')/path.name)
 path=a.out/(run+'.json');path.write_text(json.dumps(cfg,indent=2))
 screen=a.out/run
 rc=command(a.out/(run+'.log'),[sys.executable,str(a.drivers/'c4_paired_bm.py'),'run','--baseline',str(a.baseline),'--candidate',str(a.candidate),'--configs',str(path),'--out',str(screen),'--cpu',str(cpu),'--sibling-cpu',str(sibling),'--pairs','7','--group-seconds','45','--selection-threshold','1.02','--continue-numerical-triage'])
 if rc:raise RuntimeError(f'{run} runner failed {rc}')
 for i,row in enumerate(json.loads((screen/'summary.json').read_text())):
  if row['numerical']['passed']:continue
  target=a.out/f'replay-{run}-{i}'
  if row['config'].get('stage')=='two_stage':
   args=[sys.executable,str(a.drivers/'avx2_nlh_replay.py'),'run','--baseline',str(a.baseline),'--candidate',str(a.candidate),'--screen',str(screen),'--out',str(target),'--index',str(i),'--cpu',str(cpu),'--baseline-probe',str(a.baseline_probe),'--candidate-probe',str(a.candidate_probe)]
   command(target.with_suffix('.log'),args)
 # Review utility accepts arbitrary final run names through a separate derived phase.
 from avx2_review_evidence import verified
 import hashlib,numpy as np
 reports=[(x,json.loads(x.read_text())) for x in a.out.glob('**/replay.json') if not (x.parent/'INVALID.txt').exists()]
 result=[]
 for i,row in enumerate(json.loads((screen/'summary.json').read_text())):
  matching=[]
  if not row['numerical']['passed']:
   hashes={s:hashlib.sha256(np.load(screen/f'{i}-{s}.npy').tobytes()).hexdigest() for s in ('baseline','candidate')}
   for rp,r in reports:
    if not verified(r) or r.get('original')!=row['numerical']:continue
    expected=r.get('output_hashes')
    if not expected:
     expected={s:hashlib.sha256(np.load(rp.parent/f'{s}-using-{s}.npy').tobytes()).hexdigest() for s in hashes}
    if hashes==expected:matching.append(str(rp))
  result.append(dict(name=row['config']['name'],speedup=row['paired_speedup'],ci95=row['ci95'],pairs=row['pairs'],numerical_passed=row['numerical']['passed'] or bool(matching),environment_valid=row['environment']['valid'],replays=matching))
 (screen/'reviewed.json').write_text(json.dumps(result,indent=2))
(a.out/'status').write_text('completed\n')
