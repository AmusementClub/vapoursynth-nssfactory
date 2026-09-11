#!/usr/bin/env python3
"""Freeze pilot and matching separately for compiler-sensitive BM3D cases."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from match_replay import records,trace_delta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bm_numerics import compare
p=argparse.ArgumentParser(description=__doc__)
for k in ['baseline','candidate','crossover','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);report=dict(cases=[],numerical_admission=False)
source=json.loads((a.crossover/'summary.json').read_text())
for row in source['cases']:
 if all(v['passed'] for v in row['fixed_reference'].values()):continue
 index=row['index'];directory=a.out/f'c{index}';directory.mkdir();source_dir=a.crossover/f'c{index}';case=source_dir/'case.json';result=dict(index=index,parameters=row['parameters'],references={})
 for reference in ['baseline','candidate']:
  pilot=source_dir/(reference+'-pilot.npz');traces={};actual={};reproduced={}
  def invoke(side,label,replay=None):
   out=directory/(label+'.npz');record=directory/(label+'.bin');env=dict(os.environ,NSS_MATCH_RECORD=str(record));env.pop('NSS_MATCH_REPLAY',None)
   if replay:env['NSS_MATCH_REPLAY']=str(replay)
   cmd=[sys.executable,str(Path(__file__).with_name('public_crossover.py')),'worker','--plugin',str(getattr(a,side).resolve()),'--case',str(case),'--reference',str(pilot),'--out',str(out)]
   proc=subprocess.run(cmd,env=env,capture_output=True,text=True);out.with_suffix('.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
   meta=json.loads(Path(str(record)+'.meta.json').read_text())
   if not meta['calls'] or not meta['replay_consumed']:raise RuntimeError('missing/unconsumed matches')
   data=np.load(out);return np.stack([data[f'n{n}'] for n in range(3)]),record
  for side in ['baseline','candidate']:
   actual[side],traces[side]=invoke(side,side+'-ref-'+reference)
   expected=np.load(source_dir/(side+'-using-'+reference+'.npz'));reproduced[side]=bool(np.array_equal(actual[side],np.stack([expected[f'n{n}'] for n in range(3)])))
  if not all(reproduced.values()):raise RuntimeError('instrumentation changed fixed-pilot output')
  frozen={}
  for matching in ['baseline','candidate']:
   outputs=[invoke(side,side+'-ref-'+reference+'-match-'+matching,traces[matching])[0] for side in ['baseline','candidate']]
   frozen[matching]=compare(*outputs)
  delta=trace_delta(traces['baseline'],traces['candidate'])
  result['references'][reference]=dict(reproduced=reproduced,matching=delta,fixed=frozen)
 report['cases'].append(result);(a.out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(index=index,fixed=[[v['max_abs'] for v in item['fixed'].values()] for item in result['references'].values()])),flush=True)
report['hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.baseline,a.candidate,Path(__file__)]};report['passed']=all(v['passed'] for c in report['cases'] for r in c['references'].values() for v in r['fixed'].values())
(a.out/'summary.json').write_text(json.dumps(report,indent=2))
