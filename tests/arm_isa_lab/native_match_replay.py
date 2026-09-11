#!/usr/bin/env python3
"""Full-filter fixed matcher replay of an actual retained 1080p performance case."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from match_replay import records
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bm_numerics import compare
p=argparse.ArgumentParser(description=__doc__)
for key in ['baseline','candidate','pairs','out']:p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--name',default='mcwnnm_1080');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
summary=json.loads((a.pairs/'summary.json').read_text());original=next(r for r in summary['results'] if r['config']['name']==a.name)
config={k:v for k,v in original['config'].items() if not k.startswith('_')};config.update(frames=1,warmup=0)
plugins={'baseline':a.baseline,'candidate':a.candidate}
report=dict(numerical_admission=False,input_hashing='Whole-frame input hash recorded by worker; repeated per-query hash explicitly disabled.',config=config,reproduced={},fixed={},hashes={})
def invoke(side,label,replay=None):
 output=a.out/(label+'.npy');record=a.out/(label+'.bin')
 env=dict(os.environ,NSS_MATCH_RECORD=str(record),NSS_MATCH_OMIT_INPUT_HASH='1');env.pop('NSS_MATCH_REPLAY',None)
 if replay:env['NSS_MATCH_REPLAY']=str(replay)
 command=[sys.executable,str(Path(__file__).resolve().parents[1]/'c4_integration.py'),'worker','--plugin',str(plugins[side].resolve()),'--config',json.dumps(dict(config,_dump=str(output)))]
 proc=subprocess.run(command,env=env,capture_output=True,text=True);(a.out/(label+'.log')).write_text(proc.stdout+proc.stderr);proc.check_returncode()
 metadata=json.loads(proc.stdout);trace=json.loads(Path(str(record)+'.meta.json').read_text())
 if not trace['replay_consumed'] or trace['calls']<=0:raise RuntimeError('empty/unconsumed trace')
 report['hashes'][str(record)]=hashlib.sha256(record.read_bytes()).hexdigest()
 return np.load(output),record,metadata
values={};traces={};inputs=[]
for side in plugins:
 values[side],traces[side],meta=invoke(side,side);inputs.append(meta['input_sha256'])
 expected=np.load(a.pairs/a.name/(side+'.npy'))[:1]
 report['reproduced'][side]=bool(np.array_equal(values[side],expected))
 if not report['reproduced'][side]:raise RuntimeError('instrumented pixels differ from performance capture')
if len(set(inputs))!=1:raise RuntimeError('input payload mismatch')
report['input_sha256']=inputs[0];report['original']=compare(values['baseline'],values['candidate'])
left,right=[records(traces[s]) for s in plugins]
if len(left)!=len(right):raise RuntimeError('trace length mismatch')
changes=[]
for i,((ha,_,pa),(hb,_,pb)) in enumerate(zip(left,right)):
 if ha[:7]!=hb[:7]:raise RuntimeError('query geometry mismatch')
 if ha[7:]!=hb[7:] or not np.array_equal(pa.reshape(-1,5)[:,:4],pb.reshape(-1,5)[:,:4]):changes.append(i)
report['trace']=dict(calls=len(left),changed_identity_calls=changes,first_change=min(changes) if changes else None)
for reference in plugins:
 out=[invoke(side,side+'-using-'+reference,traces[reference])[0] for side in plugins]
 report['fixed'][reference]=compare(*out)
report['hashes'].update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*plugins.values(),Path(__file__)]})
(a.out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(reproduced=report['reproduced'],original=report['original']['max_abs'],calls=len(left),changed=len(changes),first_change=report['trace']['first_change'],fixed=[v['max_abs'] for v in report['fixed'].values()])))
