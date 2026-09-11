#!/usr/bin/env python3
"""Fixed-pilot replay on exact saved BM3D full-plugin performance cases."""
import argparse,hashlib,json,os,random,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bm_numerics import compare

def worker(a):
 import vapoursynth as vs
 c=json.loads(a.config.read_text());core=vs.core;core.num_threads=1;core.max_cache_size=64
 core.std.LoadPlugin(path=str(a.plugin.resolve()))
 kw=c['kwargs'];radius=kw.get('radius',0);chunk=kw.get('rolling_chunk',4) if kw.get('temporal_mode')=='rolling' else 0
 first=c.get('warmup',1)+4*radius;frames=c['frames']
 if chunk:first=(first+chunk-1)//chunk*chunk;frames=(max(frames,chunk)+chunk-1)//chunk*chunk
 length=first+frames+4*radius+1;w,h=c['size']
 raw=np.fromfile(c['sample'],np.uint8).reshape(1080,1920);base=raw[np.arange(h)*1080//h][:,np.arange(w)*1920//w].astype(np.float32)/np.float32(255)
 def payload(n):
  clean=base[:,np.clip(np.arange(w)-(n%7-3)*2,0,w-1)] if c.get('motion') else base
  rng=np.random.RandomState(42+n if c.get('motion') else 42)
  return clean+rng.randn(h,w).astype(np.float32)*np.float32(3/255)
 blank=core.std.BlankClip(width=w,height=h,length=length,format=vs.GRAYS)
 def fill(n,f):
  out=f.copy();np.asarray(out[0])[:]=payload(n);return out
 source=core.std.ModifyFrame(blank,blank,fill)
 def filtered(ref=None):
  node=core.nss.BM3D(source,**kw,**({'ref':ref} if ref is not None else {}))
  return core.nss.VAggregate(node,source,radius=radius) if radius and not chunk else node
 if a.reference:
  data=np.load(a.reference)
  if data.shape!=(length,1,h,w):raise RuntimeError('reference dimensions differ')
  def ref_fill(n,f):
   out=f.copy();np.asarray(out[0])[:]=data[n,0];return out
  reference=core.std.ModifyFrame(blank,blank,ref_fill);node=filtered(reference)
 elif a.pilot:node=filtered()
 else:node=filtered(filtered()) if c.get('stage')=='two_stage' else filtered()
 order=list(range(length)) if a.pilot else list(range(first,first+frames))
 if not a.pilot and c.get('access')=='random':random.Random(42).shuffle(order)
 if not a.pilot:order=order[:7 if w*h<1920*1080 else 2]
 values=np.stack([np.array(node.get_frame(n)[0])[None] for n in order])
 if not np.isfinite(values).all():raise RuntimeError('nonfinite replay output')
 np.save(a.out,values)

def run(a):
 a.out.mkdir(parents=True,exist_ok=False);d=json.loads((a.pairs/'summary.json').read_text());report=dict(numerical_admission=False,cases=[])
 for row in d['results']:
  config=row['config'];name=config['name']
  if config['algorithm']!='bm3d' or (config.get('stage')!='two_stage' and not config['kwargs'].get('radius')):continue
  directory=a.out/name;directory.mkdir();casepath=directory/'config.json';casepath.write_text(json.dumps(config))
  def invoke(side,label,pilot=False,reference=None):
   plugin=getattr(a,side);out=directory/(label+'.npy')
   cmd=[sys.executable,__file__,'worker','--plugin',str(plugin.resolve()),'--config',str(casepath),'--out',str(out)]
   if pilot:cmd.append('--pilot')
   if reference:cmd+=['--reference',str(reference)]
   proc=subprocess.run(cmd,capture_output=True,text=True);out.with_suffix('.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
   return np.load(out)
  original={side:invoke(side,side) for side in ['baseline','candidate']}
  reproduced={side:bool(np.array_equal(value,np.load(a.pairs/name/(side+'.npy')))) for side,value in original.items()}
  if not all(reproduced.values()):raise RuntimeError('replay does not reproduce saved performance pixels: '+name)
  pilots={side:invoke(side,side+'-pilot',pilot=True) for side in original}
  fixed={}
  for reference in original:
   values=[invoke(side,side+'-using-'+reference,reference=directory/(reference+'-pilot.npy')) for side in original]
   fixed[reference]=compare(*values)
  result=dict(name=name,reproduced=reproduced,original=compare(*original.values()),pilot=compare(*pilots.values()),fixed=fixed)
  report['cases'].append(result);(a.out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(name=name,reproduced=reproduced,original=result['original']['max_abs'],pilot=result['pilot']['max_abs'],fixed=[x['max_abs'] for x in fixed.values()])),flush=True)
 report['hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.baseline,a.candidate,Path(__file__)]}
 (a.out/'summary.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 w=sub.add_parser('worker')
 for k in ['plugin','config','out']:w.add_argument('--'+k,type=Path,required=True)
 w.add_argument('--pilot',action='store_true');w.add_argument('--reference',type=Path)
 r=sub.add_parser('run')
 for k in ['baseline','candidate','pairs','out']:r.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();worker(a) if a.command=='worker' else run(a)
