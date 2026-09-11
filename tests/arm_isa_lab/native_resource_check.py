#!/usr/bin/env python3
"""Native repeated/random/concurrent rolling requests and bounded RSS evidence."""
import argparse,gc,hashlib,json,os,platform,random,resource,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import vapoursynth as vs
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_plan01_plugin import clip,evaluate
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--plugin',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
core=vs.core;core.max_cache_size=64;core.std.LoadPlugin(path=str(a.plugin.resolve()))
def rss():
 if platform.system()=='Linux':
  return int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmRSS:')))*1024
 return int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())],text=True).strip())*1024
report=dict(passed=False,plugin_sha256=hashlib.sha256(a.plugin.read_bytes()).hexdigest(),platform=platform.platform(),tests=[])
expected={}
for workers in [1,2,4,8]:
 core.num_threads=workers
 source=clip(core,width=320,height=180,frames=24)
 node=core.nss.BM3D(source,sigma=3,block_size=8,group_size=8,block_step=8,bm_range=7,radius=1,temporal_mode='rolling',rolling_chunk=2,rolling_cache_limit=1,memory_limit_mb=32)
 core.std.SetVideoCache(node,mode=0)
 observations=[]
 def digest(n):
  value=np.array(node.get_frame(n)[0]);assert np.isfinite(value).all()
  return n,hashlib.sha256(value.tobytes()).hexdigest()
 for epoch in range(4):
  order=list(range(24))
  if epoch==1:order.reverse()
  elif epoch>=2:random.Random(120+epoch).shuffle(order)
  before=time.perf_counter()
  with ThreadPoolExecutor(max_workers=workers) as pool:values=list(pool.map(digest,order))
  for n,h in values:
   if n in expected and expected[n]!=h:raise RuntimeError('frame changed across schedules/workers')
   expected[n]=h
  observations.append(dict(epoch=epoch,rss_bytes=rss(),diagnostic_seconds=time.perf_counter()-before))
 growth=observations[-1]['rss_bytes']-observations[-2]['rss_bytes']
 if growth>16*1024*1024:raise RuntimeError('late repeated-request RSS growth exceeds bounded observation guard')
 report['tests'].append(dict(workers=workers,requests=96,node_budget_bytes=32*1024*1024,observations=observations,late_rss_growth_bytes=growth,all_revisits_exact=True))
 del node,source;gc.collect()
# Separate failure/recovery check on a real installed filter.
source=clip(core,width=640,height=360,frames=5)
kw=dict(sigma=3,block_size=8,group_size=8,block_step=8,bm_range=7,radius=1,temporal_mode='rolling',rolling_chunk=2,rolling_cache_limit=1)
try:core.nss.BM3D(source,**kw,memory_limit_mb=1).get_frame(2)
except vs.Error as error:
 if 'memory_limit_mb' not in str(error):raise
else:raise RuntimeError('budget failure silently succeeded')
evaluate(core.nss.BM3D(source,**kw,memory_limit_mb=64),2)
report.update(passed=True,budget_failure_and_recovery=True,peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='RSS includes Python/VS/allocator retention; 32MiB node ownership is enforced by plugin budget; no leak-free claim beyond this workload.')
(a.out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='tests'}))
