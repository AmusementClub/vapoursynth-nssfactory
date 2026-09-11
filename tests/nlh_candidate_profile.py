#!/usr/bin/env python3
"""Warm full-pipeline NLH timing/perf gate on an already frozen float input."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import vapoursynth as vs

from nlh_defaults_search import cpu_ticks,cpu_activity
from nlh_defaults_inputs import file_sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plugin','input','parameters-json'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--fixture-meta');parser.add_argument('--frames',type=int,default=3)
    parser.add_argument('--control');parser.add_argument('--ack');parser.add_argument('--output')
    args=parser.parse_args()
    if os.sched_getaffinity(0)!={0}:raise ValueError('run under taskset -c0')
    if args.frames<1 or bool(args.control)!=bool(args.ack):raise ValueError('invalid frame/control configuration')
    loaded=np.load(args.input)
    noisy=loaded['noisy'] if isinstance(loaded,np.lib.npyio.NpzFile) else loaded
    noisy=np.ascontiguousarray(noisy,dtype=np.float32)
    if noisy.ndim!=3 or len(noisy) not in (1,3) or not np.isfinite(noisy).all():
        raise ValueError('input must be finite C,H,W float Gray/RGB')
    c,h,w=noisy.shape;core=vs.core;core.num_threads=1
    core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
    blank=core.std.BlankClip(width=w,height=h,length=6,format=vs.RGBS if c==3 else vs.GRAYS)
    fills=[0]
    def fill(n,f):
        fills[0]+=1;frame=f.copy()
        for p in range(c):np.copyto(np.asarray(frame[p]),noisy[p])
        return frame
    cached=core.std.ModifyFrame(blank,blank,fill)
    core.std.SetVideoCache(cached,mode=1,fixedsize=6,maxsize=6)
    held=[cached.get_frame(n) for n in range(6)]
    source=core.std.Loop(cached,times=args.frames+6)
    parameters=json.loads(args.parameters_json)
    node=core.nss.NLH(source,**parameters)
    if parameters.get('radius',0):node=core.nss.VAggregate(node,source,radius=parameters['radius'])
    core.std.SetVideoCache(node,mode=0)
    started=time.perf_counter();warm=node.get_frame(1);warm_seconds=time.perf_counter()-started
    print(json.dumps(dict(event='warmup',seconds=warm_seconds)),flush=True)
    ctl=os.open(args.control,os.O_WRONLY) if args.control else None
    ack=open(args.ack) if args.ack else None
    acknowledgements=[]
    def control(value):
        if ctl is not None:
            os.write(ctl,(value+'\n').encode());reply=ack.readline();acknowledgements.append(repr(reply))
            if reply.strip('\x00\r\n ')!='ack':raise RuntimeError('unexpected perf acknowledgement: '+repr(reply))
    before_fills=fills[0];before=cpu_ticks();control('enable');started=time.perf_counter()
    for n in range(3,3+args.frames):frame=node.get_frame(n)
    elapsed=time.perf_counter()-started;control('disable');after=cpu_ticks()
    pixels=np.stack([np.asarray(frame[p]).copy() for p in range(c)])
    if fills[0]!=before_fills or not np.isfinite(pixels).all():
        raise AssertionError('source evaluation inside timing or nonfinite output')
    if args.output:np.save(args.output,pixels)
    backend={k:v.decode() if isinstance(v,bytes) else v for k,v in dict(core.nss.Backend()).items()}
    props={k:v for k,v in frame.props.items() if k.startswith('_NSS') and isinstance(v,(int,float,list))}
    metadata=json.loads(Path(args.fixture_meta).read_text()) if args.fixture_meta else None
    host=dict(hostname=platform.node(),kernel=platform.release(),machine=platform.machine(),
              boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              cpu_model=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                             if line.startswith('model name')))
    print(json.dumps(dict(schema='nss.nlh-candidate-profile.v1',shape=list(noisy.shape),frames=args.frames,
               seconds=elapsed,seconds_per_frame=elapsed/args.frames,warmup_seconds=warm_seconds,
               parameters=parameters,resolved=props,fixture=metadata,input_file_sha256=file_sha(args.input),
               input_sha256=hashlib.sha256(noisy.tobytes()).hexdigest(),
               output_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
               plugin_sha256=file_sha(args.plugin),script_sha256=file_sha(__file__),backend=backend,
               host=host,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               timed_source_fills=fills[0]-before_fills,cpu_activity=cpu_activity(before,after),
               affinity=sorted(os.sched_getaffinity(0)),perf_control_ack=acknowledgements,
               timing_scope='warm complete get_frame; six source frames held; unique interior requests; serialization excluded'),allow_nan=False),flush=True)


if __name__=='__main__':main()
