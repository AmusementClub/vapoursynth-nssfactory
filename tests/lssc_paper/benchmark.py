#!/usr/bin/env python3
"""Warm paired SC/SSC timing, stage costs, and optional Python profile.

This times the inspectable Python reference including its trace allocations, not
an optimized C++ implementation. Linux can require a verified CPU0 affinity lane.
"""
import argparse
from contextlib import redirect_stdout
import cProfile
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import pstats
import resource
import statistics
import time

import numpy as np
from scipy.io import loadmat

import reference
from run import read_image, sha


def cpu_ticks():
    path = Path('/proc/stat')
    if not path.exists():
        return None
    result = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in ('cpu0','cpu1'):
            result[fields[0]] = list(map(int,fields[1:9]))
    return result


def cpu_delta(before, after):
    if before is None or after is None:
        return None
    result = {}
    for name, previous in before.items():
        values = np.asarray(after[name])-previous
        total = int(values.sum())
        result[name] = {'ticks':total,'busy_fraction':float((total-values[3]-values[4])/total) if total else 0.,
                        'steal_ticks':int(values[7])}
    return result


def digest(pixels):
    return hashlib.sha256(pixels.astype('<f4').tobytes()).hexdigest()


def eligible(activity):
    return activity is not None and activity['cpu1']['busy_fraction'] <= .01 and not any(
        value['steal_ticks'] for value in activity.values())


def stage_costs(image, dictionary, sigma):
    """Diagnostic wrappers; restored even on failure, never change array values."""
    names = ('extract_patches','encode_groups','aggregate','greedy_groups')
    original = {name:getattr(reference,name) for name in names}
    costs = {}
    calls = {}
    def wrap(name):
        def invoke(*args, **kwargs):
            index = calls.get(name,0)
            calls[name] = index+1
            start = time.perf_counter()
            value = original[name](*args,**kwargs)
            costs[f'{name}_{index}'] = time.perf_counter()-start
            return value
        return invoke
    try:
        for name in names:
            setattr(reference,name,wrap(name))
        start = time.perf_counter()
        result = reference.denoise(image,dictionary,sigma)
        total = time.perf_counter()-start
    finally:
        for name,value in original.items():
            setattr(reference,name,value)
    costs['other'] = total-sum(costs.values())
    return {'total_seconds':total,'components':costs,'output_sha256':digest(result.output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--dictionary',required=True)
    parser.add_argument('--sigma',type=float,required=True)
    parser.add_argument('--width',type=int)
    parser.add_argument('--height',type=int)
    parser.add_argument('--out',required=True)
    parser.add_argument('--pairs',type=int,default=5)
    parser.add_argument('--require-cpu0',action='store_true')
    parser.add_argument('--profile',action='store_true')
    args = parser.parse_args()
    if args.pairs < 1:
        raise ValueError('positive pairs required')
    for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):
        if os.getenv(variable) != '1':
            raise ValueError(f'{variable}=1 must be set before Python starts')
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
    if args.require_cpu0 and affinity != [0]:
        raise RuntimeError('launch with taskset -c 0 on the verified C4 lane')
    image = read_image(args.input,args.width,args.height)
    dictionary = (loadmat(args.dictionary,variable_names=['D'])['D'] if Path(args.dictionary).suffix=='.mat'
                  else np.load(args.dictionary,allow_pickle=False))
    dictionary = reference.validate_dictionary(dictionary)
    sigma = args.sigma/255
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    expected = {}
    for variant in ('sc','ssc'):
        result = reference.denoise(image,dictionary,sigma,stage=variant)
        expected[variant] = digest(result.output)
        result.output.astype('<f4').tofile(out/(variant+'.f32'))
        del result
    rows = []
    for pair in range(args.pairs):
        order = ('sc','ssc') if pair % 2 == 0 else ('ssc','sc')
        for variant in order:
            gc.collect()
            before = cpu_ticks()
            start = time.perf_counter()
            result = reference.denoise(image,dictionary,sigma,stage=variant)
            seconds = time.perf_counter()-start
            activity = cpu_delta(before,cpu_ticks())
            output_hash = digest(result.output)
            if output_hash != expected[variant]:
                raise AssertionError('same-input warm repeat output changed')
            row = {'pair':pair,'variant':variant,'seconds':seconds,'output_sha256':output_hash,
                   'cpu_activity':activity,'idle_sibling_verified':eligible(activity)}
            rows.append(row)
            with (out/'samples.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            del result
    valid_pairs = [pair for pair in range(args.pairs) if all(eligible(r['cpu_activity']) for r in rows if r['pair']==pair)]
    stage = stage_costs(image,dictionary,sigma)
    if stage['output_sha256'] != expected['ssc']:
        raise AssertionError('instrumentation changed output')
    if args.profile:
        profiler = cProfile.Profile()
        result = profiler.runcall(reference.denoise,image,dictionary,sigma)
        if digest(result.output) != expected['ssc']:
            raise AssertionError('profiling changed output')
        with (out/'profile.txt').open('w') as stream:
            pstats.Stats(profiler,stream=stream).sort_stats('cumtime').print_stats(35)
        del result
    stream = io.StringIO()
    with redirect_stdout(stream):
        np.show_config()
    use = [r for r in rows if r['pair'] in valid_pairs] if args.require_cpu0 else rows
    medians = {variant:statistics.median(r['seconds'] for r in use if r['variant']==variant)
               for variant in ('sc','ssc')} if use else None
    summary = {'schema':'nss.paper-python-timing.v1','samples':rows,'stage_costs':stage,
        'median_seconds':medians,'valid_idle_pairs':len(valid_pairs),'requested_pairs':args.pairs,
        'formal_c4_gate':False,'same_implementation_repeat_hashes_equal':True,
        'timing_boundary':'warm denoise including reference trace allocation; input/dictionary read and serialization excluded',
        'reference_only':'does not predict optimized C++ runtime or include dictionary learning',
        'input_sha256':sha(args.input),'dictionary_sha256':sha(args.dictionary),
        'source_hashes':{name:sha(Path(__file__).with_name(name)) for name in ('reference.py','benchmark.py')},
        'environment':{'platform':platform.platform(),'python':platform.python_version(),'affinity':affinity,
            'numpy_config':stream.getvalue(),'maxrss_platform_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    (out/'result.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'median_seconds':medians,'valid_idle_pairs':len(valid_pairs),'stage_costs':stage},indent=2))


if __name__ == '__main__':
    main()
