#!/usr/bin/env python3
"""Read-only idle core selection on a non-dedicated Linux host."""
import argparse
import json
from pathlib import Path
import time

def cpu_rows():
    return {row.split()[0]:list(map(int,row.split()[1:])) for row in Path('/proc/stat').read_text().splitlines() if row.startswith('cpu')}

def expand(value):
    result=[]
    for part in value.strip().split(','):
        pair=part.split('-');result.extend(range(int(pair[0]),int(pair[-1])+1))
    return result

def probe(seconds):
    before=cpu_rows();time.sleep(seconds);after=cpu_rows()
    pairs=[]
    for cpu in expand(Path('/sys/devices/system/cpu/online').read_text()):
        siblings=expand(Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text())
        if len(siblings)!=2 or cpu!=min(siblings):continue
        idle=[]
        for value in siblings:
            delta=[b-a for a,b in zip(before[f'cpu{value}'],after[f'cpu{value}'])]
            idle.append(delta[3]/sum(delta[:8]))
        pairs.append(dict(cpu=siblings[0],sibling_cpu=siblings[1],idle=idle))
    pairs.sort(key=lambda p:(min(p['idle']),sum(p['idle'])) ,reverse=True)
    return dict(seconds=seconds,before=before,after=after,pairs=pairs,
                selected=next((p for p in pairs if min(p['idle'])>=.999),None),
                loadavg=Path('/proc/loadavg').read_text().strip())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=5);p.add_argument('--out',type=Path)
    a=p.parse_args();result=probe(a.seconds)
    if a.out:a.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result['selected']))
