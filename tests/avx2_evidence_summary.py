#!/usr/bin/env python3
"""Compact view of immutable raw campaign evidence (does not rewrite it)."""
import argparse,json
from pathlib import Path

def summarize(root):
 result={}
 for path in sorted(root.glob('mask-*')):
  if not path.is_dir():continue
  phase=dict(status=(path/'status').read_text().strip() if (path/'status').exists() else 'running',runs={})
  for run in ('bench','bench-remaining','followup'):
   p=path/run/'summary.json'
   if not p.exists():continue
   phase['runs'][run]=[dict(name=r['config']['name'],config=r['config'],speedup=r['paired_speedup'],
       ci95=r['ci95'],pairs=r['pairs'],frames=r.get('effective_config',r['config']).get('frames'),
       numerical=r['numerical'],environment_valid=r['environment']['valid'],selected=r.get('selected')) for r in json.loads(p.read_text())]
  for run in ('crossover','fixed-match-replay'):
   p=path/run/'replay.json'
   if p.exists():phase[run]=json.loads(p.read_text())
  result[path.name]=phase
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
 text=json.dumps(summarize(a.root),indent=2)
 if a.out:a.out.write_text(text)
 else:print(text)
