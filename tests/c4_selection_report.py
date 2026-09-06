#!/usr/bin/env python3
"""Audit the user's bounded per-configuration selection policy from raw pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from collections import defaultdict


def report(root):
    output=[]
    for summary_path in sorted(root.rglob('summary.json')):
        directory=summary_path.parent
        selection_path=directory/'selection.json'
        if not selection_path.exists() and not (directory/'inputs-reverified.json').exists():
            continue
        summary=json.loads(summary_path.read_text())
        selection=json.loads(selection_path.read_text()) if selection_path.exists() else dict(threshold=1.02)
        raw=defaultdict(lambda: defaultdict(dict))
        for line in (directory/'raw.jsonl').read_text().splitlines():
            row=json.loads(line)
            key=row['config']['name']
            assert row['pair'] not in raw[key][row['variant']], (directory,key,'duplicate pair')
            raw[key][row['variant']][row['pair']]=row
        chosen=[]
        for item in summary:
            name=item['config']['name'];rows=raw[name]
            assert set(rows['baseline'])==set(rows['candidate'])==set(range(item['pairs'])), (directory,name,'partial pairs')
            ratios=[]
            for pair in range(item['pairs']):
                a,b=rows['baseline'][pair],rows['candidate'][pair]
                assert a['timed_frames']==b['timed_frames'],(directory,name,'unequal frame counts')
                assert a['timed_first']==b['timed_first'],(directory,name,'unequal timed range')
                ratios.append(a['ms']/b['ms'])
            speed=statistics.median(ratios)
            assert speed==item['paired_speedup'] and ratios==item['ratios'], (directory,name,'statistics mismatch')
            env=item['environment'];before,after=env['before'],env['after']
            sibling=f"cpu{env.get('sibling_cpu',1)}"
            delta=[b-a for a,b in zip(before[sibling],after[sibling])]
            valid=delta[3]/sum(delta[:8])>=.999 and after['cpu'][7]==before['cpu'][7]
            accepted=speed>selection['threshold'] and item['numerical']['passed'] and valid
            assert accepted==item['selected'],(directory,name,'selection mismatch')
            if accepted:chosen.append(name)
            output.append(dict(directory=str(directory),name=name,speedup=speed,ci95=item['ci95'],
                pairs=item['pairs'],accepted=accepted,numerical_passed=item['numerical']['passed'],
                environment_valid=valid,group_seconds=item['budget']['wall_seconds'],
                frames=item['effective_config']['frames'],
                raw_sha256=hashlib.sha256((directory/'raw.jsonl').read_bytes()).hexdigest()))
        if 'selected' in selection:
            assert chosen==selection['selected'],(directory,'selected list mismatch')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args();rows=report(args.root)
    value=json.dumps(rows,indent=2)
    if args.out:args.out.write_text(value)
    print(value)
