#!/usr/bin/env python3
"""Syntax-audit current CPU TUs as Highway SVE2_128 without editing source.

This deliberately does not build/load a production SVE plugin or change its
NEON admission policy. Fixed-vector type feasibility is a separate small probe.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
entries = json.loads((args.build/'compile_commands.json').read_text())
report = dict(schema='nssfactory.sve-compile-audit.v1', complete=False,
              target='Highway SVE2_128', vector_bits=128, runtime_admission=False, results=[])
try:
    for entry in entries:
        path = Path(entry['file'])
        if '/src/cpu/' not in str(path):
            continue
        command = entry.get('arguments') or shlex.split(entry['command'])
        rewritten = []
        skip = False
        for token in command:
            if skip:
                skip = False
                continue
            if token in ('-o','-MF','-MT','-MQ'):
                skip = True
                continue
            if token in ('-c','-MD','-MMD') or token.startswith(('-DHWY_DISABLED_TARGETS=', '-DHWY_COMPILE_ONLY_', '-march=', '-mcpu=', '-msve-vector-bits=')):
                continue
            rewritten.append(token)
        rewritten += ['-DHWY_DISABLED_TARGETS=0', '-DHWY_COMPILE_ONLY_STATIC=1',
                      '-march=armv9-a+sve2', '-msve-vector-bits=128', '-fsyntax-only', '-fmax-errors=3']
        name = str(path).split('/src/cpu/',1)[1].replace('/','_')
        start = time.monotonic()
        result = subprocess.run(rewritten,cwd=entry['directory'],capture_output=True,text=True,timeout=90)
        (args.out/(name+'.log')).write_text(result.stdout+result.stderr)
        row = dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   command=rewritten,returncode=result.returncode,seconds=time.monotonic()-start,
                   first_errors=[s for s in result.stderr.splitlines() if 'error:' in s][:3])
        report['results'].append(row)
        print(json.dumps(dict(source=name,returncode=result.returncode,errors=row['first_errors'])),flush=True)
    report['complete'] = True
finally:
    (args.out/'summary.json').write_text(json.dumps(report,indent=2))
