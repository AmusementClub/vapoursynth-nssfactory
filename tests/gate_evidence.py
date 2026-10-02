# SPDX-License-Identifier: GPL-2.0-only
"""Immutable per-invocation evidence; no root summary or latest-run inference."""
import json
import os
from pathlib import Path
import tempfile
import uuid


class GateEvidence:
    def __init__(self, root):
        root = Path(root).resolve()
        if root.exists() and not root.is_dir():
            raise ValueError(f'--out must be a directory: {root}')
        root.mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix='run-', dir=root))
        self.run_id = self.path.name
        self.invocation_id = os.environ.get('NSS_GATE_INVOCATION_ID') or uuid.uuid4().hex
        self.record = dict(run_id=self.run_id, invocation_id=self.invocation_id,
                           summary_path=str(self.path / 'summary.json'))
        print('NSS_GATE_STARTED ' + json.dumps(self.record), flush=True)

    def finish(self, report):
        report = dict(report, **self.record, completed=True)
        # The report becomes visible only when its complete contents exist.
        temporary = self.path / 'summary.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(self.path / 'summary.json')
        print('NSS_GATE_COMPLETED ' + json.dumps(dict(self.record, passed=report['passed'])), flush=True)
        return report['passed']
