#!/usr/bin/env python3
"""Gate perf at the same frame-request boundary as integration wall timing."""
import argparse
import json
import os
import time
import types

import c4_integration as integration


def run(plugin, config, control, acknowledge):
    ctl = os.open(control, os.O_RDWR)
    ack = os.open(acknowledge, os.O_RDWR)
    calls = 0

    def command(value):
        os.write(ctl, (value + '\n').encode())
        if os.read(ack, 128).strip() != b'ack':
            raise RuntimeError('unexpected perf control acknowledgement')

    def boundary():
        nonlocal calls
        calls += 1
        if calls == 1:
            command('enable')
            return time.perf_counter()
        if calls == 2:
            value = time.perf_counter()
            command('disable')
            return value
        raise RuntimeError('frame timing boundary changed')

    previous_time = integration.time
    integration.time = types.SimpleNamespace(perf_counter=boundary)
    try:
        result = integration.worker(plugin, config)
        if calls != 2:
            raise RuntimeError('frame timing boundary was not entered and exited')
        result['pmu_boundary'] = 'frame_requests_only'
        return result
    finally:
        integration.time = previous_time
        os.close(ctl)
        os.close(ack)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plugin', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--ack', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plugin, json.loads(args.config), args.control, args.ack)))
