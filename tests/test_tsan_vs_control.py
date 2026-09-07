#!/usr/bin/env python3
"""Stock VapourSynth/NumPy control: deliberately does not load NSS."""
import gc
import numpy as np
import vapoursynth as vs
from test_plan01_plugin import clip
core = vs.core
core.num_threads = 8
source = clip(core, width=35, height=33, frames=5)
for repeat in range(100):
    node = core.std.Expr(source, 'x 0.9 *')
    for n in (4, 0, 2, 1):
        frame = node.get_frame(n)
        values = np.array(frame[0])
        assert np.isfinite(values).all()
        del values, frame
    del node
    gc.collect()
print('stock VS/NumPy control passed; no NSS plugin loaded')
