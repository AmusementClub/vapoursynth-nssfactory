# SPDX-License-Identifier: GPL-2.0-only
"""Cross-backend reference cases (CUDA plan C1).

One module defines the deterministic inputs and the per-filter cases so that
the reference generator (make_refs.py) and the comparison harness
(../cuda_compare.py) build bit-identical input clips. Inputs are crops of the
local ds/ snapshot images plus seeded AWGN; ds/ is not in git, so the manifest
records each source image's SHA-256 and every generated input array's hash.
"""
import hashlib
from pathlib import Path

import numpy as np

SCHEMA = "nssfactory.cuda_refs.v1"
REPO = Path(__file__).resolve().parents[2]
DS = REPO / "ds"
CROP = 256

# name: (image, kind, size, sigma8, frames, pan). size=None keeps the full image.
INPUTS = {
    "g256": ("MAPPA.png", "gray", CROP, 3.0, 1, False),
    "g256s25": ("MAPPA.png", "gray", CROP, 25.0, 1, False),
    "g128": ("MAPPA.png", "gray", 128, 3.0, 1, False),
    "rgb256": ("Ufotable.png", "rgb", CROP, 3.0, 1, False),
    "rgb256s25": ("Ufotable.png", "rgb", CROP, 25.0, 1, False),
    "yuv256": ("Ufotable.png", "yuv420", CROP, 3.0, 1, False),
    "g1080": ("MAPPA.png", "gray", None, 3.0, 1, False),
    "rgb1080": ("Ufotable.png", "rgb", None, 3.0, 1, False),
    "gseq": ("MAPPA.png", "gray", CROP, 3.0, 5, True),
    "rgbseq": ("Ufotable.png", "rgb", CROP, 3.0, 5, True),
}

SPATIAL = (0,)
TEMPORAL = (0, 2)  # clipped boundary window and a full window


def case(cid, filt, inp, args=None, frames=SPATIAL, aggregate=0):
    return dict(id=cid, filter=filt, input=inp, args=args or {}, frames=list(frames), aggregate=aggregate)


LIGHT_TWSC = dict(block_step=8, group_size=8, iters=2)
CASES = [
    # BM3D (+ VAggregate / rolling)
    case("bm3d-g256", "BM3D", "g256"),
    case("bm3d-g256-s25", "BM3D", "g256s25", dict(sigma=25)),
    case("bm3d-g256-s25-final", "BM3D", "g256s25", dict(sigma=25, ref={"$call": "BM3D", "args": dict(sigma=25)})),
    case("bm3d-g256-s25-b4", "BM3D", "g256s25", dict(sigma=25, block_size=4)),
    case("bm3d-g256-s25-b12g16", "BM3D", "g256s25", dict(sigma=25, block_size=12, group_size=16)),
    case("bm3d-g256-s25-b16g16", "BM3D", "g256s25", dict(sigma=25, block_size=16, group_size=16)),
    case("bm3d-g256-s25-b8g32-step4", "BM3D", "g256s25", dict(sigma=25, group_size=32, block_step=4, bm_range=12)),
    case("bm3d-rgb256", "BM3D", "rgb256"),
    case("bm3d-yuv256", "BM3D", "yuv256", dict(sigma=[3, 2, 2])),
    case("bm3d-g1080", "BM3D", "g1080"),
    case("bm3d-gseq-r1-legacy", "BM3D", "gseq", dict(radius=1), TEMPORAL, aggregate=1),
    case("bm3d-gseq-r2-legacy", "BM3D", "gseq", dict(radius=2), TEMPORAL, aggregate=2),
    case("bm3d-gseq-r1-rolling", "BM3D", "gseq", dict(radius=1, temporal_mode="rolling"), TEMPORAL),
    case("bm3d-rgbseq-r1-legacy", "BM3D", "rgbseq", dict(radius=1), TEMPORAL, aggregate=1),
    # NLM (default d=1 is temporal)
    case("nlm-gseq", "NLM", "gseq", frames=TEMPORAL),
    case("nlm-g256s25-d0-h3", "NLM", "g256s25", dict(d=0, h=3.0)),
    case("nlm-rgbseq", "NLM", "rgbseq", frames=TEMPORAL),
    case("nlm-yuv256-d0-uv", "NLM", "yuv256", dict(d=0, channels="UV")),
    case("nlm-g1080-d0", "NLM", "g1080", dict(d=0)),
    # WNNM
    case("wnnm-g256", "WNNM", "g256"),
    case("wnnm-g256-s25", "WNNM", "g256s25", dict(sigma=25)),
    case("wnnm-g256-s25-step4-g16-res", "WNNM", "g256s25", dict(sigma=25, block_step=4, group_size=16, residual=1)),
    case("wnnm-rgb256", "WNNM", "rgb256"),
    case("wnnm-gseq-r1", "WNNM", "gseq", dict(radius=1), TEMPORAL, aggregate=1),
    case("wnnm-g1080", "WNNM", "g1080"),
    # MCWNNM
    case("mcwnnm-rgb256", "MCWNNM", "rgb256"),
    case("mcwnnm-rgb256-s25", "MCWNNM", "rgb256s25", dict(sigma=25)),
    case("mcwnnm-rgbseq-r1", "MCWNNM", "rgbseq", dict(radius=1), TEMPORAL, aggregate=1),
    case("mcwnnm-rgb1080", "MCWNNM", "rgb1080"),
    # NCSR
    case("ncsr-g256", "NCSR", "g256"),
    case("ncsr-g256-s25", "NCSR", "g256s25", dict(sigma=25)),
    case("ncsr-rgb256", "NCSR", "rgb256"),
    case("ncsr-gseq-r1", "NCSR", "gseq", dict(radius=1), TEMPORAL, aggregate=1),
    case("ncsr-g1080", "NCSR", "g1080"),
    # LSSC
    case("lssc-g256", "LSSC", "g256"),
    case("lssc-g256-s25", "LSSC", "g256s25", dict(sigma=25)),
    case("lssc-g256-s25-step4", "LSSC", "g256s25", dict(sigma=25, block_step=4)),
    # LSSC has no temporal mode (radius>0 is rejected at creation).
    case("lssc-g1080", "LSSC", "g1080"),
    # TWSC (default geometry only on a small crop; it is ~hours/frame at 1080p)
    case("twsc-g128-default", "TWSC", "g128"),
    case("twsc-g256-light", "TWSC", "g256", LIGHT_TWSC),
    case("twsc-g256-s25-light", "TWSC", "g256s25", dict(LIGHT_TWSC, sigma=25)),
    case("twsc-rgb256-light", "TWSC", "rgb256", LIGHT_TWSC),
    case("twsc-g256-blind-light", "TWSC", "g256", dict(LIGHT_TWSC, sigma=None, estimate_sigma=1)),
    case("twsc-gseq-r1-light", "TWSC", "gseq", dict(LIGHT_TWSC, radius=1), TEMPORAL, aggregate=1),
    case("twsc-g1080-light", "TWSC", "g1080", LIGHT_TWSC),
    # NLH
    case("nlh-g256-s3", "NLH", "g256", dict(sigma=3)),
    case("nlh-g256-s25", "NLH", "g256s25", dict(sigma=25)),
    case("nlh-g256-blind", "NLH", "g256"),
    case("nlh-rgb256-blind", "NLH", "rgb256"),
    case("nlh-yuv256-s3", "NLH", "yuv256", dict(sigma=[3, 2, 2])),
    case("nlh-gseq-r1-s3", "NLH", "gseq", dict(sigma=3, radius=1), TEMPORAL, aggregate=1),
    case("nlh-g1080-s3", "NLH", "g1080", dict(sigma=3)),
]
CASE_BY_ID = {c["id"]: c for c in CASES}
assert len(CASE_BY_ID) == len(CASES), "duplicate case id"


def _seed(label):
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_rgb(image):
    from PIL import Image
    with Image.open(DS / image) as pic:
        return np.asarray(pic.convert("RGB"), dtype=np.float64).transpose(2, 0, 1) / 255.0


def _convert(rgb, kind):
    if kind == "rgb":
        return list(rgb)
    if kind == "gray":
        return [np.einsum("c,chw->hw", np.array([0.299, 0.587, 0.114]), rgb)]
    if kind == "yuv420":  # full-range BT.709, centred chroma, 2x2 box subsampling
        r, g, b = rgb
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        u = (b - y) / 1.8556
        v = (r - y) / 1.5748
        sub = lambda c: c.reshape(c.shape[0] // 2, 2, c.shape[1] // 2, 2).mean(axis=(1, 3))
        return [y, sub(u), sub(v)]
    raise ValueError(kind)


def make_input(name):
    """Return (clean, noisy): lists over frames of lists of float32 planes."""
    image, kind, size, sigma8, frames, pan = INPUTS[name]
    rgb = _load_rgb(image)
    _, h, w = rgb.shape
    clean_frames, noisy_frames = [], []
    for t in range(frames):
        if size is None:
            crop = rgb
        else:
            top, left = (h - size) // 2 + (t if pan else 0), (w - size) // 2 + (2 * t if pan else 0)
            crop = rgb[:, top:top + size, left:left + size]
        clean, noisy = [], []
        for p, plane in enumerate(_convert(crop, kind)):
            rng = np.random.Generator(np.random.PCG64(_seed(f"{SCHEMA}:{name}:{t}:{p}")))
            clean.append(np.ascontiguousarray(plane, dtype=np.float32))
            noisy.append((plane + rng.standard_normal(plane.shape) * (sigma8 / 255.0)).astype(np.float32))
        clean_frames.append(clean)
        noisy_frames.append(noisy)
    return clean_frames, noisy_frames


def arrays_sha256(frames):
    h = hashlib.sha256()
    for planes in frames:
        for plane in planes:
            h.update(np.ascontiguousarray(plane, dtype="<f4").tobytes())
    return h.hexdigest()


def to_clip(core, vs, frames, kind):
    fmt = {"gray": vs.GRAYS, "rgb": vs.RGBS, "yuv420": vs.YUV420PS}[kind]
    height, width = frames[0][0].shape
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=len(frames))

    def fill(n, f):
        out = f.copy()
        for p, plane in enumerate(frames[n]):
            np.asarray(out[p])[:] = plane
        return out
    return core.std.ModifyFrame(blank, blank, fill)


def build(core, ns, clip, c):
    """Instantiate case c on namespace ns over clip."""
    plugin = getattr(core, ns)

    def resolve(args):
        out = {}
        for key, value in args.items():
            if value is None:
                continue
            if isinstance(value, dict) and "$call" in value:
                value = getattr(plugin, value["$call"])(clip, **resolve(value["args"]))
            out[key] = value
        return out
    node = getattr(plugin, c["filter"])(clip, **resolve(c["args"]))
    if c["aggregate"]:
        node = plugin.VAggregate(node, clip, radius=c["aggregate"])
    return node


def fetch(node, frames):
    out = []
    for n in frames:
        f = node.get_frame(n)
        out.append([np.array(f[p], dtype=np.float32, copy=True) for p in range(f.format.num_planes)])
    return out


def psnr(a, b, peak=1.0):
    num = sum(float(np.sum((x.astype(np.float64) - y) ** 2)) for x, y in zip(a, b))
    count = sum(x.size for x in a)
    mse = num / count
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)


def compare(test, ref):
    """test/ref: lists over frames of lists of planes."""
    flat_t = [p for planes in test for p in planes]
    flat_r = [p for planes in ref for p in planes]
    if [p.shape for p in flat_t] != [p.shape for p in flat_r]:
        raise ValueError("shape mismatch")
    diff = np.concatenate([np.abs(t.astype(np.float64) - r).ravel() for t, r in zip(flat_t, flat_r)])
    return dict(psnr=psnr(flat_t, flat_r), max_abs=float(diff.max()),
                frac_gt_1e3=float(np.mean(diff > 1e-3)), finite=bool(all(np.isfinite(t).all() for t in flat_t)))
