#!/usr/bin/env python3
"""Run and archive the explicitly partial, fixed-dictionary paper reference."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

from reference import denoise, validate_dictionary


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_image(path, width=None, height=None):
    path = Path(path)
    if path.suffix == '.npy':
        pixels = np.load(path, allow_pickle=False)
    elif path.suffix == '.f32':
        if width is None or height is None or min(width, height) < 1:
            raise ValueError('raw float32 input requires --width and --height')
        pixels = np.fromfile(path, dtype='<f4').reshape(height, width)
    else:
        raise ValueError('input must be .npy or little-endian .f32; no implicit rescale')
    if pixels.ndim != 2 or not np.isfinite(pixels).all() or np.iscomplexobj(pixels):
        raise ValueError('finite 2D grayscale input required')
    return np.asarray(pixels, dtype=np.float64)


def metrics(clean, pixels):
    clean, pixels = np.asarray(clean, dtype=np.float64), np.asarray(pixels, dtype=np.float64)
    if clean.shape != pixels.shape:
        raise ValueError('metric image sizes differ')
    mse = float(np.mean((clean-pixels)**2))
    result = {'mse': mse, 'psnr_db': -10*np.log10(mse) if mse else None}
    if min(clean.shape) >= 11:
        smooth = lambda x: gaussian_filter(x, sigma=1.5, radius=5, mode='reflect')
        a, b = smooth(clean), smooth(pixels)
        va, vb, cov = smooth(clean*clean)-a*a, smooth(pixels*pixels)-b*b, smooth(clean*pixels)-a*b
        ssim = ((2*a*b+.01**2)*(2*cov+.03**2))/((a*a+b*b+.01**2)*(va+vb+.03**2))
        result['ssim'] = float(ssim[5:-5, 5:-5].mean())
    return result


def stage_arrays(stage, prefix):
    lengths = np.array([len(r.support) for r in stage.pursuits], dtype=np.int64)
    return {prefix+'_support_offsets': np.r_[0, np.cumsum(lengths)],
            prefix+'_support_indices': np.concatenate([r.support for r in stage.pursuits]),
            prefix+'_residual_squared': np.array([r.residual_squared for r in stage.pursuits]),
            prefix+'_epsilon': np.array([r.epsilon for r in stage.pursuits]),
            prefix+'_patches': stage.patches}


def stage_summary(stage):
    return {'groups': len(stage.groups), 'patches': len(stage.patches),
            'group_size_histogram': dict(sorted(Counter(len(g) for g in stage.groups).items())),
            'support_size_histogram': dict(sorted(Counter(len(r.support) for r in stage.pursuits).items())),
            'stop_reasons': dict(Counter(r.stop_reason for r in stage.pursuits)),
            'max_residual_budget_excess': max(r.residual_squared-r.epsilon for r in stage.pursuits)}


def preview(path, panels):
    tile, label_height = 256, 42
    sheet = Image.new('RGB', ((tile+8)*len(panels)+8, tile+label_height+8), '#efefef')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for index, (label, pixels) in enumerate(panels):
        x = 8 + index*(tile+8)
        draw.text((x,4),label,fill='black',font=font)
        image = Image.fromarray(np.rint(np.clip(pixels,0,1)*255).astype(np.uint8))
        scale = min(tile/image.width, tile/image.height)
        size = (max(1,round(image.width*scale)), max(1,round(image.height*scale)))
        image = image.resize(size, Image.Resampling.NEAREST)
        sheet.paste(image,(x+(tile-size[0])//2,label_height+(tile-size[1])//2))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--dictionary', required=True, help='External .mat D or .npy (m x atoms), MATLAB pixel order')
    parser.add_argument('--out', required=True, help='New directory; never overwrite an existing run')
    parser.add_argument('--sigma', required=True, type=float, help='Noise standard deviation in 8-bit units')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--tau', type=float, default=.8)
    parser.add_argument('--window', type=int, default=32)
    parser.add_argument('--stage', choices=['sc','ssc'], default='ssc')
    parser.add_argument('--clean', help='Metrics/preview only: never passed to the denoising pipeline')
    args = parser.parse_args()
    source = read_image(args.input, args.width, args.height)
    dictionary_path = Path(args.dictionary)
    if dictionary_path.suffix == '.mat':
        dictionary = loadmat(dictionary_path, variable_names=['D'])['D']
    elif dictionary_path.suffix == '.npy':
        dictionary = np.load(dictionary_path, allow_pickle=False)
    else:
        raise ValueError('dictionary must be .mat or .npy')
    dictionary = validate_dictionary(dictionary)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = denoise(source, dictionary, args.sigma/255, tau=args.tau, window=args.window, stage=args.stage,
                     progress=lambda stage,n,total: print(f'{stage} {n}/{total}', flush=True))
    elapsed = time.perf_counter()-started
    output = result.output.astype('<f4')
    pilot = result.pilot.astype('<f4')
    output.tofile(out/'output.f32')
    pilot.tofile(out/'pilot.f32')
    np.savez_compressed(out/'trace.npz', positions=result.positions, coverage=result.coverage,
                        labels=result.labels, **stage_arrays(result.pilot_stage,'pilot'),
                        **stage_arrays(result.final_stage,'final'))
    metadata = {'schema':'nss.lssc-paper-p1.v1', 'algorithm':'fixed_dictionary_'+args.stage,
                'complete_lssc':False, 'dictionary_learning':False,
                'unverified_choices':['MATLAB/F patch order vs author MEX', 'valid-grid boundary handling',
                    'raster greedy disjoint cover and even-window anchoring',
                    'matching on re-extracted aggregated SC pilot',
                    'literal raw-SSD threshold versus author internal normalization'],
                'parameters':{'sigma_8bit':args.sigma, 'tau':args.tau,'window':args.window,
                    'patch_area':dictionary.shape[0], 'atoms':dictionary.shape[1],
                    'matching_raw_ssd_threshold':result.matching_threshold},
                'shape':list(source.shape), 'seconds':elapsed,
                'timing_boundary':'one unpaired local Python denoise call; not a C4 performance result',
                'input_sha256':sha(args.input), 'dictionary_sha256':sha(args.dictionary),
                'output_sha256':sha(out/'output.f32'), 'pilot_sha256':sha(out/'pilot.f32'),
                'trace_sha256':sha(out/'trace.npz'),
                'source_hashes':{name:sha(Path(__file__).with_name(name)) for name in ('reference.py','run.py')},
                'environment':{'platform':platform.platform(),'python':platform.python_version(),
                    'numpy':np.__version__,'scipy':scipy.__version__,
                    'thread_variables':{name:os.getenv(name) for name in (
                        'OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}},
                'pilot':stage_summary(result.pilot_stage), 'final':stage_summary(result.final_stage)}
    panels = [('Noisy',source),('Fixed D: SC pilot',pilot),('Fixed D: '+args.stage.upper(),output)]
    if args.clean:
        clean = read_image(args.clean, args.width, args.height)
        metadata['clean_sha256'] = sha(args.clean)
        metadata['quality'] = {name:metrics(clean,pixels) for name,pixels in (
            ('noisy',source),('pilot',pilot),('output',output))}
        panels.insert(0,('Clean',clean))
    preview(out/'preview.png',panels)
    (out/'result.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'out':str(out),'seconds':elapsed,'quality':metadata.get('quality'),
                      'groups':len(result.final_stage.groups)},indent=2))


if __name__ == '__main__':
    main()
