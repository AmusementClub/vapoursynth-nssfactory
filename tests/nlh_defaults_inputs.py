#!/usr/bin/env python3
"""Freeze image-level splits and inputs for the NLH defaults experiment.

No denoiser output is consulted when selecting images, splits, crops or seeds.
Raw downloads remain artifacts; only this preparation procedure belongs in git.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

SEED = 20260909
OLD_IMAGES = {805, 822, 840}
AUTHOR_REVISION = 'e36d833ba39cdffc66dd048896687da31a6ea4bb'
AUTHOR_ROOT = f'https://raw.githubusercontent.com/njusthyk1972/NLH/{AUTHOR_REVISION}'
SIGMAS = (5, 15, 25, 50, 75, 100)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    return digest(Path(path).read_bytes())


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def selected_div2k():
    candidates = sorted(set(range(801, 901)) - OLD_IMAGES,
                        key=lambda i: digest(f'{SEED}:DIV2K:{i}'.encode()))
    return candidates[:30]


def download(root):
    from div2k_prepare import download as official_download
    raw = root / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    div = raw / 'div2k'
    provenance = div / 'download-provenance.json'
    if provenance.exists():
        recorded = json.loads(provenance.read_text())
        for row in recorded['members']:
            if file_sha(div / row['file']) != row['sha256']:
                raise ValueError('existing DIV2K download changed')
        if {int(Path(r['file']).stem) for r in recorded['members']} != set(selected_div2k()) | OLD_IMAGES:
            raise ValueError('existing DIV2K image selection differs')
    else:
        official_download(SimpleNamespace(ids=selected_div2k() + sorted(OLD_IMAGES), out=str(div)))
    cc = raw / 'cc'
    cc.mkdir(exist_ok=True)
    url = f'https://api.github.com/repos/njusthyk1972/NLH/contents/NLH_CC/real_images?ref={AUTHOR_REVISION}'
    with urllib.request.urlopen(url, timeout=45) as response:
        listing = json.load(response)

    def fetch(row):
        path = cc / row['name']
        if path.exists():
            payload = path.read_bytes()
        else:
            with urllib.request.urlopen(row['download_url'], timeout=45) as response:
                payload = response.read()
            path.write_bytes(payload)
        git_blob = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
        if git_blob != row['sha']:
            raise ValueError('CC source identity mismatch: ' + row['name'])
        return dict(file=path.name, url=row['download_url'], git_blob=git_blob,
                    sha256=digest(payload), bytes=len(payload))

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, [r for r in listing if r['name'].endswith('.png')]))
    save_json(cc / 'download-provenance.json', dict(revision=AUTHOR_REVISION, files=records))


def prepare(root):
    images = []
    selected = selected_div2k()
    for number in selected + sorted(OLD_IMAGES):
        path = root / 'raw' / 'div2k' / f'{number:04d}.png'
        rank = selected.index(number) if number in selected else -1
        split = 'development' if rank < 18 else 'selection' if rank < 24 else 'test'
        with Image.open(path) as image:
            size = list(image.size)
        images.append(dict(id=f'div2k-{number:04d}', dataset='DIV2K', group=f'div2k-{number:04d}',
                           split=split, control=number in OLD_IMAGES, size=size,
                           clean=str(path.relative_to(root)), clean_sha256=file_sha(path)))

    # The repository includes identical 5dma/5dmark3 aliases. Preserve their
    # provenance, but never count an alias as an independent photograph.
    seen = {}
    cc_images = []
    for path in sorted((root / 'raw' / 'cc').glob('*_mean.png')):
        if path.name.startswith('._'):
            continue
        name = path.name[:-len('_mean.png')]
        noisy = path.with_name(name + '_real.png')
        identity = (file_sha(path), file_sha(noisy))
        if identity in seen:
            seen[identity]['aliases'].append(name)
            continue
        camera = name.split('_iso')[0].replace('5dma', '5dmark3')
        with Image.open(path) as clean_image, Image.open(noisy) as noisy_image:
            if clean_image.size != noisy_image.size:
                raise ValueError('unaligned CC pair')
            size = list(clean_image.size)
        row = dict(id='cc-' + name, dataset='CC', group='cc-camera-' + camera,
                   control=False, size=size, aliases=[], clean=str(path.relative_to(root)),
                   noisy=str(noisy.relative_to(root)), clean_sha256=identity[0], noisy_sha256=identity[1])
        seen[identity] = row
        cc_images.append(row)
    groups = sorted({r['group'] for r in cc_images},
                    key=lambda g: (-sum(r['group'] == g for r in cc_images), digest(f'{SEED}:{g}'.encode())))
    targets = dict(development=.6*len(cc_images), selection=.2*len(cc_images), test=.2*len(cc_images))
    assigned = dict.fromkeys(targets, 0)
    for group in groups:
        split = max(targets, key=lambda s: targets[s]-assigned[s])
        members = [r for r in cc_images if r['group'] == group]
        for row in members:
            row['split'] = split
        assigned[split] += len(members)
    images.extend(cc_images)

    cases = []
    for image in images:
        if image['dataset'] == 'CC':
            cases.append(dict(id=image['id']+'-blind', image=image['id'], split=image['split'],
                              dataset='CC', format='rgb', sigma=None, seed=None, realization=0,
                              control=False))
        else:
            for fmt in ('gray', 'rgb'):
                for sigma in SIGMAS:
                    for realization in (0, 1):
                        label=f"{image['id']}-{fmt}-s{sigma}-r{realization}"
                        seed=int(digest(f'{SEED}:{label}'.encode())[:16], 16)
                        cases.append(dict(id=label, image=image['id'], split=image['split'],
                                          dataset='DIV2K', format=fmt, sigma=sigma, seed=seed,
                                          realization=realization, control=image['control']))
    manifest = dict(schema='nss.nlh-defaults-inputs.v1', seed=SEED, images=images, cases=cases,
                    split_policy='30 new DIV2K images:18/6/6; old3 development controls. CC grouped conservatively by camera across ISO/crops; exact aliases deduplicated.',
                    input_policy='Encoded RGB8/255; BT.601 luma for Gray. Unclipped float32 AWGN, sigma=injected sigma, generated on the full original before cropping. CC mean/real pairs unmodified.',
                    selection_policy='Image identities, center crops and splits fixed before running any denoiser.',
                    source_revision=AUTHOR_REVISION)
    validate_manifest(manifest)
    path = root / 'inputs.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('refusing to change the frozen input manifest')
    save_json(path, manifest)
    sheet = Image.new('RGB', (5*240, ((len(images)+4)//5)*186), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(images):
        with Image.open(root / row['clean']) as original:
            preview = original.convert('RGB')
            preview.thumbnail((230, 150))
        x, y = (i % 5)*240, (i//5)*186
        sheet.paste(preview, (x+5, y+30))
        draw.text((x+5,y+2), row['id'], fill='black')
        draw.text((x+5,y+15), row['split']+(' / control' if row['control'] else ''), fill='black')
    sheet.save(root / 'input-contact-sheet.jpg', quality=92)
    print(json.dumps(dict(manifest_sha256=file_sha(path), images=len(images), cases=len(cases),
                          cc_unique=len(cc_images), cc_splits=assigned), indent=2))


def validate_manifest(manifest):
    groups = {}; hashes = {}; ids = set()
    for row in manifest['images']:
        if row['id'] in ids:
            raise ValueError('duplicate image identifier')
        ids.add(row['id'])
        for key, mapping in ((row['group'], groups), (row['clean_sha256'], hashes)):
            if key in mapping and mapping[key] != row['split']:
                raise ValueError('source leakage across splits')
            mapping[key] = row['split']
        if row.get('control') and row['split'] != 'development':
            raise ValueError('old image escaped development')
    image_map = {r['id']:r for r in manifest['images']}
    for row in manifest['cases']:
        if row['split'] != image_map[row['image']]['split']:
            raise ValueError('case split differs from original')


def load_case(root, manifest, case, size=256):
    """Deterministic full-image noise keeps nested crops on the same realization."""
    row = next(r for r in manifest['images'] if r['id'] == case['image'])
    clean_path = root / row['clean']
    if file_sha(clean_path) != row['clean_sha256']:
        raise ValueError('clean image hash mismatch')
    with Image.open(clean_path) as image:
        clean = np.asarray(image.convert('RGB'), dtype=np.float64).transpose(2,0,1)/255
    if case['format'] == 'gray':
        clean = np.einsum('c,chw->hw', np.array([.299,.587,.114]), clean)[None]
    clean = clean.astype(np.float32)
    if case['dataset'] == 'CC':
        noisy_path = root / row['noisy']
        if file_sha(noisy_path) != row['noisy_sha256']:
            raise ValueError('noisy image hash mismatch')
        with Image.open(noisy_path) as image:
            noisy = (np.asarray(image.convert('RGB'), dtype=np.float64).transpose(2,0,1)/255).astype(np.float32)
    else:
        noise = np.random.Generator(np.random.PCG64(case['seed'])).standard_normal(clean.shape)
        noisy = (clean.astype(float)+noise*(case['sigma']/255)).astype(np.float32)
    _,h,w=clean.shape
    if size:
        if size>min(h,w): raise ValueError('requested crop exceeds original')
        y,x=(h-size)//2,(w-size)//2
        clean=clean[:,y:y+size,x:x+size].copy();noisy=noisy[:,y:y+size,x:x+size].copy()
        crop=[x,y,size,size]
    else:
        crop=[0,0,w,h]
    return clean,noisy,crop


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('download','prepare'))
    parser.add_argument('--out',required=True)
    args=parser.parse_args();root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=True)
    globals()[args.mode](root)


if __name__ == '__main__':
    main()
