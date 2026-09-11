#!/usr/bin/env python3
"""Default-API denoiser comparison on saved paper_compare float32 fixtures.

Only sigma is supplied where the API accepts it. NLM retains h=1.2 and d=1.
MCWNNM receives the identical gray input in all RGB planes, not independent noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from paper_compare import cpu_activity, cpu_ticks, fixture, quality, save_json, sha

ALGORITHMS = ('NLM', 'BM3D', 'WNNM', 'MCWNNM', 'TWSC', 'NLH', 'NCSR', 'LSSC')


def plane_quality(clean, pixels):
    references = [clean]*len(pixels) if clean.ndim == 2 else clean
    assert len(references)==len(pixels)
    metrics = [quality(reference, plane) for reference,plane in zip(references,pixels)]
    mse = statistics.mean(m['mse'] for m in metrics)
    return dict(psnr_db=float(-10*np.log10(mse)),
                ssim=statistics.mean(m['ssim'] for m in metrics), mse=mse,
                planes=metrics)


def worker(args):
    import vapoursynth as vs

    root, case = fixture(args.fixtures, args.case)
    shape=(case.get('channels',1),case['height'],case['width'])
    noisy = np.fromfile(root/case['noisy'], dtype='<f4').reshape(shape)
    clean = np.fromfile(root/case['clean'], dtype='<f4').reshape(noisy.shape)
    core = vs.core
    core.num_threads = 1
    core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
    channels = 3 if args.algorithm == 'MCWNNM' or case.get('channels',1)==3 else 1
    if case.get('channels',1)==1 and channels==3:
        noisy=np.repeat(noisy,3,axis=0);clean=np.repeat(clean,3,axis=0)
    blank = core.std.BlankClip(width=case['width'], height=case['height'], length=6,
                              format=vs.RGBS if channels == 3 else vs.GRAYS)
    fills = 0

    def fill(n, f):
        nonlocal fills
        fills += 1
        output = f.copy()
        for p in range(channels): np.asarray(output[p])[:] = noisy[p]
        return output

    source = core.std.ModifyFrame(blank, blank, fill)
    core.std.SetVideoCache(source, mode=1, fixedsize=6, maxsize=6)
    # NLM default d=1 uses an interior frame with the same repeated noisy image,
    # not additional independent observations or an implicit d=0 override.
    for n in range(6): source.get_frame(n)
    options = {} if args.algorithm == 'NLM' else {'sigma': case['sigma']}
    if args.strength is not None:
        assert np.isfinite(args.strength) and (args.strength > 0 if args.algorithm == 'NLM' else args.strength >= 0)
        options = {'h' if args.algorithm == 'NLM' else 'sigma': args.strength}
    extra=json.loads(args.parameters_json)
    allowed={'block_size','block_step','group_size','bm_range','residual','adaptive_aggregation',
             'iters','delta','admm_iter','q','a','s','lambda2'}
    assert isinstance(extra,dict) and set(extra)<=allowed
    options.update(extra)
    node = getattr(core.nss, args.algorithm)(source, **options)
    if not args.cold_evaluation:
        node.get_frame(1)
    before = fills
    start = time.perf_counter()
    frame = node.get_frame(4)
    seconds = time.perf_counter()-start
    assert fills == before, 'Source generated inside timing boundary'
    pixels = np.stack([np.asarray(frame[p]).copy() for p in range(channels)]).astype('<f4')
    assert np.isfinite(pixels).all()
    noise = noisy.astype(np.float64)-clean.astype(np.float64)
    error = pixels.astype(np.float64)-clean.astype(np.float64)
    noise_energy=float(np.sum(noise*noise))
    pixels.tofile(args.output)
    info = dict(case=case['id'], image=case['image'], sigma=case['sigma'], algorithm=args.algorithm,
        channels=channels, width=case['width'], height=case['height'], supplied_parameters=options,
        noisy_sha256=case['noisy_sha256'], clean_sha256=case['clean_sha256'],
        plugin_sha256=sha(args.plugin), harness_sha256=sha(__file__), output=Path(args.output).name,
        output_sha256=sha(args.output), seconds=seconds, quality=plane_quality(clean,pixels),
        noise_projection_retention=float(np.sum(error*noise)/noise_energy),
        removed_signal_rms_over_input_noise=float(np.sqrt(np.sum((pixels-noisy)**2)/noise_energy)),
        noisy_quality=plane_quality(clean,noisy), source_frames_preloaded=6, timed_source_fills=0,
        timing_boundary=('quality-only cold get_frame(4); not a warm benchmark' if args.cold_evaluation else
                         'warm get_frame(4) after get_frame(1); source preloaded; serialization excluded'),
        warmup_performed=not args.cold_evaluation,
        input_policy='same saved RGB with independent channel noise' if case.get('channels',1)==3 else
            'same noisy gray copied to RGB planes' if channels==3 else 'same saved noisy gray',
        temporal_policy='6 identical noisy frames; NLM default d=1; no independent temporal noise',
        backend={k:(v.decode() if isinstance(v,bytes) else v) for k,v in core.nss.Backend().items()})
    save_json(str(args.output)+'.json', info)


def eligible(row):
    cpu = row['cpu_activity_during_worker']
    return cpu['cpu1']['busy_fraction'] <= .01 and not any(v['steal_ticks'] for v in cpu.values())


def campaign(args):
    root = Path(args.out).resolve(); root.mkdir(parents=True, exist_ok=True)
    assert os.sched_getaffinity(0) == {0}
    cases = json.loads((Path(args.fixtures)/'fixtures.json').read_text())['cases']
    if args.cases:
        cases = [c for c in cases if c['id'] in args.cases]
        assert len(cases) == len(args.cases), 'Unknown requested case'
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    save_json(root/'environment.json', dict(uname=list(os.uname()), cpuinfo=Path('/proc/cpuinfo').read_text(),
        affinity=sorted(os.sched_getaffinity(0)), plugin_sha256=sha(args.plugin), harness_sha256=sha(__file__),
        metrics_harness_sha256=sha(Path(__file__).with_name('paper_compare.py')),
        thread_environment={k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))
    result_path = root/'results.jsonl'
    rows = [json.loads(line) for line in result_path.read_text().splitlines()] if result_path.exists() else []
    done = {(r['case'],r['algorithm'],r['repeat']) for r in rows if r['ok']}
    for case in cases:
        for repeat in range(args.repeat_start, args.repeat_start+args.repeats):
            order = args.algorithms if repeat % 2 == 0 else list(reversed(args.algorithms))
            for algorithm in order:
                if (case['id'],algorithm,repeat) in done: continue
                output = root/f"{case['id']}-{algorithm}-r{repeat}.f32"
                cmd = [sys.executable, str(Path(__file__).resolve()), 'worker', '--fixtures',
                    str(Path(args.fixtures).resolve()), '--case',case['id'],'--algorithm',algorithm,
                    '--plugin',str(Path(args.plugin).resolve()),'--output',str(output)]
                row = dict(case=case['id'],algorithm=algorithm,repeat=repeat)
                try:
                    before = cpu_ticks()
                    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.timeout)
                    row['cpu_activity_during_worker'] = cpu_activity(before,cpu_ticks())
                    output.with_suffix('.log').write_bytes(proc.stdout)
                    if proc.returncode: raise RuntimeError(proc.stdout[-2500:].decode(errors='replace'))
                    row.update(json.loads(Path(str(output)+'.json').read_text()),ok=True)
                    row['timing_eligible'] = eligible(row)
                except Exception as error:
                    row.update(ok=False,error=str(error))
                rows.append(row)
                with result_path.open('a') as stream: stream.write(json.dumps(row)+'\n')
                print(f"{case['id']} {algorithm} r{repeat} " +
                    (f"{row['quality']['psnr_db']:.3f}dB {row['seconds']:.6f}s idle={row['timing_eligible']}" if row['ok'] else row['error']), flush=True)
    save_json(root/'results.json', dict(schema='nss.defaults-comparison.v1',rows=rows))


def report(args):
    from PIL import Image, ImageDraw, ImageFont

    root = Path(args.results).resolve()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    native = out/'native'; native.mkdir(exist_ok=True)
    fixtures = Path(args.fixtures).resolve()
    cases = json.loads((fixtures/'fixtures.json').read_text())['cases']
    rows = json.loads((root/'results.json').read_text())['rows']
    failed = [r for r in rows if not r['ok']]
    groups = {}
    for row in rows:
        if not row['ok']: continue
        froot, case = fixture(fixtures,row['case'])
        assert sha(root/row['output']) == row['output_sha256']
        assert row['noisy_sha256'] == case['noisy_sha256']
        assert row['clean_sha256'] == case['clean_sha256']
        assert row['timed_source_fills'] == 0
        pixels = np.fromfile(root/row['output'],dtype='<f4').reshape(row['channels'],row['height'],row['width'])
        clean_shape=(3,row['height'],row['width']) if case.get('channels',1)==3 else (row['height'],row['width'])
        clean = np.fromfile(froot/case['clean'],dtype='<f4').reshape(clean_shape)
        metrics = plane_quality(clean,pixels)
        for key in ('psnr_db','ssim','mse'): assert abs(metrics[key]-row['quality'][key]) < 1e-10
        groups.setdefault((row['case'],row['algorithm']),[]).append(row)
    cells = {}
    for key, items in groups.items():
        assert len({r['output_sha256'] for r in items}) == 1, 'Non-deterministic repeat output'
        good = [r for r in items if eligible(r)]
        cells[key] = dict(row=items[0],quality=items[0]['quality'],samples=len(items),
            timing_samples=len(good),seconds=statistics.median(r['seconds'] for r in good) if good else None)
    expected = {(c['id'],a) for c in cases for a in ALGORITHMS}
    missing = sorted(expected-cells.keys())
    summary = []
    for algorithm in ALGORITHMS:
        selected = [cells[(c['id'],algorithm)] for c in cases if (c['id'],algorithm) in cells]
        if not selected: continue
        summary.append(dict(algorithm=algorithm,cases=len(selected),
            mean_psnr=statistics.mean(c['quality']['psnr_db'] for c in selected),
            mean_ssim=statistics.mean(c['quality']['ssim'] for c in selected),
            mean_seconds=statistics.mean(c['seconds'] for c in selected) if all(c['seconds'] is not None for c in selected) else None,
            worse_than_noisy=sum(c['quality']['psnr_db'] < c['row']['noisy_quality']['psnr_db']-.01 for c in selected)))
    save_json(out/'summary.json',dict(expected_cells=len(expected),completed_cells=len(cells),missing=missing,
        failed=failed,total_rows=len(rows),all_hashes_and_metrics_verified=True,repeat_hashes_equal=True,
        degradation_count_threshold_db=.01,
        timing_flags=[dict(case=r['case'],algorithm=r['algorithm'],repeat=r['repeat'],cpu=r['cpu_activity_during_worker']) for r in rows if r['ok'] and not eligible(r)],
        cells=[dict(case=k[0],algorithm=k[1],**{f:v for f,v in c.items() if f!='row'}) for k,c in cells.items()],overall=summary))

    def raster(pixels):
        if pixels.ndim == 3:
            pixels = pixels[0] if pixels.shape[0]==1 else pixels.transpose(1,2,0)
        return Image.fromarray(np.rint(np.clip(pixels,0,1)*255).astype(np.uint8)).convert('RGB')

    def font(size):
        for path in ('/System/Library/Fonts/Menlo.ttc','/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'):
            try: return ImageFont.truetype(path,size)
            except OSError: pass
        return ImageFont.load_default()

    title_font, label_font = font(20), font(15)
    tile, gap, label_h, heading = args.tile, 12, 66, 58
    width = 5*tile+6*gap
    height = heading+2*(tile+label_h+gap)+gap
    case_sheets = {}
    lines = ['# 本项目 8 算法默认参数对照','',
        '同一份未裁剪 float32 AWGN 输入；除匹配输入的 sigma 外不传算法参数。NLM 没有 sigma，保持默认 h=1.2、d=1；用 6 张完全相同的含噪静帧，分别在内部帧 1 warmup、4 计时，无独立时域观测。MCWNNM 用同一灰度含噪图复制三通道，标为 RGB-copy，不代表真实彩色表现或与灰度相同的工作量。',
        'BM3D 默认未提供 ref，因此测的是单次基础阶段；没有添加第二次 Wiener 阶段。VAggregate 是聚合器，Version/Backend 是查询，均非独立去噪算法。',
        'C4 CPU0、单 VS 线程，6 张源帧全部预载，源填充次数为零。每个组合 3 个独立 worker、交替算法执行顺序；不合格 CPU1 活动或 steal 样本不计入耗时，原始记录保留。',
        'PSNR/SSIM 对原始 float 输出计算；RGB-copy 的 PSNR 使用三通道合并 MSE，SSIM 取通道均值。展示图统一裁剪到 [0,1] 后映射到 8-bit，无自动对比度、锐化或平滑；128 图使用 2× 最近邻，256 图使用原尺寸。native 内 PNG 为原尺寸 8-bit 预览；精确 float 输出见原始结果目录。','',
        '| 算法 | 平均 PSNR dB | 平均 SSIM | 平均毫秒/帧 | PSNR 比输入下降 >0.01dB 的项数 |',
        '|---|---:|---:|---:|---:|']
    if any(c.get('channels',1)==3 for c in cases):
        lines[2]='同一份真实 RGB planar float32 AWGN 输入，各通道噪声独立；除匹配输入的 sigma 外不传算法参数。NLM 保留默认 h=1.2、d=1、AUTO→RGB；所有算法均收到同一份 RGB 输入，MCWNNM 不使用灰度复制。6 张相同含噪静帧不提供独立时域观测。'
        lines[5]=f'PSNR 对全部通道合并 MSE 计算，SSIM 为通道均值；均基于未裁剪 float。显示统一裁剪到 [0,1] 后映射到8bit。各原生512 ROI 不经缩放进入算法，对照格为{tile}像素；native内保存512原尺寸预览。来源DIV2K官网：https://data.vision.ee.ethz.ch/cvl/DIV2K/ 。'
    for r in summary:
        ms=f"{r['mean_seconds']*1000:.3f}" if r['mean_seconds'] is not None else '未验证'
        lines.append(f"| {r['algorithm']} | {r['mean_psnr']:.3f} | {r['mean_ssim']:.4f} | {ms} | {r['worse_than_noisy']} |")
    for case in cases:
        shape=(case['height'],case['width'])
        input_shape=(case.get('channels',1),*shape)
        clean=np.fromfile(fixtures/case['clean'],dtype='<f4').reshape(input_shape)
        noisy=np.fromfile(fixtures/case['noisy'],dtype='<f4').reshape(input_shape)
        entries=[('Clean',clean,None,None),('Noisy',noisy,plane_quality(clean,noisy),None)]
        for a in ALGORITHMS:
            cell=cells.get((case['id'],a))
            if cell:
                row=cell['row']; pixels=np.fromfile(root/row['output'],dtype='<f4').reshape(row['channels'],*shape)
                label=a+(' RGB-copy' if a=='MCWNNM' and case.get('channels',1)==1 else ' h=1.2,d=1' if a=='NLM' else '')
                entries.append((label,pixels,cell['quality'],cell['seconds']))
            else: entries.append((a+' FAILED',np.zeros(shape),None,None))
        sheet=Image.new('RGB',(width,height),'#f1f3f5'); draw=ImageDraw.Draw(sheet)
        draw.text((gap,10),f"{case['image']}  {case['width']}x{case['height']}  sigma={case['sigma']:g}  |  DEFAULT API",fill='#17212e',font=title_font)
        draw.text((gap,35),'Same input / fixed 0-1 display / PSNR & SSIM on unclipped float output',fill='#465468',font=label_font)
        for i,(label,pixels,metrics,seconds) in enumerate(entries):
            x=gap+(i%5)*(tile+gap); y=heading+(i//5)*(tile+label_h+gap)
            draw.text((x,y),label,fill='#17212e',font=label_font)
            if metrics:
                draw.text((x,y+19),f"{metrics['psnr_db']:.2f} dB | SSIM {metrics['ssim']:.3f}",fill='#465468',font=label_font)
            if seconds is not None: draw.text((x,y+38),f'{seconds*1000:.2f} ms',fill='#465468',font=label_font)
            picture=raster(pixels)
            filename=f"{case['id']}-{('clean','noisy',*ALGORITHMS)[i]}.png"
            picture.save(native/filename)
            sheet.paste(picture.resize((tile,tile),Image.Resampling.NEAREST),(x,y+label_h))
        path=out/(case['id']+'.png'); sheet.save(path); case_sheets[case['id']]=sheet
        lines += ['',f"## {case['id']}",'',f'![default comparison]({path.name})','',
            '| 算法 | PSNR dB | SSIM | 毫秒 | 有效计时重复 |','|---|---:|---:|---:|---:|']
        for a in ALGORITHMS:
            if (case['id'],a) not in cells: continue
            c=cells[(case['id'],a)]; ms=f"{c['seconds']*1000:.3f}" if c['seconds'] is not None else '未验证'
            lines.append(f"| {a} | {c['quality']['psnr_db']:.3f} | {c['quality']['ssim']:.4f} | {ms} | {c['timing_samples']}/{c['samples']} |")
    for sigma in sorted({c['sigma'] for c in cases}):
        selected=[case_sheets[c['id']] for c in cases if c['sigma']==sigma]
        sheet=Image.new('RGB',(width,len(selected)*height),'#f1f3f5')
        for i,picture in enumerate(selected): sheet.paste(picture,(0,i*height))
        sheet.save(out/f'all-defaults-s{sigma:g}.png')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(completed_cells=len(cells),expected_cells=len(expected),overall=summary),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('worker')
    for name in ('fixtures','case','algorithm','plugin','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--strength',type=float,help='Explicit sigma override, or h for NLM, for bounded tuning only')
    p.add_argument('--cold-evaluation',action='store_true',help='Quality-search-only single filtered call; its timing is not a warm benchmark')
    p.add_argument('--parameters-json',default='{}',help='Bounded public structural overrides for tuning')
    p=sub.add_parser('campaign')
    for name in ('fixtures','out','plugin'): p.add_argument('--'+name,required=True)
    p.add_argument('--algorithms',choices=ALGORITHMS,nargs='+',default=list(ALGORITHMS))
    p.add_argument('--cases',nargs='+'); p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--repeat-start',type=int,default=0);p.add_argument('--timeout',type=int,default=300)
    p=sub.add_parser('report')
    for name in ('results','fixtures','out'): p.add_argument('--'+name,required=True)
    p.add_argument('--tile',type=int,default=256)
    args=parser.parse_args();globals()[args.mode](args)


if __name__=='__main__': main()
