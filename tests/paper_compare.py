#!/usr/bin/env python3
"""Exploratory author-reference comparison; never a paper-reproduction oracle.

Author software stays outside the source tree. All implementations read the same
saved float32 input, widened to double for Octave without regenerating noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import statistics

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(args):
    from PIL import Image

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for image_index, name in enumerate(args.images):
        source = Path(args.reference) / "data" / (name + ".png")
        full = np.asarray(Image.open(source), dtype=np.uint8)
        if full.ndim != 2:
            raise ValueError("Expected author's grayscale fixture")
        size = args.size
        if min(full.shape) < size:
            raise ValueError("Fixture smaller than requested crop")
        top, left = (full.shape[0] - size) // 2, (full.shape[1] - size) // 2
        clean = (full[top:top + size, left:left + size].astype(np.float64) / 255).astype('<f4')
        clean_path = out / (name + "-clean.f32")
        clean.tofile(clean_path)
        for sigma in args.sigmas:
            seed = 20260908 + image_index * 1000 + int(sigma)
            noise = np.random.Generator(np.random.PCG64(seed)).standard_normal(clean.shape)
            noisy = (clean.astype(np.float64) + noise * (sigma / 255)).astype('<f4')
            name_case = f"{name}-{size}-s{sigma:g}"
            noisy_path = out / (name_case + "-noisy.f32")
            noisy.tofile(noisy_path)
            cases.append(dict(id=name_case, image=name, width=size, height=size, sigma=sigma,
                              seed=seed, clean=clean_path.name, noisy=noisy_path.name,
                              clean_sha256=sha(clean_path), noisy_sha256=sha(noisy_path),
                              original_sha256=sha(source), crop=[left, top, size, size]))
    save_json(out / "fixtures.json", dict(schema="nss.paper-fixtures.v1", cases=cases,
              input_policy="identical saved float32; unclipped AWGN; no resizing; one noise seed per case"))


def fixture(directory, case_id):
    root = Path(directory)
    entries = json.loads((root / "fixtures.json").read_text())["cases"]
    case = next(row for row in entries if row["id"] == case_id)
    for kind in ("clean", "noisy"):
        if sha(root / case[kind]) != case[kind + "_sha256"]:
            raise ValueError("Fixture hash mismatch")
    return root, case


def plugin(args):
    import vapoursynth as vs

    root, case = fixture(args.fixtures, args.case)
    values = np.fromfile(root / case["noisy"], dtype='<f4').reshape(case["height"], case["width"])
    core = vs.core
    core.num_threads = 1
    core.std.LoadPlugin(path=str(Path(args.plugin).resolve()))
    blank = core.std.BlankClip(width=case["width"], height=case["height"], length=2, format=vs.GRAYS)
    source_fills = 0

    def fill(n, f):
        nonlocal source_fills
        source_fills += 1
        output = f.copy()
        np.asarray(output[0])[:] = values
        return output

    source = core.std.ModifyFrame(blank, blank, fill)
    core.std.SetVideoCache(source, mode=1, fixedsize=2, maxsize=2)
    # Both source frames are fully materialized outside the timing boundary.
    source.get_frame(0)
    source.get_frame(1)
    options = dict(sigma=case["sigma"], block_size=8,
                   block_step=1 if args.variant in ("dense", "matched") else 8, radius=0)
    if args.algorithm == "NCSR":
        options.update(group_size=8, bm_range=7, iters=2, delta=.1)
        if args.variant == "matched":
            sigma=case['sigma']
            block,group,iterations=(6,13,9) if sigma<=15 else (7,16,9) if sigma<=30 else (9,18,12) if sigma<=50 else (8,20,12)
            options.update(block_size=block,group_size=group,bm_range=30,iters=iterations,delta=.02)
    elif args.variant == 'matched':
        raise ValueError('Matched author public parameters are only supported for NCSR')
    node = getattr(core.nss, args.algorithm)(source, **options)
    node.get_frame(0)
    fills_before = source_fills
    start = time.perf_counter()
    frame = node.get_frame(1)
    elapsed = time.perf_counter() - start
    if source_fills != fills_before:
        raise RuntimeError("Source was generated inside the timing boundary")
    np.asarray(frame[0]).astype('<f4').tofile(args.output)
    save_json(args.output + ".json", dict(seconds=elapsed, parameters=options,
              plugin_sha256=sha(args.plugin), output_sha256=sha(args.output),
              timing_boundary="warm get_frame; source materialized; output serialization excluded",
              source_frames_preloaded=2, timed_source_fills=0))


def octave_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def reference(args):
    root, case = fixture(args.fixtures, args.case)
    reference_root = Path(args.reference).resolve()
    output = Path(args.output).resolve()
    prelude = f"""more off; pkg load image; pkg load signal;
f=fopen({octave_quote((root/case['noisy']).resolve())},'rb');
I=reshape(fread(f,Inf,'single=>double'),{case['width']},{case['height']})'; fclose(f);
f=fopen({octave_quote((root/case['clean']).resolve())},'rb');
Io=reshape(fread(f,Inf,'single=>double'),{case['width']},{case['height']})'; fclose(f);
"""
    if args.algorithm == "LSSC":
        base = reference_root / "denoise_iccv09"
        block = 9 if case["sigma"] <= 25 else 12 if case["sigma"] <= 50 else 16
        j1, j2, window = (5, 0, 16) if args.variant == "fast" else (20, 5, 32)
        parameters = dict(block=block, atoms=512, J1=j1, J2=j2, window=window, threads=1)
        expression = f"""
cd({octave_quote(base)}); addpath(pwd);
load('dicts/dict_n{block}.mat');
before_I_hex=num2hex(I(:)); before_D_hex=num2hex(D(:));
before_I_hash=hash('sha256',before_I_hex(:)'); before_D_hash=hash('sha256',before_D_hex(:)');
for timing_pass=0:1
rand('state',0); randn('state',0);
timer=tic;
[result,Dout]=mexDenoise(I,Io,D,{case['sigma']}/255,{block},80,{j1},{j2},(32*{case['sigma']}/255)^2,{window},10000,1,1,80);
elapsed=toc(timer);
after_I_hex=num2hex(I(:)); after_D_hex=num2hex(D(:));
assert(strcmp(before_I_hash,hash('sha256',after_I_hex(:)')));
assert(strcmp(before_D_hash,hash('sha256',after_D_hex(:)')));
end
"""
    else:
        base = reference_root / "NCSR" / "NCSR" / "NCSR_Denoising"
        parameters = dict(outer_override=1 if args.variant == "fast" else None,
                          inner=3, reference="author Parameters_setting; full retains noise-dependent settings")
        expression = f"""
cd({octave_quote(base)}); addpath(fullfile(pwd,'Utilities'));
for timing_pass=0:1
par=Parameters_setting({case['sigma']});
par.I=Io*255; par.nim=I*255;
{'par.K=1;' if args.variant == 'fast' else ''}
timer=tic;
[result,psnr_unused,ssim_unused]=NCSR_Denoising(par);
elapsed=toc(timer);
end
result=result/255;
"""
    driver = output.with_suffix(".m")
    driver.write_text(prelude + expression + f"""
assert(all(size(result)==size(I)) && all(isfinite(result(:))));
f=fopen({octave_quote(output)},'wb'); fwrite(f,result','single'); fclose(f);
f=fopen({octave_quote(str(output)+'.seconds')},'w'); fprintf(f,'%.17g',elapsed); fclose(f);
""")
    started = time.perf_counter()
    result = subprocess.run(["octave", "--no-gui", "--quiet", str(driver)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.timeout)
    output.with_suffix(".log").write_bytes(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Octave exit={result.returncode}: {result.stdout[-3000:].decode(errors='replace')}")
    seconds = float(Path(str(output) + ".seconds").read_text())
    save_json(str(output) + ".json", dict(seconds=seconds, process_seconds=time.perf_counter()-started,
              parameters=parameters, output_sha256=sha(output),
              timing_boundary="second author denoising call after identical warmup; file input and dictionary load excluded; author diagnostics included",
              author_input_dictionary_immutability_checked=args.algorithm=='LSSC',
              implementation="author binary/source under Octave, not verified against MATLAB"))


def filter_gaussian(array):
    x = np.arange(-5, 6, dtype=np.float64)
    kernel = np.exp(-x*x/(2*1.5**2)); kernel /= kernel.sum()
    result = array.astype(np.float64)
    for axis in (0, 1):
        pads = [(0, 0), (0, 0)]; pads[axis] = (5, 5)
        extended = np.pad(result, pads, mode="reflect")
        result = np.zeros_like(result)
        for i, weight in enumerate(kernel):
            slices = [slice(None), slice(None)]; slices[axis] = slice(i, i + array.shape[axis])
            result += weight * extended[tuple(slices)]
    return result


def quality(clean, output):
    a, b = clean.astype(np.float64), output.astype(np.float64)
    error = a-b
    mse = float(np.mean(error**2))
    u, v = filter_gaussian(a), filter_gaussian(b)
    va = filter_gaussian(a*a)-u*u
    vb = filter_gaussian(b*b)-v*v
    cov = filter_gaussian(a*b)-u*v
    ssim = ((2*u*v+.01**2)*(2*cov+.03**2))/((u*u+v*v+.01**2)*(va+vb+.03**2))
    return dict(psnr_db=-10*math.log10(mse) if mse else None,
                ssim=float(np.mean(ssim[5:-5,5:-5])), mse=mse,
                min=float(b.min()), max=float(b.max()))


def cpu_ticks():
    result = {}
    for line in Path('/proc/stat').read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in ('cpu0','cpu1'):
            result[fields[0]] = list(map(int, fields[1:9]))
    return result


def cpu_activity(before, after):
    result = {}
    for cpu, values in before.items():
        delta = [b-a for a,b in zip(values,after[cpu])]
        total = sum(delta)
        result[cpu] = dict(total_ticks=total, busy_fraction=(total-delta[3]-delta[4])/total if total else 0,
                           steal_ticks=delta[7])
    return result


def campaign(args):
    directory = Path(args.out).resolve(); directory.mkdir(parents=True, exist_ok=True)
    fixtures = Path(args.fixtures).resolve()
    cases = json.loads((fixtures / "fixtures.json").read_text())["cases"]
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases]
    script = str(Path(__file__).resolve())
    rows = []
    results_path = directory / "results.jsonl"
    if results_path.exists():
        rows = [json.loads(line) for line in results_path.read_text().splitlines()]
    completed = {(r['case'],r['algorithm'],r['variant'],r['repeat']) for r in rows if r.get('ok')}
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MKL_DYNAMIC="FALSE", OMP_DYNAMIC="FALSE", KMP_DUPLICATE_LIB_OK="true")
    environment['LD_LIBRARY_PATH'] = environment.get('LD_LIBRARY_PATH','') + ':' + str(Path(args.reference).resolve()/"denoise_iccv09"/"libs")
    if os.sched_getaffinity(0) != {0}:
        raise RuntimeError("Campaign must run under taskset -c 0")
    save_json(directory/'environment.json',dict(affinity=sorted(os.sched_getaffinity(0)),
              platform=list(os.uname()),thread_environment={k:environment[k] for k in (
                  'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','MKL_DYNAMIC','OMP_DYNAMIC')},
              octave_version=subprocess.check_output(['octave','--version'],stderr=subprocess.STDOUT,text=True),
              cpuinfo=Path('/proc/cpuinfo').read_text(),
              plugin_sha256=sha(args.plugin),harness_sha256=sha(script)))
    for case in cases:
        clean = np.fromfile(fixtures / case['clean'],dtype='<f4').reshape(case['height'],case['width'])
        noisy = np.fromfile(fixtures / case['noisy'],dtype='<f4').reshape(clean.shape)
        for repeat in range(args.repeat_start,args.repeat_start+args.repeats):
            for algorithm in args.algorithms:
                variants = args.variants if repeat % 2 == 0 else list(reversed(args.variants))
                for variant in variants:
                    if (case['id'],algorithm,variant,repeat) in completed:
                        continue
                    output = directory / f"{case['id']}-{algorithm}-{variant}-r{repeat}.f32"
                    mode = 'plugin' if variant in ('default','dense','matched') else 'reference'
                    command = [sys.executable,script,mode,'--fixtures',str(fixtures),'--case',case['id'],
                               '--algorithm',algorithm,'--variant',variant,'--output',str(output)]
                    if mode=='plugin': command += ['--plugin',str(Path(args.plugin).resolve())]
                    else: command += ['--reference',str(Path(args.reference).resolve()),'--timeout',str(args.timeout)]
                    print(f"START {case['id']} {algorithm} {variant} r{repeat}",flush=True)
                    row = dict(case=case['id'],algorithm=algorithm,variant=variant,repeat=repeat,
                               sigma=case['sigma'],image=case['image'],width=case['width'],height=case['height'],
                               noisy_sha256=case['noisy_sha256'],clean_sha256=case['clean_sha256'],
                               noisy_quality=quality(clean,noisy),output=output.name)
                    try:
                        cpu_before=cpu_ticks()
                        run = subprocess.run(command,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                             timeout=args.timeout+30)
                        row['cpu_activity_during_worker']=cpu_activity(cpu_before,cpu_ticks())
                        output.with_suffix('.worker.log').write_bytes(run.stdout)
                        if run.returncode: raise RuntimeError(run.stdout[-2500:].decode(errors='replace'))
                        pixels = np.fromfile(output,dtype='<f4').reshape(clean.shape)
                        if not np.isfinite(pixels).all(): raise ValueError('nonfinite output')
                        row.update(ok=True,quality=quality(clean,pixels),**json.loads(Path(str(output)+'.json').read_text()))
                    except Exception as error:
                        row.update(ok=False,error=str(error))
                    with results_path.open('a') as stream: stream.write(json.dumps(row,allow_nan=False)+'\n')
                    rows.append(row)
                    print(json.dumps(row,allow_nan=False),flush=True)
    save_json(directory/'results.json',dict(schema='nss.paper-comparison.v1',rows=rows,
              ssim_policy='Gaussian 11x11 sigma=1.5 population covariance; crop 5 pixel border; data_range=1',
              caveats=['author source/binary executed in Octave; MATLAB parity unverified',
                       'implementation runtime ratios do not predict a C++ port runtime',
                       'single saved noise realization per image/sigma; no statistical quality confidence interval']))


def report(args):
    from PIL import Image, ImageDraw, ImageFont

    root = Path(args.results).resolve()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    fixture_root = Path(args.fixtures).resolve()
    fixtures = json.loads((fixture_root/'fixtures.json').read_text())['cases']
    data = json.loads((root/'results.json').read_text())
    rows = [row for row in data['rows'] if row['ok']]
    original_timing_flags = [dict(case=r['case'], algorithm=r['algorithm'], variant=r['variant'],
        seconds=r['seconds'], cpu=r['cpu_activity_during_worker']) for r in rows
        if r['cpu_activity_during_worker']['cpu1']['busy_fraction'] > .01
        or any(c['steal_ticks'] for c in r['cpu_activity_during_worker'].values())]
    timing_replacements = []
    if args.timing_repair:
        repair_root = Path(args.timing_repair).resolve()
        repairs = json.loads((repair_root/'results.json').read_text())['rows']
        for replacement in repairs:
            assert replacement['ok']
            assert sha(repair_root/replacement['output']) == replacement['output_sha256']
            assert replacement['cpu_activity_during_worker']['cpu1']['busy_fraction'] <= .01
            assert not any(c['steal_ticks'] for c in replacement['cpu_activity_during_worker'].values())
            match = [r for r in rows if all(r[k] == replacement[k] for k in ('case','algorithm','variant','repeat'))]
            assert len(match) == 1
            row = match[0]
            for field in ('output_sha256','noisy_sha256','clean_sha256','quality'):
                assert row[field] == replacement[field], f'Timing repair changed {field}'
            timing_replacements.append(dict(case=row['case'], algorithm=row['algorithm'], variant=row['variant'],
                original_seconds=row['seconds'], replacement_seconds=replacement['seconds'],
                output_sha256=row['output_sha256'], source=str(repair_root/'results.json')))
            row['seconds'] = replacement['seconds']
            row['cpu_activity_during_worker'] = replacement['cpu_activity_during_worker']
    remaining_timing_flags = [dict(case=r['case'], algorithm=r['algorithm'], variant=r['variant']) for r in rows
        if r['cpu_activity_during_worker']['cpu1']['busy_fraction'] > .01
        or any(c['steal_ticks'] for c in r['cpu_activity_during_worker'].values())]
    by_case = {}
    for row in rows:
        path = root/row['output']
        if sha(path) != row['output_sha256']:
            raise ValueError('Output hash mismatch during reporting')
        froot,case = fixture(fixture_root,row['case'])
        if row['noisy_sha256'] != case['noisy_sha256']:
            raise ValueError('Reference did not use the identical noisy fixture')
        clean=np.fromfile(froot/case['clean'],dtype='<f4').reshape(case['height'],case['width'])
        pixels=np.fromfile(path,dtype='<f4').reshape(clean.shape)
        recomputed=quality(clean,pixels)
        for metric in ('psnr_db','ssim','mse'):
            if abs(recomputed[metric]-row['quality'][metric])>1e-10:
                raise ValueError(f'Recomputed metric mismatch: {metric}')
        by_case.setdefault((row['case'],row['algorithm'],row['variant']),[]).append(row)
    cells={}
    for key,items in by_case.items():
        # Exact repeated input should give identical outputs. Preserve a failure
        # here instead of hiding nondeterminism behind averaged image metrics.
        hashes={r['output_sha256'] for r in items}
        cells[key]=dict(seconds=statistics.median(r['seconds'] for r in items),
                        samples=len(items),hashes_equal=len(hashes)==1,
                        quality=items[0]['quality'],row=items[0],
                        timing_min=min(r['seconds'] for r in items),timing_max=max(r['seconds'] for r in items))
    variants=('default','dense','fast','full')
    labels={'default':'当前默认','dense':'当前 step=1','fast':'参考版减预算','full':'参考版完整配置'}
    expected={(c['id'],a,v) for c in fixtures for a in ('LSSC','NCSR') for v in variants}
    missing=sorted(expected-cells.keys())
    aggregate=[]
    for algorithm in ('LSSC','NCSR'):
        for sigma in sorted({c['sigma'] for c in fixtures}):
            cases=[c for c in fixtures if c['sigma']==sigma and all((c['id'],algorithm,v) in cells for v in variants)]
            if not cases: continue
            for variant in variants:
                selected=[cells[(c['id'],algorithm,variant)] for c in cases]
                full=[cells[(c['id'],algorithm,'full')] for c in cases]
                default=[cells[(c['id'],algorithm,'default')] for c in cases]
                aggregate.append(dict(algorithm=algorithm,sigma=sigma,variant=variant,cases=len(cases),
                     mean_psnr_db=statistics.mean(c['quality']['psnr_db'] for c in selected),
                     mean_ssim=statistics.mean(c['quality']['ssim'] for c in selected),
                     mean_seconds=statistics.mean(c['seconds'] for c in selected),
                     mean_psnr_gap_full=statistics.mean(f['quality']['psnr_db']-c['quality']['psnr_db'] for f,c in zip(full,selected)),
                     geometric_time_ratio_default=math.exp(statistics.mean(math.log(c['seconds']/b['seconds']) for c,b in zip(selected,default)))))
    overall=[]
    for algorithm in ('LSSC','NCSR'):
        cases=[c for c in fixtures if all((c['id'],algorithm,v) in cells for v in variants)]
        if not cases: continue
        for variant in variants:
            selected=[cells[(c['id'],algorithm,variant)] for c in cases]
            full=[cells[(c['id'],algorithm,'full')] for c in cases]
            base=[cells[(c['id'],algorithm,'default')] for c in cases]
            gaps=[f['quality']['psnr_db']-c['quality']['psnr_db'] for f,c in zip(full,selected)]
            overall.append(dict(algorithm=algorithm,variant=variant,cases=len(cases),
                mean_psnr_db=statistics.mean(c['quality']['psnr_db'] for c in selected),
                mean_ssim=statistics.mean(c['quality']['ssim'] for c in selected),
                mean_seconds=statistics.mean(c['seconds'] for c in selected),
                mean_gap_full_db=statistics.mean(gaps),min_gap_full_db=min(gaps),max_gap_full_db=max(gaps),
                geometric_time_ratio_default=math.exp(statistics.mean(math.log(c['seconds']/b['seconds']) for c,b in zip(selected,base)))))
    worse_than_noisy=[]
    for (case,algorithm,variant),cell in cells.items():
        difference=cell['quality']['psnr_db']-cell['row']['noisy_quality']['psnr_db']
        if difference<0:
            worse_than_noisy.append(dict(case=case,algorithm=algorithm,variant=variant,
                psnr_change_db=difference,noisy_psnr_db=cell['row']['noisy_quality']['psnr_db'],
                output_psnr_db=cell['quality']['psnr_db']))
    save_json(out/'summary.json',dict(schema='nss.paper-summary.v1',expected_cells=len(expected),
                 completed_cells=len(cells),missing_cells=missing,aggregate=aggregate,overall=overall,
                 worse_than_noisy=worse_than_noisy,
                 original_timing_flags=original_timing_flags, timing_replacements=timing_replacements,
                 remaining_timing_flags=remaining_timing_flags,
                 repeat_hashes_equal=all(c['hashes_equal'] for c in cells.values()),
                 metric_recomputation_passed=True))
    image_names='、'.join(dict.fromkeys(c['image'] for c in fixtures))
    image_sizes='、'.join(dict.fromkeys(f"{c['width']}×{c['height']}" for c in fixtures))
    lines=['# LSSC / NCSR 作者参考程序对照 · 2026-09-08','',
      f'已完成 {len(cells)}/{len(expected)} 个图像／算法／配置组合。空间灰度，固定输入，单 CPU，源代码哈希见 provenance.json。','',
      '## 口径','',
      f'- 图像为作者包的 {image_names}，按清单取 {image_sizes} 中心区域，不缩放；每个图像／噪声等级一份保存的 AWGN 实现。所有配置使用同一份未裁剪到 [0,1] 的 float32 输入；Octave 只做精确 double 扩宽。',
      '- PSNR 对干净图计算，峰值固定为 1；SSIM 为 Gaussian 11×11、sigma=1.5、总体协方差、去掉 5 像素边框。输出不裁剪。报告端重新读取输出验证 SHA256，并独立重算全部指标。',
      '- 当前实现：冻结 dirty tree、GCC 13.3 Release、默认 dynamic x86 dispatch。源帧预生成，先 warmup，再计时 get_frame。',
      '- 作者参考：LSSC 原始 Linux MEX 二进制和预训练字典；NCSR 作者 MATLAB 源码，均在 Octave 8.4 执行。先完整 warmup 一次，再计时同输入的第二次调用；排除数据读写和预训练字典载入，保留作者函数内部诊断打印及 NCSR 临时图像输出。',
      '- LSSC 快速配置：作者明确建议的 J1=5、J2=0、window=16；完整配置 J1=20、J2=5、window=32。两者均使用包内 512 atoms 字典、单线程。',
      '- NCSR 快速配置：本轮把作者外循环 K 改为 1，保留 3 次内循环；这属于本轮预算变体，并非作者命名的快速版。完整配置保留作者按噪声选择的全部参数和 K。',
      '- Octave 适配仅将 fspecial 的 gauss 缩写展开为 gaussian；LSSC 增加 .mex 符号链接和运行库路径。未验证 Octave 与 MATLAB 的逐像素一致性。',
      '- 耗时倍率只比较这些具体实现；尤其 NCSR 的 Octave 源码耗时不能预测优化后的 C++ 原版耗时。小图探索不构成 1080p 性能门槛或发布准入。',
      '- 全矩阵每个组合一次 warm 测量；时间复测独立归档。跨图均值不是统计置信区间，也不外推自然视频、色度或时域。','',
      '参考来源：[Mairal 等，ICCV 2009](https://www.di.ens.fr/~fbach/iccv09_mairal.pdf)、[LSSC 作者程序与字典](https://lear.inrialpes.fr/people/mairal/resources/denoise_ICCV09.tar.gz)、[Dong 等，TIP 2013 项目与代码](https://www4.comp.polyu.edu.hk/~cslzhang/NCSR.htm)。作者包、适配、当前插件及输入的哈希见归档 provenance.json 和逐项记录。','',
      '## 汇总','',
      '| 算法 | 配置 | 平均 PSNR dB | 平均 SSIM | 平均秒/帧 | 相对当前默认耗时几何均值 | 与完整参考的平均 PSNR 差 |',
      '|---|---|---:|---:|---:|---:|---:|']
    if timing_replacements or original_timing_flags:
        lines[lines.index('## 汇总'):lines.index('## 汇总')] = [
            '## 计时环境审计','',
            f'原始 {len(original_timing_flags)} 项 CPU 1 活动超过 1% 或存在 steal；共 {len(timing_replacements)} 项计时采用顺序补跑，补跑输入和输出 SHA256 与原始项完全相同。原始数据未改写。当前剩余计时标记 {len(remaining_timing_flags)} 项。',
            '旧快照的自动 apt-daily-upgrade 在主矩阵期间升级过部分系统包；数学运行库、编译器版本未变且未重启。后台作业结束后暂停该临时 VM 的 apt timers 再补跑。CPU 活动取完整 worker 区间而非毫秒级计时区间，本轮仅为探索性运行时间，不是严格固定 OS 的性能准入。','']
    for r in overall:
        lines.append(f"| {r['algorithm']} | {labels[r['variant']]} | {r['mean_psnr_db']:.3f} | {r['mean_ssim']:.4f} | {r['mean_seconds']:.5f} | {r['geometric_time_ratio_default']:.2f}× | {r['mean_gap_full_db']:.3f} dB |")
    lines += ['','## 输出 PSNR 低于噪声输入的样本','']
    if worse_than_noisy:
        lines += ['| 图像 | 算法 | 配置 | 噪声输入 PSNR | 输出 PSNR | 变化 dB |','|---|---|---|---:|---:|---:|']
        for r in worse_than_noisy:
            lines.append(f"| {r['case']} | {r['algorithm']} | {labels[r['variant']]} | {r['noisy_psnr_db']:.3f} | {r['output_psnr_db']:.3f} | {r['psnr_change_db']:.3f} |")
    else:lines.append('本组样本未观察到输出 PSNR 低于噪声输入。')
    lines += ['','## 按噪声等级（同档图像平均）','',
      '| 算法 | sigma | 配置 | PSNR dB | SSIM | 秒/帧 | 完整参考减本配置 dB |','|---|---:|---|---:|---:|---:|---:|']
    for r in aggregate:
        lines.append(f"| {r['algorithm']} | {r['sigma']:g} | {labels[r['variant']]} | {r['mean_psnr_db']:.3f} | {r['mean_ssim']:.4f} | {r['mean_seconds']:.5f} | {r['mean_psnr_gap_full']:.3f} |")
    lines += ['','## 逐图对照','',
      '| 图像 | 算法 | 配置 | PSNR dB | SSIM | 秒/帧 |','|---|---|---|---:|---:|---:|']
    for c in fixtures:
        for a in ('LSSC','NCSR'):
            for v in variants:
                if (c['id'],a,v) not in cells:continue
                r=cells[(c['id'],a,v)]
                lines.append(f"| {c['id']} | {a} | {labels[v]} | {r['quality']['psnr_db']:.3f} | {r['quality']['ssim']:.4f} | {r['seconds']:.5f} |")
    if missing:lines += ['','未完成组合：',*['- '+str(m) for m in missing]]
    # Visualization uses a fixed [0,1] display mapping. Metrics above use raw
    # unclipped values. Each cell is enlarged with nearest-neighbor sampling.
    for sigma in (25,50):
        cases=[c for c in fixtures if c['sigma']==sigma]
        if not cases:continue
        tile=256; label_h=46; left=132; gap=8
        sheet=Image.new('RGB',(left+6*(tile+gap),len(cases)*2*(tile+label_h+gap)+34),'#f2f2f2')
        draw=ImageDraw.Draw(sheet)
        try:font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',15)
        except OSError:font=ImageFont.load_default()
        for ci,c in enumerate(cases):
            shape=(c['height'],c['width'])
            clean=np.fromfile(fixture_root/c['clean'],dtype='<f4').reshape(shape)
            noisy=np.fromfile(fixture_root/c['noisy'],dtype='<f4').reshape(shape)
            for ai,a in enumerate(('LSSC','NCSR')):
                y=34+(ci*2+ai)*(tile+label_h+gap)
                draw.text((8,y+label_h),f"{c['image']}\n{a}\nsigma={sigma}",fill='black',font=font)
                display=[('Clean',clean,None),('Noisy',noisy,quality(clean,noisy)['psnr_db'])]
                for v in variants:
                    r=cells.get((c['id'],a,v))
                    pixels=np.fromfile(root/r['row']['output'],dtype='<f4').reshape(shape) if r else np.zeros(shape,dtype=np.float32)
                    display.append(({'default':'Current default','dense':'Current step=1','fast':'Reference reduced','full':'Reference full'}[v],pixels,r['quality']['psnr_db'] if r else None))
                for col,(label,pixels,psnr) in enumerate(display):
                    x=left+col*(tile+gap)
                    draw.text((x,y),label+(f'\n{psnr:.2f} dB' if psnr is not None else ''),fill='black',font=font)
                    picture=Image.fromarray(np.rint(np.clip(pixels,0,1)*255).astype(np.uint8)).resize((tile,tile),Image.Resampling.NEAREST).convert('RGB')
                    sheet.paste(picture,(x,y+label_h))
        filename=f'comparison-s{sigma}.png';sheet.save(out/filename)
        lines += ['',f'## sigma={sigma} 可视对照','',f'![sigma {sigma} comparison]({filename})']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(cells=len(cells),missing=len(missing),overall=overall),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--reference',required=True); p.add_argument('--out',required=True)
    p.add_argument('--size',type=int,default=128); p.add_argument('--sigmas',type=float,nargs='+',default=[5,15,25,50])
    p.add_argument('--images',nargs='+',default=['house','barbara','peppers256'])
    for mode in ('plugin','reference'):
        p=sub.add_parser(mode)
        p.add_argument('--fixtures',required=True);p.add_argument('--case',required=True)
        p.add_argument('--algorithm',choices=['LSSC','NCSR'],required=True)
        p.add_argument('--variant',choices=['default','dense','matched','fast','full'],required=True)
        p.add_argument('--output',required=True)
        if mode=='plugin': p.add_argument('--plugin',required=True)
        else:
            p.add_argument('--reference',required=True);p.add_argument('--timeout',type=int,default=600)
    p=sub.add_parser('campaign')
    for name in ('fixtures','out','reference','plugin'):p.add_argument('--'+name,required=True)
    p.add_argument('--cases',nargs='+');p.add_argument('--algorithms',nargs='+',default=['LSSC','NCSR'])
    p.add_argument('--variants',nargs='+',default=['default','dense','fast','full'])
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--timeout',type=int,default=600)
    p.add_argument('--repeat-start',type=int,default=0)
    p=sub.add_parser('report')
    for name in ('results','fixtures','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--timing-repair', help='Overlay verified identical-output idle rerun timings; preserve original rows on disk')
    args=parser.parse_args()
    globals()[args.mode](args)


if __name__=='__main__':main()
