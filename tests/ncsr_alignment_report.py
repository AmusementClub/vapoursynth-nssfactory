#!/usr/bin/env python3
"""Audit raw NCSR alignment outputs and report paired quality/runtime changes."""
import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np
from paper_compare import fixture, quality, save_json, sha


def eligible(row):
    cpu=row['cpu_activity']
    return cpu['cpu1']['busy_fraction']<=.01 and all(c['steal_ticks']==0 for c in cpu.values())


def report_dataset(directory,fixtures,metrics_cache):
    data=json.loads((directory/'results.json').read_text());rows=data['rows']
    cells={};failures=[];flags=[]
    for row in rows:
        if not row['ok']:
            failures.append(row);continue
        froot,case=fixture(fixtures,row['case'])
        for field in ('noisy_sha256','clean_sha256'):assert row[field]==case[field]
        path=directory/row['output'];assert sha(path)==row['output_sha256']
        assert row['output_sha256']==row['warm_output_sha256'] and row['repeat_hashes_equal']
        assert row['timed_source_fills']==0
        cachekey=(row['output_sha256'],row['clean_sha256'])
        if cachekey not in metrics_cache:
            shape=(case['height'],case['width'])
            clean=np.fromfile(froot/case['clean'],dtype='<f4').reshape(shape)
            pixels=np.fromfile(path,dtype='<f4').reshape(shape)
            metrics_cache[cachekey]=quality(clean,pixels)
        for metric in ('psnr_db','ssim','mse'):
            assert abs(metrics_cache[cachekey][metric]-row['quality'][metric])<1e-10
        cells.setdefault((row['case'],row['variant']),[]).append(row)
        if not eligible(row):flags.append(dict(case=row['case'],variant=row['variant'],repeat=row['repeat'],cpu=row['cpu_activity']))
    for items in cells.values():assert len({r['output_sha256'] for r in items})==1
    for (case,variant),items in cells.items():
        if variant=='hook_control' and (case,'baseline') in cells:
            assert items[0]['output_sha256']==cells[(case,'baseline')][0]['output_sha256']
    summaries=[]
    for variant in dict.fromkeys(r['variant'] for r in rows):
        selected=[(case,items) for (case,v),items in cells.items() if v==variant]
        if not selected:continue
        percase=[]
        for case,items in selected:
            base_variant='baseline'
            if directory.name=='nss-align-beta-dense':base_variant='legacy_step1'
            if directory.name=='nss-align-matched':
                base_variant='legacy_author_budget' if variant in ('D_author_budget','legacy_author_budget') else 'legacy_step1'
            base=cells.get((case,base_variant))
            by_repeat={r['repeat']:r for r in base or []}
            pairs=[(row,by_repeat[row['repeat']]) for row in items if row['repeat'] in by_repeat and eligible(row) and eligible(by_repeat[row['repeat']])]
            ratio=math.exp(statistics.mean(math.log(a['seconds']/b['seconds']) for a,b in pairs)) if pairs else None
            valid=[r for r in items if eligible(r)]
            stages={}
            if valid and valid[0]['traces'][0]:
                for stage in ('training_seconds','matching_seconds','filtering_seconds'):
                    stages[stage]=statistics.median(sum(i[stage] for i in trace['iterations']) for r in valid for trace in r['traces'])
            percase.append(dict(case=case,psnr_db=items[0]['quality']['psnr_db'],ssim=items[0]['quality']['ssim'],
                seconds=statistics.median(r['seconds'] for r in valid) if valid else None,
                psnr_change_db=items[0]['quality']['psnr_db']-base[0]['quality']['psnr_db'] if base else None,
                paired_ratio=ratio,eligible_pairs=len(pairs),eligible_runs=len(valid),total_runs=len(items),
                below_noisy=items[0]['quality']['psnr_db']<items[0]['noisy_quality']['psnr_db'],stages=stages,
                output=items[0]['output'],baseline_variant=base_variant))
        # Quality can always be reported; a paired timing claim requires >=2
        # eligible pairs for every compared case, not merely one pooled median.
        timing_claim=all(c['eligible_pairs']>=2 for c in percase)
        summaries.append(dict(variant=variant,cases=len(percase),
            mean_psnr_db=statistics.mean(c['psnr_db'] for c in percase),mean_ssim=statistics.mean(c['ssim'] for c in percase),
            mean_seconds=statistics.mean(c['seconds'] for c in percase) if all(c['seconds'] is not None for c in percase) else None,
            mean_psnr_change_db=statistics.mean(c['psnr_change_db'] for c in percase) if all(c['psnr_change_db'] is not None for c in percase) else None,
            geometric_paired_ratio=math.exp(statistics.mean(math.log(c['paired_ratio']) for c in percase)) if timing_claim else None,
            timing_claim_supported=timing_claim,per_case=percase))
    result=dict(rows=len(rows),successful_rows=sum(r['ok'] for r in rows),failures=failures,
                all_successful_output_hashes_and_metrics_verified=True,repeat_hashes_equal=True,cpu_flags=flags,summaries=summaries)
    save_json(directory/'audit-summary.json',result)
    return result


def contact_sheet(root,fixtures,source,variants,outname,sigma=25):
    from PIL import Image,ImageDraw,ImageFont
    rows=json.loads((source/'results.json').read_text())['rows']
    lookup={(r['case'],r['variant']):r for r in rows if r['ok']}
    cases=[c for c in json.loads((fixtures/'fixtures.json').read_text())['cases'] if c['sigma']==sigma]
    if not cases:return
    tile=224;gap=10;label_h=48;left=112
    image=Image.new('RGB',(left+(len(variants)+2)*(tile+gap),len(cases)*(tile+label_h+gap)+30),'#f2f2f2')
    draw=ImageDraw.Draw(image)
    try:font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',14)
    except OSError:font=ImageFont.load_default()
    for ci,case in enumerate(cases):
        y=30+ci*(tile+label_h+gap);shape=(case['height'],case['width'])
        clean=np.fromfile(fixtures/case['clean'],dtype='<f4').reshape(shape)
        noisy=np.fromfile(fixtures/case['noisy'],dtype='<f4').reshape(shape)
        display=[('Clean',clean,None),('Noisy',noisy,quality(clean,noisy)['psnr_db'])]
        for variant in variants:
            r=lookup.get((case['id'],variant))
            pixels=np.fromfile(source/r['output'],dtype='<f4').reshape(shape) if r else np.zeros(shape)
            display.append((variant,pixels,r['quality']['psnr_db'] if r else None))
        draw.text((8,y+label_h),f"{case['image']}\nsigma={sigma}",fill='black',font=font)
        for col,(label,pixels,psnr) in enumerate(display):
            x=left+col*(tile+gap)
            draw.text((x,y),label+(f'\n{psnr:.2f} dB' if psnr is not None else ''),fill='black',font=font)
            picture=Image.fromarray(np.rint(np.clip(pixels,0,1)*255).astype(np.uint8))
            image.paste(picture.resize((tile,tile),Image.Resampling.NEAREST).convert('RGB'),(x,y+label_h))
    image.save(root/outname)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--fixtures128',required=True);parser.add_argument('--fixtures256',required=True)
    args=parser.parse_args();root=Path(args.root).resolve();cache={};all_results={}
    for name in ('main','budget','matched','beta-dense','confirm256','fullhd'):
        directory=root/('nss-align-'+name)
        if not (directory/'results.json').exists():continue
        fixtures=Path(args.fixtures128).resolve()
        if name=='confirm256':fixtures=Path(args.fixtures256).resolve()
        if name=='fullhd':fixtures=root/'nss-align-fixtures-hd'
        all_results[name]=report_dataset(directory,fixtures,cache)
    save_json(root/'summary.json',all_results)
    lines=['# NCSR 部分对齐实验 · 2026-09-08','',
        '保持生产滤镜不变；隔离源码副本，单 C4 Spot、CPU0、同一份预生成输入。每个配置三轮交错顺序重复，worker 内先 warmup，再计时完整帧；所有字典训练均计入。',
        'A=统计/噪声/阈值；W=权重修正；B=目标独立中心与目标回填；C=AWB 加分簇完整 PCA 字典和三步复用；D=在 C 上改变预算。不是作者原版的完整复现。',
        '所有原始输出重新验哈希并重算 PSNR/SSIM。计时剔除 CPU1 >1% 或非零 steal 的配对，每项至少两对才报告倍率。质量平均不能替代逐图退化记录。','']
    for name,result in all_results.items():
        lines += [f'## {name}', '',f"成功 {result['successful_rows']}/{result['rows']}；原始 CPU 活动标记 {len(result['cpu_flags'])} 项。",'',
            '| 配置 | 图数 | 平均 PSNR | 平均 SSIM | 平均秒/帧 | 相对对照增益 dB | 配对耗时倍率 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for r in result['summaries']:
            seconds=f"{r['mean_seconds']:.6f}" if r['mean_seconds'] is not None else 'N/A'
            gain=f"{r['mean_psnr_change_db']:+.3f}" if r['mean_psnr_change_db'] is not None else 'N/A'
            ratio=f"{r['geometric_paired_ratio']:.3f}×" if r['geometric_paired_ratio'] is not None else '证据不足'
            lines.append(f"| {r['variant']} | {r['cases']} | {r['mean_psnr_db']:.3f} | {r['mean_ssim']:.4f} | {seconds} | {gain} | {ratio} |")
        lines += ['', '逐图结果：','', '| 图像 | 配置 | PSNR | 相对对照 dB | 秒 | 有效配对 |','|---|---|---:|---:|---:|---:|']
        for r in result['summaries']:
            for c in r['per_case']:
                gain=f"{c['psnr_change_db']:+.3f}" if c['psnr_change_db'] is not None else 'N/A'
                seconds=f"{c['seconds']:.6f}" if c['seconds'] is not None else 'N/A'
                lines.append(f"| {c['case']} | {r['variant']} | {c['psnr_db']:.3f} | {gain} | {seconds} | {c['eligible_pairs']} |")
        if name=='matched':lines += ['', 'matched 的对照分别为相同 step=1 的旧核，或相同 block=7、group=16、range=30、iters=9、delta=.02 的旧核，不是默认预算。']
    if 'main' in all_results:
        contact_sheet(root,Path(args.fixtures128),root/'nss-align-main',['baseline','A','W','B','C'],'models-s25.png')
        contact_sheet(root,Path(args.fixtures128),root/'nss-align-main',['baseline','W','B','C'],'models-s5.png',5)
        lines+=['','![Models sigma25](models-s25.png)','','![Models sigma5](models-s5.png)']
    if 'budget' in all_results:
        contact_sheet(root,Path(args.fixtures128),root/'nss-align-budget',['baseline','C','D_step2','D_group16','D_author_budget'],'budget-s25.png')
        lines+=['','![Budget sigma25](budget-s25.png)']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:dict(rows=v['rows'],failures=len(v['failures']),flags=len(v['cpu_flags'])) for k,v in all_results.items()},indent=2))


if __name__=='__main__':main()
