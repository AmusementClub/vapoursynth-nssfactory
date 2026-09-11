#!/usr/bin/env python3
"""Render and verify oracle-tuned and noise-response-matched RGB comparisons."""
import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from defaults_compare import ALGORITHMS, eligible, plane_quality
from paper_compare import fixture, save_json, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('results','fixtures','probe','out'):p.add_argument('--'+name,required=True)
    args=p.parse_args();root=Path(args.results).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    native=out/'native';native.mkdir(exist_ok=True)
    data=json.loads((root/'results.json').read_text());rows=data['rows'];selected={s['case']:s for s in data['selections']}
    joint=data['schema']=='nss.joint-tuning.v1'
    cells={};inputs={};audits=[]
    for row in rows:
        froot,case=fixture(args.fixtures,row['case']);proot,probe=fixture(args.probe,row['case'])
        shape=(3,case['height'],case['width'])
        clean=np.fromfile(froot/case['clean'],dtype='<f4').reshape(shape)
        noisy=np.fromfile(froot/case['noisy'],dtype='<f4').reshape(shape)
        noisy2=np.fromfile(proot/probe['noisy'],dtype='<f4').reshape(shape)
        assert row['noisy_sha256']==case['noisy_sha256'] and row['timed_source_fills']==0
        assert sha(root/row['output'])==row['output_sha256']
        assert sha(root/row['probe_output'])==row['probe_output_sha256']
        pixels=np.fromfile(root/row['output'],dtype='<f4').reshape(shape)
        pixels2=np.fromfile(root/row['probe_output'],dtype='<f4').reshape(shape)
        metrics=plane_quality(clean,pixels)
        probe_metrics=plane_quality(clean,pixels2)
        for metric in ('psnr_db','ssim','mse'):assert abs(metrics[metric]-row['quality'][metric])<1e-10
        denom=float(np.sum((noisy.astype(np.float64)-noisy2)**2))
        rho=math.sqrt(float(np.sum((pixels.astype(np.float64)-pixels2)**2))/denom)
        assert abs(rho-row['selection']['response'])<1e-10
        if row['mode']=='matched':
            assert (abs(rho-row['selection']['target_response'])<=row['selection']['tolerance'])==row['selection']['matched']
        key=(row['case'],row['algorithm'],row['mode'])
        cells.setdefault(key,[]).append(row)
        if row['repeat']==0:
            audits.append(dict(case=row['case'],algorithm=row['algorithm'],mode=row['mode'],
                primary_psnr_db=metrics['psnr_db'],second_noise_psnr_db=probe_metrics['psnr_db'],response=rho))
        inputs[row['case']]=(case,clean,noisy)
    aggregates={}
    for key,items in cells.items():
        assert len(items)==(1 if joint else 3) and len({r['output_sha256'] for r in items})==1
        if joint:
            assert items[0]['independent_repeat_hash_verified']
            assert items[0]['search_output_sha256']==items[0]['output_sha256']
        good=[r for r in items if eligible(r)]
        aggregates[key]=dict(row=items[0],seconds=statistics.median(r['seconds'] for r in good) if good else None,
            good_repeats=len(good),total_repeats=len(items))
    expected=len(selected)*8*2
    assert len(aggregates)==expected
    font_path='/System/Library/Fonts/Menlo.ttc'
    title_font=ImageFont.truetype(font_path,21);label_font=ImageFont.truetype(font_path,17)
    tile,gap,head,label_h=512,12,70,94
    width=tile*5+gap*6;height=head+2*(tile+label_h+gap)+gap
    def display(array):
        return Image.fromarray(np.rint(np.clip(array.transpose(1,2,0),0,1)*255).astype(np.uint8))
    def sheet(case_id,mode,focus=False):
        case,clean,noisy=inputs[case_id];target=selected[case_id]['target_response']
        picture=Image.new('RGB',(width,height),'#f2f4f7');draw=ImageDraw.Draw(picture)
        text='PSNR-tuned (oracle)' if mode=='best' else f'Noise matched | target response={target:.3f}'
        draw.text((gap,8),f"{case['image']} | injected sigma={case['sigma']:g} | {text}",fill='#142033',font=title_font)
        draw.text((gap,38),('Center256 detail, nearest2x; labels/metrics refer to full512 ROI' if focus else
            'Native512 RGB crop; no resizing; raw-float PSNR/SSIM; same two-noise calibration'),fill='#495a70',font=label_font)
        entries=[('Clean',clean,None),('Noisy',noisy,None)]
        for algorithm in ALGORITHMS:
            c=aggregates[(case_id,algorithm,mode)];r=c['row']
            pixels=np.fromfile(root/r['output'],dtype='<f4').reshape(clean.shape)
            entries.append((algorithm,pixels,c))
        for i,(name,pixels,cell) in enumerate(entries):
            x=gap+(i%5)*(tile+gap);y=head+(i//5)*(tile+label_h+gap)
            draw.text((x,y),name,fill='#142033',font=title_font)
            if cell:
                r=cell['row'];s=r['selection'];metric=r['quality']
                params=r['supplied_parameters'];short=[]
                labels={'block_step':'step','group_size':'group','block_size':'block','adaptive_aggregation':'adapt','residual':'res'}
                for k,v in params.items():short.append(f'{labels.get(k,k)}={v:.4g}' if isinstance(v,(int,float)) else f'{k}={v}')
                draw.text((x,y+24),', '.join(short),fill='#495a70',font=label_font)
                draw.text((x,y+45),f"PSNR {metric['psnr_db']:.2f} dB | SSIM {metric['ssim']:.3f}",fill='#495a70',font=label_font)
                ms=(f"{cell['seconds']:.2f}s" if cell['seconds']>=1 else f"{cell['seconds']*1000:.1f}ms") if cell['seconds'] is not None else 'timing unverified'
                if joint:ms+=' cold'
                suffix=' NOT MATCHED' if mode=='matched' and not s['matched'] else ''
                draw.text((x,y+66),f"response={s['response']:.3f} | {ms}{suffix}",fill='#a02020' if suffix else '#495a70',font=label_font)
            elif name=='Noisy':
                metric=plane_quality(clean,noisy)
                draw.text((x,y+26),f"PSNR {metric['psnr_db']:.2f} dB | response=1.000",fill='#495a70',font=label_font)
            view=display(pixels)
            if not focus:
                path=native/f'{case_id}-{mode}-{name}.png';view.save(path)
                assert np.array_equal(np.asarray(Image.open(path)),np.asarray(view))
            else:view=view.crop((128,128,384,384)).resize((512,512),Image.Resampling.NEAREST)
            picture.paste(view,(x,y+label_h))
        filename=f'{case_id}-{mode}'+('-detail' if focus else '')+'.png';picture.save(out/filename)
        return filename
    lines=['# DIV2K 纹理：参数搜索与近似同降噪强度对照','',
        '每图原生512×512 RGB裁切，注入已保存的独立通道 AWGN。最高PSNR配置使用干净原图挑选，为有界搜索中的最佳已测点，不保证全局最优，也不是可部署的盲调参。',
        '同强度图使用两份独立噪声的响应比 rho=||D(y1)-D(y2)||/||y1-y2||。它估计随机残余噪声及噪声诱发的纹理不稳定性，不是完整感知指标。目标由各算法较优设置的中位值落入共同实测非旁路范围，允许max(0.005,5%目标)偏差。sigma=0旁路不用于配平。',
        '结构参数若有调整，会完整标在每格和原始JSON中；同一结构配置下重新搜索sigma，NLM搜索h。配平图并非每算法最高PSNR点。所有图统一0..1显示映射，不自动增强对比度、锐化或平滑。',
        '候选搜索为单次冷调用，其耗时不作性能比较。最终选中输出重新warmup并测3次，逐字节核验；计时只汇总CPU1活动不超过1%且无steal的重复。精确float输出及第二噪声输出保存在原始结果目录。',
        '来源：[DIV2K官网及学术研究许可](https://data.vision.ee.ethz.ch/cvl/DIV2K/)。引用：Agustsson与Timofte，NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study；Timofte等，NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results。','']
    if joint:
        lines[5]='本轮聚焦三张sigma25纹理图，优先完成公开结构参数搜索及强度配平。每个选中输出单独再运行一次，并与先前搜索输出逐字节核验；最佳与匹配参数完全相同时共用这次实测。图中的cold时间仅是单次过滤调用的成本提示，不是warm配对性能门槛。'
    for case_id in selected:
        case,_,_=inputs[case_id]
        lines += [f'## {case_id}','',f"共同噪声响应目标：{selected[case_id]['target_response']:.4f}。",'',
            '| 算法 | PSNR调优参数 | PSNR | 配平参数 | 配平PSNR | 响应比 | 配平成功 |',
            '|---|---|---:|---|---:|---:|---|']
        for a in ALGORITHMS:
            b=aggregates[(case_id,a,'best')]['row'];m=aggregates[(case_id,a,'matched')]['row']
            lines.append(f"| {a} | {json.dumps(b['supplied_parameters'])} | {b['quality']['psnr_db']:.3f} | {json.dumps(m['supplied_parameters'])} | {m['quality']['psnr_db']:.3f} | {m['selection']['response']:.4f} | {m['selection']['matched']} |")
        for mode in ('matched','best'):
            name=sheet(case_id,mode)
            lines += ['',f'### {"近似同强度" if mode=="matched" else "最高已测PSNR"}','',f'![comparison]({name})']
            if case['sigma']==25:
                detail=sheet(case_id,mode,focus=True)
                lines += ['',f'[中心256纹理放大]({detail})']
    flags=[dict(case=r['case'],algorithm=r['algorithm'],mode=r['mode'],repeat=r['repeat'],cpu=r['cpu_activity_during_worker']) for r in rows if not eligible(r)]
    mismatches=[dict(case=k[0],algorithm=k[1],selection=c['row']['selection']) for k,c in aggregates.items()
        if k[2]=='matched' and not c['row']['selection']['matched']]
    save_json(out/'delivery-audit.json',dict(rows=len(rows),cells=len(aggregates),expected_cells=expected,
        selected_float_hashes_verified=True,metrics_recomputed=True,two_noise_response_recomputed=True,
        triplicate_hashes_equal=None if joint else True,independent_repeat_hashes_verified=joint,
        unique_final_measurements=len({r.get('measurement_id',r['output']) for r in rows}),
        second_noise_metrics=audits,
        timing_policy=data.get('timing_policy','warm triplicate'),native_png_mapping_verified=True,timing_flags=flags,unmatched=mismatches,
        minimum_eligible_repeats=min(c['good_repeats'] for c in aggregates.values())))
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(rows=len(rows),cells=len(aggregates),unmatched=len(mismatches),timing_flags=len(flags)),indent=2))


if __name__=='__main__':main()
