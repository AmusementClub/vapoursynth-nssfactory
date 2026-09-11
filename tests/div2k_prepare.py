#!/usr/bin/env python3
"""Select official DIV2K HR members by HTTP range, and freeze textured RGB ROIs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import struct
import urllib.request
import zipfile
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from paper_compare import save_json, sha

OFFICIAL='https://data.vision.ee.ethz.ch/cvl/DIV2K/'


def previews(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        names=sorted(n for n in archive.namelist() if n.endswith('.png'))
        for group in range(0,len(names),25):
            sheet=Image.new('RGB',(5*260,5*208),'#f4f4f4');draw=ImageDraw.Draw(sheet)
            font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',18)
            for i,name in enumerate(names[group:group+25]):
                pic=Image.open(io.BytesIO(archive.read(name))).convert('RGB')
                pic.thumbnail((250,178))
                x=(i%5)*260;y=(i//5)*208
                draw.text((x+5,y+3),Path(name).name,fill='black',font=font)
                sheet.paste(pic,(x+5,y+28))
            sheet.save(out/f'preview-{group+1:03d}-{min(group+25,len(names)):03d}.jpg',quality=95)


def get_range(url,start,end):
    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(req,timeout=45) as response:
        if response.status!=206: raise RuntimeError('Server did not honor bounded Range request')
        data=response.read(end-start+2)
        assert len(data)==end-start+1
        assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
        return data


def download(args):
    url=OFFICIAL+'DIV2K_valid_HR.zip'
    req=urllib.request.Request(url,method='HEAD')
    with urllib.request.urlopen(req,timeout=30) as response:
        size=int(response.headers['Content-Length']);etag=response.headers.get('ETag')
    tail=get_range(url,size-65557,size-1)
    pos=tail.rfind(b'PK\x05\x06');assert pos>=0
    record=list(struct.unpack('<4s4H2LH',tail[pos:pos+22]))
    central_size,central_offset=record[5:7]
    assert record[1]==record[2]==0 and record[3]==record[4] and record[4]<65535
    central=get_range(url,central_offset,central_offset+central_size-1)
    record[6]=0
    index=zipfile.ZipFile(io.BytesIO(central+struct.pack('<4s4H2LH',*record)))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    selected=[index.getinfo(f'DIV2K_valid_HR/{int(i):04d}.png') for i in args.ids]
    def extract(info):
        header=get_range(url,info.header_offset,info.header_offset+29)
        values=struct.unpack('<4s5H3L2H',header);assert values[0]==b'PK\x03\x04'
        assert not values[2]&1 and info.compress_type in (0,8)
        offset=info.header_offset+30+values[-2]+values[-1]
        payload=get_range(url,offset,offset+info.compress_size-1)
        pixels=zlib.decompress(payload,-15) if info.compress_type==8 else payload
        assert len(pixels)==info.file_size and zlib.crc32(pixels)==info.CRC
        path=out/Path(info.filename).name
        if path.exists(): assert path.read_bytes()==pixels
        else:path.write_bytes(pixels)
        return dict(file=path.name,member=info.filename,sha256=sha(path),crc32=f'{info.CRC:08x}',
            uncompressed_bytes=info.file_size,compressed_bytes=info.compress_size,header_offset=info.header_offset)
    with ThreadPoolExecutor(max_workers=3) as pool:members=list(pool.map(extract,selected))
    save_json(out/'download-provenance.json',dict(url=url,archive_bytes=size,etag=etag,members=members,
        license='Academic research only; copyright remains with original image owners',
        source_page=OFFICIAL,method='Official ZIP central directory and selected members via verified HTTP Range; ZIP CRC32 and per-image SHA256'))
    print(json.dumps(members,indent=2))


def prepare(args):
    source=Path(args.source);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    selections=json.loads(Path(args.selection).read_text())['images']
    cases=[]
    sheet=Image.new('RGB',(3*640,560),'#f1f3f5');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',20)
    for index,selection in enumerate(selections):
        image_id=selection['id'];path=source/f'{image_id:04d}.png'
        original=Image.open(path).convert('RGB');left,top,size,size_h=selection['crop']
        assert size==size_h and size==512
        assert 0<=left and 0<=top and left+size<=original.width and top+size<=original.height
        crop=original.crop((left,top,left+size,top+size))
        clean=(np.asarray(crop,dtype=np.float64)/255).transpose(2,0,1).astype('<f4')
        clean_path=out/f'div2k-{image_id:04d}-clean.f32';clean.tofile(clean_path)
        preview=original.copy();preview.thumbnail((620,470))
        sx=preview.width/original.width;sy=preview.height/original.height
        d=ImageDraw.Draw(preview);d.rectangle((left*sx,top*sy,(left+size)*sx,(top+size)*sy),outline='#ff3030',width=3)
        draw.text((index*640+10,8),f'DIV2K {image_id:04d} | {selection["label"]}',fill='#17212e',font=font)
        draw.text((index*640+10,34),f'ROI x={left}, y={top}, 512x512',fill='#465468',font=font)
        sheet.paste(preview,(index*640+10,68))
        crop.save(out/f'div2k-{image_id:04d}-clean.png')
        for sigma in (5,15,25,50):
            seed=20260908+image_id*100+sigma
            noise=np.random.Generator(np.random.PCG64(seed)).standard_normal(clean.shape)
            noisy=(clean.astype(np.float64)+noise*(sigma/255)).astype('<f4')
            name=f'div2k-{image_id:04d}-512-s{sigma}';noisy_path=out/(name+'-noisy.f32');noisy.tofile(noisy_path)
            cases.append(dict(id=name,image=f'DIV2K-{image_id:04d}',width=size,height=size,channels=3,
                sigma=sigma,seed=seed,clean=clean_path.name,noisy=noisy_path.name,
                clean_sha256=sha(clean_path),noisy_sha256=sha(noisy_path),original_sha256=sha(path),
                original_size=list(original.size),crop=selection['crop'],texture_label=selection['label'],
                selection_reason=selection['reason']))
    sheet.save(out/'selection-overview.png')
    save_json(out/'fixtures.json',dict(schema='nss.paper-fixtures.v1',cases=cases,
        input_policy='Original DIV2K RGB8 encoded values /255; fixed512 ROI no resize; independent channel AWGN; saved unclipped planar float32',
        source_page=OFFICIAL,selection='ROIs fixed by inspecting clean images before any denoiser output'))


def probe(args):
    source=Path(args.fixtures);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'fixtures.json').read_text())
    for case in manifest['cases']:
        clean=np.fromfile(source/case['clean'],dtype='<f4').reshape(case['channels'],case['height'],case['width'])
        clean.tofile(out/case['clean'])
        case['seed']+=100000000
        noise=np.random.Generator(np.random.PCG64(case['seed'])).standard_normal(clean.shape)
        noisy=(clean.astype(np.float64)+noise*(case['sigma']/255)).astype('<f4')
        noisy.tofile(out/case['noisy']);case['noisy_sha256']=sha(out/case['noisy'])
    manifest['input_policy']='Independent second AWGN realization for paired-output residual-noise calibration only'
    save_json(out/'fixtures.json',manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('previews');s.add_argument('--archive',required=True);s.add_argument('--out',required=True)
    s=sub.add_parser('download');s.add_argument('--ids',type=int,nargs='+',required=True);s.add_argument('--out',required=True)
    s=sub.add_parser('prepare')
    for name in ('source','selection','out'):s.add_argument('--'+name,required=True)
    s=sub.add_parser('probe');s.add_argument('--fixtures',required=True);s.add_argument('--out',required=True)
    args=p.parse_args();globals()[args.mode](args)


if __name__=='__main__':main()
