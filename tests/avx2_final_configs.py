#!/usr/bin/env python3
"""Final permission-routing gates: explicit tested dimensions, no nine-image sweep."""
import argparse,copy,json
from pathlib import Path
from avx2_port_configs import case

def rows():
    result=[case(group=g) for g in (16,32,64)]
    result += [case('wnnm',block=b,group=g,stage='basic') for b,g in ((8,32),(12,16),(16,16))]
    result += [case(a,group=32,stage='basic') for a in ('twsc','ncsr')]
    result += [case(block=12,group=g) for g in (1,8,16,32)]+[case(block=12,radius=1)]
    result += [case(block=16,group=g) for g in (1,8,16,32)]+[case(block=16,radius=1)]
    result += [case('nlh',stage='basic',radius=r) for r in (0,1)]
    result += [case('nlh',block=4,stage='basic'),case('nlh',stage='two_stage')]
    result += [case('mcwnnm',group=8,stage='basic',size=[320,180]),
               dict(name='lssc_default',algorithm='lssc',size=[640,360],frames=2,warmup=1)]
    return result

def write(out):
    out.mkdir(parents=True,exist_ok=True)
    main=rows()
    fallback=[case(block=4,group=16),case(group=8),case('nlh',stage='basic',q=2),case('nlh',stage='basic',group=8)]
    for name,data in [('main',main),('fallback',fallback)]:
        (out/(name+'.json')).write_text(json.dumps(data,indent=2))
    small=copy.deepcopy(main+fallback)
    for r in small:
        r.pop('sample',None);r['size']=[75,49];r['frames']=2
        if 'kwargs' in r:r['kwargs']['bm_range']=3
    (out/'numeric.json').write_text(json.dumps(small,indent=2))
    avx3=[case(group=32),case(block=12),case(block=16,group=1),case(block=16),
          case('nlh',stage='basic'),case('wnnm',block=12,stage='basic'),
          case('mcwnnm',group=8,stage='basic',size=[320,180]),main[-1]]
    (out/'avx3.json').write_text(json.dumps(avx3,indent=2))
    extra=[];images=out/'images';images.mkdir(exist_ok=True)
    # The final plugin also checks real-image LSSC and the larger MCWNNM shape.
    for target in [case(group=32),case(block=12),case(block=16,group=1),case('nlh',stage='two_stage'),main[-2],main[-1]]:
        for name in ('8bit','Ufotable'):
            r=copy.deepcopy(target);r['name']+='_'+name
            w,h=r['size'];src=Path('/opt/nss-c4/samples/gray8')/(name+'.gray8')
            if [w,h]!=[1920,1080]:
                import hashlib,numpy as np
                raw=src.read_bytes();a=np.frombuffer(raw,np.uint8).reshape(1080,1920);x,y=(1920-w)//2,(1080-h)//2
                dst=images/f'{w}x{h}-{name}.gray8';dst.write_bytes(a[y:y+h,x:x+w].copy().tobytes())
                (dst.with_suffix('.json')).write_text(json.dumps(dict(source=str(src),source_sha256=hashlib.sha256(raw).hexdigest(),crop=[x,y,w,h],sha256=hashlib.sha256(dst.read_bytes()).hexdigest()),indent=2))
                src=dst
            r['sample']=str(src);extra.append(r)
    (out/'real-images.json').write_text(json.dumps(extra,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);write(p.parse_args().out)
