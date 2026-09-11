#!/usr/bin/env python3
"""Isolated Linux/Octave black-box probes of the external ICCV author binary.

No reference binary, dictionary, or source is bundled. Zero learning budgets are
only requests: returned dictionary changes are measured, never assumed absent.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
from scipy.io import loadmat, savemat

from benchmark import cpu_delta, cpu_ticks, eligible
from run import read_image, sha


def quote(value):
    return "'"+str(value).replace("'","''")+"'"


def driver(root, temporary, output, *, sigma, block, j1, j2, window, threshold, warmups, repeats):
    """I is also the diagnostic Io; no clean image enters the black-box call."""
    return f"""more off;
cd({quote(root)}); addpath({quote(temporary)});
load({quote(Path(temporary)/'inputs.mat')});
Ih=hash('sha256',num2hex(I(:))(:)'); Dh=hash('sha256',num2hex(D(:))(:)');
times=zeros({repeats},1); changes=zeros({repeats},1);
for pass=1:{warmups+repeats}
  rand('state',0); randn('state',0);
  fprintf('NSS_PASS_BEGIN %d\\n',pass);
  tic;
  [result,Dout]=mexDenoise(I,I,D,{sigma:.17g},{block},80,{j1},{j2},{threshold:.17g},{window},10000,1,1,80);
  elapsed=toc;
  fprintf('NSS_PASS_END %d %.17g\\n',pass,elapsed);
  assert(strcmp(Ih,hash('sha256',num2hex(I(:))(:)')));
  assert(strcmp(Dh,hash('sha256',num2hex(D(:))(:)')));
  assert(all(size(result)==size(I)) && all(isfinite(result(:))));
  if pass>{warmups}
    index=pass-{warmups}; times(index)=elapsed;
    changes(index)=max(abs(Dout(:)-D(:)));
    f=fopen(sprintf({quote(str(Path(output)/'output-%d.f64'))},index),'wb');
    fwrite(f,result','double'); fclose(f);
  end
end
save('-mat7-binary',{quote(Path(output)/'timing.mat')},'times','changes');
"""


def run_probe(image, dictionary, root, output, *, sigma, j1=0, j2=0, window=32,
              threshold=None, warmups=1, repeats=3, timeout=600):
    root,output=Path(root).resolve(),Path(output).resolve()
    image=np.asarray(image,dtype=np.float64)
    dictionary=np.asarray(dictionary,dtype=np.float64)
    if image.ndim!=2 or dictionary.ndim!=2 or not np.isfinite(image).all() or not np.isfinite(dictionary).all():
        raise ValueError('finite image and dictionary matrices required')
    block=int(np.sqrt(dictionary.shape[0]))
    if (block*block!=dictionary.shape[0] or min(image.shape)<block or sigma<0 or not np.isfinite(sigma)
            or min(j1,j2,warmups)<0 or repeats<1 or window<1):
        raise ValueError('invalid author probe parameters')
    threshold=(32*sigma)**2 if threshold is None else threshold
    if not np.isfinite(threshold) or threshold<0:
        raise ValueError('invalid matching threshold')
    binary=root/'mexDenoise.mexa64'
    if not binary.is_file():
        raise FileNotFoundError(binary)
    output.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='nss-lssc-author-probe-') as temporary:
        temporary=Path(temporary)
        (temporary/'mexDenoise.mex').symlink_to(binary)
        savemat(temporary/'inputs.mat',{'I':image,'D':dictionary})
        script=driver(root,temporary,output,sigma=sigma,block=block,j1=j1,j2=j2,window=window,
                      threshold=threshold,warmups=warmups,repeats=repeats)
        (output/'driver.m').write_text(script)
        before=cpu_ticks()
        try:
            completed=subprocess.run(['octave','--no-gui','--quiet',str(output/'driver.m')],
                                     capture_output=True,text=True,timeout=timeout)
        except subprocess.TimeoutExpired as error:
            captured=error.stdout or b''
            if isinstance(captured,bytes):
                captured=captured.decode(errors='replace')
            (output/'author.log').write_text(captured+'\nAUTHOR PROBE TIMEOUT\n')
            raise
        activity=cpu_delta(before,cpu_ticks())
    (output/'author.log').write_text(completed.stdout+completed.stderr)
    if completed.returncode:
        raise RuntimeError(f'author process exit {completed.returncode}: {completed.stderr[-2000:]}')
    timing=loadmat(output/'timing.mat')
    records=[]
    for index in range(repeats):
        path=output/f'output-{index+1}.f64'
        pixels=np.fromfile(path,dtype='<f8').reshape(image.shape)
        if not np.isfinite(pixels).all():
            raise ValueError('nonfinite author output')
        records.append({'seconds':float(timing['times'].ravel()[index]),'output_sha256':sha(path),
                        'returned_dictionary_max_abs_change':float(timing['changes'].ravel()[index])})
    counters={name:[int(v) for v in re.findall(r'\b'+name+r'\s*:\s*(\d+)',completed.stdout)]
              for name in ('num_groups','numPatches','Mcount')}
    report={'schema':'nss.lssc-author-probe.v1','parameters':{'sigma':sigma,'block':block,'j1':j1,'j2':j2,
        'window':window,'threshold_argument':threshold,'warmups':warmups,'repeats':repeats},
        'records':records,'counters':counters,'binary_sha256':sha(binary),'cpu_activity':activity,
        'input_shape':list(image.shape),'input_float64_sha256':hashlib.sha256(image.astype('<f8').tobytes()).hexdigest(),
        'dictionary_float64_sha256':hashlib.sha256(dictionary.astype('<f8').tobytes(order='F')).hexdigest(),
        'harness_sha256':sha(__file__),
        'idle_sibling_verified':eligible(activity),'input_and_dictionary_immutability_checked':True,
        'repeat_output_hashes_equal':len({r['output_sha256'] for r in records})==1,
        'zero_learning_is_verified':j1==j2==0 and all(r['returned_dictionary_max_abs_change']==0 for r in records),
        'timing_boundary':'warm external author calls; data and dictionary load excluded; whole-process CPU monitor',
        'matlab_equivalence_verified':False}
    (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--dictionary',required=True)
    parser.add_argument('--author-root',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--width',type=int)
    parser.add_argument('--height',type=int)
    parser.add_argument('--sigma',required=True,type=float,help='8-bit units')
    parser.add_argument('--j1',type=int,default=0)
    parser.add_argument('--j2',type=int,default=0)
    parser.add_argument('--window',type=int,default=32)
    parser.add_argument('--threshold',type=float)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    image=read_image(args.input,args.width,args.height)
    dictionary=loadmat(args.dictionary,variable_names=['D'])['D']
    print(json.dumps(run_probe(image,dictionary,args.author_root,args.out,sigma=args.sigma/255,
        j1=args.j1,j2=args.j2,window=args.window,threshold=args.threshold,repeats=args.repeats),indent=2))
