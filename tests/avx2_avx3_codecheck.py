import collections,hashlib,json,re,subprocess,sys
from pathlib import Path
out=Path(sys.argv[3]);out.mkdir(exist_ok=False)
def scan(path,side):
 text=subprocess.check_output(['objdump','-d','-C','--no-show-raw-insn',path],text=True);(out/(side+'.asm')).write_text(text)
 functions={};key=None
 for line in text.splitlines():
  m=re.match(r'^[0-9a-f]+ <(.+)>:$',line)
  if m:
   name=m[1];key=name.split('(')[0] if name.startswith('nss::N_AVX3::') else None
   if key: functions.setdefault(key,collections.Counter())
  elif key:
   m=re.match(r'\s*[0-9a-f]+:\s+(v\w+)\s',line)
   if m:functions[key][m[1]]+=1
 return functions
a=scan(sys.argv[1],'baseline');b=scan(sys.argv[2],'candidate');common=set(a)&set(b)
r=dict(scope='AVX3 vector instruction histograms; branch/call relocation and wrapper ABI changes are assessed separately by numeric and whole-filter regressions.',common_functions=len(common),identical_histograms=sum(a[k]==b[k] for k in common),changed={k:dict(baseline=a[k],candidate=b[k]) for k in sorted(common) if a[k]!=b[k]},hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sys.argv[1:3]})
(out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='changed'}))
