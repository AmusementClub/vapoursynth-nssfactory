#!/usr/bin/env python3
"""Install an unchanged native plugin into an isolated prefix and test loading."""
import argparse,hashlib,json,os,platform,shutil,subprocess,sys,tarfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for k in ['plugin','source','gate','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
gate=json.loads((a.gate/'summary.json').read_text())
if not gate['passed'] or gate['backend']['target_name']!='NEON':raise RuntimeError('native NEON gate required')
original=hashlib.sha256(a.plugin.read_bytes()).hexdigest()
if original!=gate['plugin_sha256']:raise RuntimeError('plugin differs from functional gate')
prefix=a.out/'install';library=prefix/'lib/vapoursynth'/a.plugin.name;library.parent.mkdir(parents=True)
shutil.copy2(a.plugin,library);logs=a.out/'checks';logs.mkdir()
report=dict(passed=False,platform=platform.platform(),architecture=platform.machine(),plugin_sha256=original,source_inputs_sha256=gate['source_inputs_sha256'],backend=gate['backend'],checks=[])
def run(name,cmd,env=None,required=True):
 proc=subprocess.run(cmd,capture_output=True,text=True,env=env);(logs/(name+'.log')).write_text(proc.stdout+proc.stderr)
 report['checks'].append(dict(name=name,command=cmd,returncode=proc.returncode))
 if required:proc.check_returncode()
 return proc.stdout+proc.stderr
run('file',['file',str(library)])
if platform.system()=='Darwin':
 run('dependencies',['otool','-L',str(library)]);run('load-commands',['otool','-l',str(library)])
 symbols=run('exports',['nm','-gU',str(library)]);run('signature',['codesign','-dv','--verbose=4',str(library)],required=False)
 scope='Tested on the recorded native macOS/Apple Silicon host. Other macOS versions and notarized distribution are not certified by this bundle.'
else:
 dependencies=run('dependencies',['ldd',str(library)])
 if 'not found' in dependencies:raise RuntimeError('missing runtime dependency')
 symbols=run('exports',['nm','-D','--defined-only',str(library)])
 run('elf',['readelf','-h','-l','-d','--version-info',str(library)])
 scope='Tested on the recorded native Linux AArch64 host with its GCC15 runtime libraries. See checks/dependencies.log and checks/elf.log for dynamic library and symbol-version requirements.'
if 'VapourSynthPluginInit2' not in symbols:raise RuntimeError('required plugin entry point missing')
env=dict(os.environ,NSS_SO=str(library.resolve()))
run('installed-backend',[sys.executable,str((a.source/'tests/test_backend_plugin.py').resolve())],env)
run('installed-eight-algorithms',[sys.executable,str((a.source/'tests/test_plan01_plugin.py').resolve()),'--out',str((logs/'installed-eight-algorithms.json').resolve())],env)
if hashlib.sha256(library.read_bytes()).hexdigest()!=original:raise RuntimeError('installation modified measured binary')
(prefix/'README.txt').write_text('NSSFactory native NEON preview bundle\n\n'+scope+'\n\nRequires a native arm64/AArch64 VapourSynth R75 Python runtime. Copy lib/vapoursynth/'+library.name+' into a user-selected plugin directory, or explicitly load it:\n\nimport vapoursynth as vs\nvs.core.std.LoadPlugin(path="/absolute/path/to/'+library.name+'")\nprint(vs.core.nss.Backend())\n\nThe plugin keeps the existing 32-bit float format and per-algorithm parameter restrictions. SVE/SME/Metal are outside this CPU release. See the accompanying Plan02 release report and support matrix for measured shapes and numerical policy.\n')
shutil.copy2(a.source/'LICENSE',prefix/'LICENSE')
shutil.copy2(a.source/'NOTICE',prefix/'NOTICE')
highway_paths = [value.split('=', 1)[1] for command in gate['commands']
                 for value in command['argv']
                 if value.startswith('-DFETCHCONTENT_SOURCE_DIR_HIGHWAY=')]
if len(set(highway_paths)) != 1:
 raise RuntimeError('one pinned Highway source is required for its license')
shutil.copy2(Path(highway_paths[0])/'LICENSE',prefix/'HIGHWAY-LICENSE')
(prefix/'SHA256SUMS').write_text(original+'  lib/vapoursynth/'+library.name+'\n')
report.update(passed=True,installed_binary_unchanged=True,compatibility_scope=scope)
(a.out/'manifest.json').write_text(json.dumps(report,indent=2))
archive=a.out/(('macos-arm64' if platform.system()=='Darwin' else 'linux-aarch64')+'-neon.tar.gz')
with tarfile.open(archive,'w:gz') as tf:tf.add(prefix,arcname='nssfactory-neon')
(archive.with_suffix(archive.suffix+'.sha256')).write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
print(json.dumps(dict(passed=True,archive=str(archive),plugin_sha256=original,architecture=platform.machine())))
