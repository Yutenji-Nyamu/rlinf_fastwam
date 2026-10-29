import json,os,subprocess
from pathlib import Path
D=Path('/home/chenyiteng/deployment-20261029/single-extend400-sz1-g67');R=D/'source'
P='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
env=dict(os.environ,PYTHONPATH=str(R),CUDA_VISIBLE_DEVICES='')
tests=subprocess.run([P,'-B','-m','pytest','-q','tools/pi05_distillation/test_resume_extension.py','tools/pi05_distillation/test_single_formal.py'],cwd=R,env=env,text=True,capture_output=True)
(D/'cpu-tests.log').write_text(tests.stdout+tests.stderr)
print(tests.stdout,tests.stderr)
assert tests.returncode==0
entry=subprocess.run([P,'-B','tools/pi05_distillation/train.py','--help'],cwd=R,env=env,text=True,capture_output=True)
assert entry.returncode==0 and '--extend-completed-run' in entry.stdout,entry.stderr[-2000:]
compile_result=subprocess.run([P,'-B','-m','py_compile','tools/pi05_distillation/single_formal_queue.py'],cwd=R,env=env,text=True,capture_output=True)
assert compile_result.returncode==0,compile_result.stderr
(D/'readiness.json').write_text(json.dumps({'tests_passed':True,'entry_help_passed':True,'queue_compiled':True,'gpu_contexts_used':False}))
print('Training CLI and queue passed CPU checks; no GPU context created')
