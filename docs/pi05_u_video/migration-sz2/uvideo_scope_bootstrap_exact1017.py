from pathlib import Path
D=Path('/data/chenyiteng/deployment-20261017/pi05-u-video-sz2-v2');P=Path('/data/chenyiteng/deployment-20261014/grpo-u-pillbottle-tau123-v1/scope-tau2/bootstrap/sitecustomize.py')
s=P.read_text().replace('if path:',"if path and os.environ.get('CUDA_VISIBLE_DEVICES'):")
(D/'bootstrap/sitecustomize.py').write_text(s)
print('EXISTING_SCOPE_FAIL_CLOSED_CPU_SKIPPED')
