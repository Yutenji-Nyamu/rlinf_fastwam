"""Server CPU-only: actual concurrent torch.compile forward/backward on tmpfs."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def child(root):
    os.environ.update(CUDA_VISIBLE_DEVICES='',TORCHINDUCTOR_CACHE_DIR=str(root/'inductor'),
                      TRITON_CACHE_DIR=str(root/'triton'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import torch
    assert not torch.cuda.is_initialized()
    torch.manual_seed(123)
    def fn(x):return ((x.sin()+x.cos())*x).sum()
    compiled=torch.compile(fn,fullgraph=True)
    x=torch.randn(16,128,dtype=torch.float64,requires_grad=True)
    y=x.detach().clone().requires_grad_(True)
    a=compiled(x);a.backward();b=fn(y);b.backward()
    assert torch.allclose(a,b,atol=1e-8,rtol=1e-8)
    assert torch.allclose(x.grad,y.grad,atol=1e-8,rtol=1e-8)
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(pid=os.getpid(),torch=torch.__version__,forward_backward_match=True,
         cache=str(root),cuda_initialized=False)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child',type=Path);args=parser.parse_args()
    assert os.getuid()==20001
    if args.child:child(args.child);return
    root=Path('/dev/shm/chenyiteng-wan-goal-cpu-cache-20261001-v1');assert not root.exists()
    root.mkdir(mode=0o700)
    started=time.time()
    children=[subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--child',str(root)],
              stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
              env={**os.environ,'CUDA_VISIBLE_DEVICES':''}) for _ in range(2)]
    rows=[]
    for process in children:
        out,err=process.communicate(timeout=150)
        assert process.returncode==0, err[-3000:]
        rows.append(json.loads(out.strip().splitlines()[-1]))
    files=[p for p in root.rglob('*') if p.is_file()]
    report=dict(time=time.time(),ok=True,seconds=time.time()-started,children=rows,
                cache_files=len(files),cache_bytes=sum(p.stat().st_size for p in files),
                scope='Actual two-process CPU compilation/cache import and matching gradients; no GPU smoke claim')
    destination=Path('/data/chenyiteng/projects/wan-goal-sz3/logs/tmpfs-compile-check-20261001.json')
    with destination.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report))

if __name__=='__main__':main()
