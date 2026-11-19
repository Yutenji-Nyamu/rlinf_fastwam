import json,sys
from pathlib import Path
def evals(folder,rounds):
    p=Path(folder)/'evaluations.jsonl'
    if not p.is_file():return []
    out={}
    for line in p.read_text().splitlines():
        r=json.loads(line);i=r['round'];m=r['metrics']
        if r['phase']=='formal' and i in rounds:
            n=int(m['num_trajectories'])
            out[i]={'round':i,'successes':round(m['success_once']*n),'episodes':n}
    return [out[k] for k in sorted(out)]
if __name__=='__main__':
    base,pilot,dst=map(Path,sys.argv[1:4])
    a=evals(base,(24,26,28,30));b=evals(pilot,(32,34,36,38))
    dst.write_text(json.dumps({'baseline':a,'pilot':b,'all_four_observed':len(b)==4},indent=2))

