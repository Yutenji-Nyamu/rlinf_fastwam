import csv
import json
from pathlib import Path
import statistics
from helpers import write_csv


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((out/'rounds.csv').open()))
    samples=[]
    with (out/'memory.jsonl').open() as f:
        for line in f:
            try:samples.append(json.loads(line))
            except ValueError:pass
    for row in rows:
        a,b,c,d=(float(row[key]) for key in ['start','pre_replay','post_replay','end'])
        observed=[m['gib'] for m in samples if a<=m['clock']<=b or c<=m['clock']<=d]
        row['peak_card_sampled_gib']=max(observed) if observed else ''
    write_csv(out/'rounds.csv',rows)
    parts=['environment','policy','entropy_weights','update','other_s']
    fig,axes=plt.subplots(1,4,figsize=(12,3.2))
    summary=[]
    for k,ax in zip([1,2,4,8],axes):
        group=[r for r in rows if int(r['K'])==k and r['warmup']=='False']
        assert len(group)==3
        rec={key:statistics.mean(float(r[key]) for r in group) for key in parts+['wall_s','paired_delta_s']}
        rec.update(K=k,incremental_percent=100*rec['paired_delta_s']/rec['wall_s'])
        summary.append(rec)
        wedges,_=ax.pie([rec[p] for p in parts])
        ax.set_title(f"K={k}   {rec['wall_s']:.1f} s")
    fig.legend(wedges,['Environment','Joint inference','Entropy / weights','Update','Other'],loc='lower center',ncol=5)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'figure3.{ext}',bbox_inches='tight',dpi=170)
    write_csv(out/'summary.csv',summary)

