"""Summarize the new deployment's existing logs without touching the training process."""
import collections
import json
import sys
from pathlib import Path

d=Path(__file__).resolve().parent
def read(path):
    return json.loads(path.read_text())
def lines(path):
    if not path.exists():
        return []
    return [json.loads(v) for v in path.read_text().splitlines() if v.strip()]

p=read(d/"plan.json")
training=[]
zero=[]
groups=[]
layout_lists=[]
for g in p["gpus"]:
    batches=lines(d/f"native-g{g}/completed-waves.jsonl")
    ev=[b for b in batches if b["mode"]=="eval"]
    if ev: zero.extend(ev[0]["rows"])
    wave=[b for b in batches if b["mode"]=="train"][:2]
    layout_lists.append([len(b["rows"]) for b in wave])
    for w in wave:
        rows=w["rows"]
        training.extend(rows)
        for i in range(0,len(rows),8):
            group=rows[i:i+8]
            assert len(group)==8 and len({r["layout"] for r in group})==1
            groups.append((len({r["success"] for r in group})>1,len({r["episode_return"] for r in group})>1))

def aggregate(rows,mode):
    if not rows: return None
    assert all(r["episode_return"]==(r["bottles"]*.25 if mode=="train" else int(r["success"])) for r in rows)
    return dict(episodes=len(rows),official_successes=sum(r["success"] for r in rows),
                at_least_n_bottles=[sum(r["bottles"]>=n for r in rows) for n in range(1,5)],
                four_bottles_without_official_success=sum(r["bottles"]==4 and not r["success"] for r in rows),
                average_episode_return=sum(r["episode_return"] for r in rows)/len(rows))
out=dict(task=p["task"],initialization="original_pi05_no_resume",phase=read(d/"status.json")["phase"],
         baseline=aggregate(zero,"eval"),first_training_round=aggregate(training,"train"),
         first_train_wave_sizes_by_rank=layout_lists,
         first_train_groups=dict(total=len(groups),binary_success_varies=sum(a for a,b in groups),
              bottle_return_varies=sum(b for a,b in groups),newly_informative=sum(b and not a for a,b in groups),
              only_binary_varies=sum(a and not b for a,b in groups)))
cks=Path(p["run"])/"bottles-quarters-noise05/checkpoints"
steps=[int(m.parent.name.split("_")[-1]) for m in cks.glob("global_step_*/dojo-complete.json")]
out["latest_complete_checkpoint"]=max(steps,default=0)
if "--reference" in sys.argv and training:
    previous=Path(sys.argv[sys.argv.index("--reference")+1])
    matches=[]
    for g in p["gpus"]:
        def training_groups(directory):
            where=directory/f"native-g{g}"
            waves=[w for w in lines(where/"completed-waves.jsonl") if w["mode"]=="train"][:2]
            requests=[r for r in lines(where/"requested-layouts.jsonl") if r["mode"]=="train"]
            groups=[]
            for request,wave in zip(requests,waves):
                for i in range(0,len(wave["rows"]),8):
                    lanes=wave["rows"][i:i+8]
                    groups.append((request["requested"][i],tuple(r["layout"] for r in lanes),
                                   sum(r["success"] for r in lanes)))
            return groups
        for old,new in zip(training_groups(previous),training_groups(d)):
            if old[:2] == new[:2]:
                matches.append((old[2],new[2]))
    out["before_first_update_matched_layout_groups"]=dict(groups=len(matches),
        episodes_per_setting=8*len(matches),noise03_successes=sum(a for a,b in matches),
        noise05_successes=sum(b for a,b in matches),
        note="Same group position, requested layout and observed layout IDs; sampling draws unpaired.")
print(json.dumps(out,indent=2))
if "--save" in sys.argv:
    assert zero or training
    (d/"public-run.json").write_text(json.dumps(out,indent=2)+"\n")
    if zero:
        selected=("layout","success","actions","bottles","first_bottle_steps","episode_return")
        layout_rows=[{key:row[key] for key in selected} for row in sorted(zero,key=lambda r:r["layout"])]
        (d/"baseline-layouts.json").write_text(json.dumps(layout_rows,indent=2)+"\n")
