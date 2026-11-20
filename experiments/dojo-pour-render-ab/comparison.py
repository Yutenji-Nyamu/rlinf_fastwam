"""Compare one 128-episode GRPO update from each training renderer."""
import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def read(path):
    return json.loads(path.read_text())

def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def evaluation(directory, start):
    result = []
    for gpu in (5, 6):
        waves = records(directory / f"native-g{gpu}/completed-waves.jsonl")
        wave = next(w for w in waves if w["mode"] == "eval" and w["time"] >= start)
        result.extend(wave["rows"])
    assert len(result) == 32
    return result

def training(directory, results, after):
    layouts=[]; seconds=[]; ops=[]; successes=0; mixed=0; executed=0
    for gpu in (5,6):
        native=directory/f"native-g{gpu}"
        waves = [w for w in records(native/"completed-waves.jsonl") if w["mode"]=="train" and w["time"]>=after][:2]
        timings = [w for w in records(native/"timing.jsonl") if w["event"]=="wave_complete" and w["mode"]=="train" and w["time"]>=after][:2]
        assert len(waves)==len(timings)==2
        for wave,t in zip(waves,timings):
            rows=wave["rows"]; assert len(rows)==32
            for idx in range(0,32,8):
                group=rows[idx:idx+8]; assert len({v["layout"] for v in group})==1
                n=sum(r["success"] for r in group)
                successes+=n; mixed+=int(0<n<8); layouts.append(group[0]["layout"])
            executed+=sum(int(r["actions"]) for r in rows)
            seconds.append({k:t[k] for k in ("reset_seconds","action_seconds","observation_seconds")})
            ops.append(t.get("timed_operations",{}))
    events=EventAccumulator(str(Path(results)/"tensorboard"),size_guidance={"scalars":0})
    events.Reload()
    def val(key):
        return next(e.value for e in events.Scalars(key) if e.step==0)
    times={k:val("time/"+k) for k in ("sync_weights","generate_rollouts","cal_adv_and_returns","actor_training")}
    extras={k:val(k) for k in events.Tags()["scalars"] if k in ("train/grad_norm","train/actor/grad_norm","train/lr","train/actor_loss","train/loss","rollout/advantages_std")}
    return dict(episodes=128,successes=successes,mixed_groups=mixed,groups=16,executed=executed,group_layouts=layouts,
                train_seconds=val('time/step'),successes_per_hour=successes*3600/val('time/step'),mixed_groups_per_hour=mixed*3600/val('time/step'),env_actions_per_second=executed/val('time/step'),sampler_and_update=sum(times.values()),runner_seconds=times,update_metrics=extras,native_wave_seconds=seconds,native_ops=ops)

def decide(a,b,ae,be,base):
    # A layout replaced during a native reset is comparable only to the same realized layout.
    def grouped(rows):
        groups={}
        for r in rows:groups.setdefault(r["layout"],[]).append(int(r["success"]))
        return groups
    x,y,z=map(grouped,(ae,be,base))
    common=sorted(x.keys()&y.keys()&z.keys())
    # Pair by occurrence when duplicate layouts are present.
    triples=[(x[k][j],y[k][j],z[k][j]) for k in common for j in range(min(len(x[k]),len(y[k]),len(z[k])))]
    n=len(triples); amounts=[sum(t[j] for t in triples) for j in range(3)]
    saving=1-b["train_seconds"]/a["train_seconds"]
    enough_pairs=n>=24
    reasons=[]
    if not enough_pairs:reasons.append("fewer than 24 paired evaluation episodes")
    if saving<0.10:reasons.append("full training round wall saving below 10 percent")
    if b["successes"]==0 or b["mixed_groups"]==0:reasons.append("candidate lost the training signal")
    if amounts[1]==0 or amounts[0]-amounts[1]>0.10*n+1e-8:reasons.append("official-style evaluation fell to zero or lost over ten points")
    return dict(winner="native" if reasons else "chunk",reasons=reasons,training_wall_fraction_saved=saving,
                paired_layouts=common,paired_count=n,paired_A_B_base=amounts,paired_change_from_base=[(v-amounts[2])/n for v in amounts[:2]] if n else [],
                A_only=sum(a and not b for a,b,_ in triples),B_only=sum(b and not a for a,b,_ in triples),
                equal_realized_training_groups=sum(i==j for i,j in zip(a["group_layouts"],b["group_layouts"])))

def finalize(d):
    previous=Path(read(d/"comparison-plan.json")["reference_deployment"])
    a=training(d,Path(read(d/"plan.json")["run"])/"pilot-a",read(d/"a-window.json")["start"])
    b=training(d,Path(read(d/"plan.json")["run"])/"pilot-b",read(d/"b-window.json")["start"])
    ae,be=[read(d/f"{s}-eval.json")["rows"] for s in ("a","b")]
    base=evaluation(previous,0)
    result=dict(reference=a,candidate=b,baseline=base,eval_a=ae,eval_b=be,decision=decide(a,b,ae,be,base))
    windows=records(d/"chunk-times.jsonl") if (d/"chunk-times.jsonl").exists() else []
    result["fixed_work_100_actions_per_env"]={name:[dict(gpu=w["gpu"], generation=w["generation"],
         elapsed_step_seconds=w["action_seconds"]+w["observation_seconds"], observation_seconds=w["observation_seconds"])
         for w in windows if w["arm"]==name and w["time"]>=read(d/f"{name.lower()}-window.json")["start"] and len(w["counts"])==32 and set(w["counts"])=={100}] for name in ("A","B")}
    write(d/"comparison-result.json",result)
    return result["decision"]["winner"]
