"""Report baseline and routine fixed-layout evaluations from the two native lanes."""
import argparse,json
from pathlib import Path
from statistics import mean

def load_lines(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def evaluation_progress(root):
    waves = {card: [w for w in load_lines(root / f"native-g{card}/completed-waves.jsonl") if w["mode"] == "eval"]
             for card in (5, 6)}
    prev = 0
    result = []
    for event in load_lines(root / "evaluations.jsonl"):
        stop = event["time"]
        batches = [[w for w in card_waves if prev < w["time"] <= stop] for card_waves in waves.values()]
        prev = stop
        if not all(batches):
            raise ValueError(f"Evaluation at round {event['round']} has a missing native batch")
        rows = [r for candidates in batches for r in candidates[-1]["rows"]]
        success = sum(r["success"] for r in rows)
        after_three = [r for r in rows if r["bottles"] >= 3]
        logged = event.get("successes")
        if logged is None:
            logged = round(event["metrics"]["eval/success_once"] * len(rows))
        if success != logged or len({r["layout"] for r in rows}) != 32 or len(rows) != 32:
            raise ValueError(f"Evaluation at round {event['round']} disagrees with the fixed panel")
        result.append(dict(round=event["round"],episodes=len(rows),successes=success,
                           mean_bottles=mean(r["bottles"] for r in rows),reached_three=len(after_three),
                           successes_after_three=sum(r["success"] for r in after_three)))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluation_progress(args.run),indent=2))

