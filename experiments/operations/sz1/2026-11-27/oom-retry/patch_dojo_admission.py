"""Apply the shared admission rule to an OOM retry of an existing Dojo pair."""
from pathlib import Path


ORIGINAL = '''        while any(same(who) for who in plan["predecessors"]):
            time.sleep(0.25)
        deadline = time.monotonic() + 240
        while True:
            snapshot = gpus()
            if all(not admission_card(snapshot[g], plan["admission"][str(g)])["processes"] for g in cards):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Predecessor GPU cleanup; see actual device occupancy")
            time.sleep(0.5)
'''
RETRY = '''        rules = os.environ.get("PAIR_OOM_RETRY_RULES")
        if rules:
            sys.path.insert(0, rules)
            from queue_admission import wait_retry_cleanup
            wait_retry_cleanup(d, cards)
        else:
''' + "".join("    " + line for line in ORIGINAL.splitlines(keepends=True))


def update(source):
    if RETRY in source:
        return source
    if source.count(ORIGINAL) != 1:
        raise ValueError("Expected the two-GPU Dojo admission block exactly once")
    result = source.replace(ORIGINAL, RETRY)
    compile(result, "pair_owner.py", "exec")
    return result


if __name__ == "__main__":
    import argparse
    import os
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("owner", type=Path)
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    if args.owner.name != "pair_owner.py":
        p.error("Expected pair_owner.py")
    original = args.owner.read_text()
    changed = update(original)
    if args.write and original != changed:
        tmp = args.owner.with_suffix(".oom-admission.tmp")
        with tmp.open("x") as f:
            f.write(changed)
        tmp.chmod(args.owner.stat().st_mode)
        os.replace(tmp, args.owner)
