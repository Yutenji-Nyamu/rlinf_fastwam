"""Install only the formal retry branch into the RLT task owner for a selected source tree."""
import argparse
import os
from pathlib import Path

RT2_OLD = '''                if attempt or phase["name"]!="formal" or not prepare_retry(d,p,output):
                    raise RuntimeError(f"{phase['name']} exited {code}, see {output.name}")
'''
RT2_NEW = '''                retry = False
                if not attempt and phase["name"] == "formal":
                    oom_retry = None
                    if p.get("retry_formal_gpu_oom") and code not in (124, 143, -15):
                        from rlt_formal_oom_1127 import prepare_oom_retry, drain_formal

                        oom_retry = prepare_oom_retry(d, output)
                        if oom_retry:
                            drain_formal(d)
                    retry = prepare_retry(d, p, output) if oom_retry is None else oom_retry
                if not retry:
                    raise RuntimeError(f"{phase['name']} exited {code}, see {output.name}")
'''
DOJO_RUN = '''        def run_phase(name, phase):
            p = launch(name, phase["command"], env)
            try:
                return p.wait(timeout=phase.get("timeout"))
            except subprocess.TimeoutExpired:
                who = proc(p.pid)
                exact_signal(who, signal.SIGTERM)
                try:
                    p.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    exact_signal(who, signal.SIGKILL)
                    p.wait(timeout=15)
                return 124

        for phase in plan["phases"]:
            name = phase["name"]
            status(name, started=time.time())
            eligible = name == "formal" and plan.get("retry_formal_gpu_oom")
            if eligible:
                native_log = d / f"native-g{gpu}.log"
                native_offset = native_log.stat().st_size
            code = run_phase(name, phase)
            if eligible and code and code not in (124, 143, -15):
                from rlt_formal_oom_1127 import prepare_oom_retry, drain_formal, clear_inactive_socket

                if prepare_oom_retry(d, d / "formal.log", (native_log, native_offset)):
                    drain_formal(d)
                    native.wait(timeout=15)
                    clear_inactive_socket(d / f"rpc/gpu{gpu}.sock")
                    native = start_native(f"native-g{gpu}-oom-retry")
                    status(name, started=time.time(), attempt=2)
                    code = run_phase("formal-retry", phase)
'''

def replace_section(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Expected exactly one source anchor: {before.splitlines()[0]}")
    return text.replace(before, after, 1)


def intermediate_owner(text, bench):
    if bench == "rt2":
        if RT2_NEW in text:
            return text
        return replace_section(text, RT2_OLD, RT2_NEW)
    if DOJO_RUN in text:
        return text
    begin = '        native = launch(f"native-g{gpu}", command, e, plan["native_repo"])'
    finish = '        env = dict(os.environ, **read(d / "training-environment.json"))\n'
    start, end = text.index(begin), text.index(finish, text.index(begin))
    native = text[start:end]
    body = native.replace('native = launch(f"native-g{gpu}",','native = launch(name,',1)
    body = body.replace('f"native-g{gpu}.log"', 'f"{name}.log"')
    body = ''.join('    ' + line if line.strip() else line for line in body.splitlines(True))
    new = '        def start_native(name):\n' + body + '            return native\n\n        native = start_native(f"native-g{gpu}")\n'
    text = replace_section(text,native,new)
    start = text.index('        for phase in plan["phases"]:\n',text.index(finish, start))
    end = text.index('            if code and phase["name"] in ("sac_v2", "environment_v2"):',start)
    original = text[start:end]
    assert 'p = launch(phase["name"], phase["command"], env)' in original and 'code = 124' in original
    return replace_section(text,original,DOJO_RUN)


RT2_LOOP = '''            if p.get("retry_formal_gpu_oom"):
                from rlt_formal_oom_1127 import attempt_name, drain_formal, prepare_oom_retry
            budget = {}
            attempt = 0
            while True:
                if p.get("retry_formal_gpu_oom"):
                    output = d / (attempt_name(phase["name"], attempt) + ".log")
                else:
                    output = d / (phase["name"] + ("-retry.log" if attempt else ".log"))
'''
RT2_DECIDE = '''                oom_retry = None
                if (p.get("retry_formal_gpu_oom") and phase["name"] in ("formal", "stage1", "reference_eval")
                        and code not in (124, 143, -15)):
                    oom_retry = prepare_oom_retry(d, output, budget=budget, gpu=p["gpu"], phase=phase["name"])
                if oom_retry:
                    drain_formal(d)
                    time.sleep(budget["delay"])
                elif not (oom_retry is None and attempt == 0 and phase["name"] == "formal"
                          and prepare_retry(d, p, output)):
                    raise RuntimeError(f"{phase['name']} exited {code}, see {output.name}")
                attempt += 1
'''
DOJO_LOOP = '''            eligible = name == "formal" and plan.get("retry_formal_gpu_oom")
            if not eligible:
                code = run_phase(name, phase)
            else:
                from rlt_formal_oom_1127 import attempt_name, prepare_oom_retry, drain_formal, clear_inactive_socket
                attempt, budget = 0, {}
                native_name = f"native-g{gpu}"
                while True:
                    native_log = d / (native_name + ".log")
                    native_offset = native_log.stat().st_size
                    attempt_log = attempt_name(name, attempt)
                    code = run_phase(attempt_log, phase)
                    if not code or code in (124, 143, -15):
                        break
                    if not prepare_oom_retry(d, d / (attempt_log + ".log"), (native_log, native_offset),
                                             budget=budget, gpu=gpu, phase=name):
                        break
                    drain_formal(d)
                    native.wait(timeout=15)
                    clear_inactive_socket(d / f"rpc/gpu{gpu}.sock")
                    time.sleep(budget["delay"])
                    attempt += 1
                    native_name = attempt_name(f"native-g{gpu}", attempt)
                    native = start_native(native_name)
                    status(name, started=time.time(), attempt=attempt + 1)
'''


def update(text, bench):
    """Upgrade either the original launcher or the installed single-retry version."""
    if (bench == "rt2" and RT2_DECIDE in text) or (bench == "dojo" and DOJO_LOOP in text):
        return text
    text = intermediate_owner(text, bench)
    if bench == "rt2":
        old_loop = ('            for attempt in range(2):\n'
                    '                output=d/(phase["name"] + (".log" if attempt==0 else "-retry.log"))\n')
        return replace_section(replace_section(text, old_loop, RT2_LOOP), RT2_NEW, RT2_DECIDE)
    old_loop = "            eligible = " + DOJO_RUN.split("            eligible = ", 1)[1]
    return replace_section(text, old_loop, DOJO_LOOP)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bench", choices=("rt2", "dojo"))
    p.add_argument("owner", type=Path)
    p.add_argument("--write", action="store_true", help="Apply atomically on the chosen file")
    args = p.parse_args()
    expected = "rt2_owner.py" if args.bench == "rt2" else "rlt_dojo_owner.py"
    if args.owner.name != expected or not args.owner.is_file():
        p.error(f"Expected existing {expected}")
    before = args.owner.read_text(encoding="utf-8-sig")
    after = update(before, args.bench)
    compile(after, str(args.owner), "exec")
    if after == before:
        print("Already applied", args.owner)
    elif args.write:
        tmp = args.owner.with_suffix(".oom-retry.tmp")
        with tmp.open("x",encoding="utf-8",newline="\n") as f: f.write(after)
        os.chmod(tmp, args.owner.stat().st_mode)
        os.replace(tmp,args.owner)
        print("Applied",args.owner)
    else:
        print("Validated",args.owner)


if __name__=="__main__":
    main()

