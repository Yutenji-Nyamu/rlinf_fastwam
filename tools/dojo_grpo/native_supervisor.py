# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Restart only the native simulator at clean wave boundaries, keeping RL alive."""

import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops")
from common import exact_signal, proc, same, save


def main():
    directory, gpu = Path(sys.argv[1]), int(sys.argv[2])
    command = sys.argv[3:]
    child = identity = None

    def stop(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    try:
        generation = 0
        while True:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
            identity = proc(child.pid)
            save(directory / f"native-g{gpu}-child.json", identity)
            save(
                directory / f"native-g{gpu}-generation.json",
                {"generation": generation, "pid": child.pid},
            )
            code = child.wait()
            if code != 75:
                raise SystemExit(code)
            # 75 is the explicit reset-boundary handshake, not a generic crash.
            generation += 1
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if identity and same(identity):
            exact_signal(identity, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if same(identity):
                    exact_signal(identity, signal.SIGKILL)
                child.wait(timeout=10)


if __name__ == "__main__":
    main()
