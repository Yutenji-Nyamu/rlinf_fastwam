"""Reuse the strict frozen RLT helper for the four currently running next-six jobs."""
import argparse
import importlib.util
import os
import socket
import sys
from common import *
sys.path.insert(0, str(SOURCE / 'tools'))
from expo_formal_resources import pin_signals
from expo_process import pidfd_probe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'stop', 'resume', 'status'))
    args = parser.parse_args()
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu02'
    pidfd_probe()
    path = CYCLE / 'rlt_cycle.py'
    assert sha(path) == read(CONTROL / 'staged.json')['files'][str(path)]
    spec = importlib.util.spec_from_file_location('frozen_rlt', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    m = pin_signals(module)
    if args.action == 'resume': result = m.resume(CYCLE, CONTROL / 'release.json')
    else: result = getattr(m, args.action)(CYCLE)
    if args.action == 'status': atomic(CONTROL / 'rlt-status.json', result)
    print(json.dumps(result))

if __name__ == '__main__': main()
