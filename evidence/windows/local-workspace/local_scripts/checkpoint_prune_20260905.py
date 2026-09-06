"""Exact user-approved Linux checkpoint files; plan, reviewed execution, verify."""
import argparse
import base64
import getpass
import json
import os
import shlex
from pathlib import Path
import remote_exec_autodl as ssh

DOC = Path('docs/server-admin')
PLAN = DOC/'SZ_APPROVED_CHECKPOINT_PRUNE_PREFLIGHT_20260905.json'
RESULT = DOC/'SZ_APPROVED_CHECKPOINT_PRUNE_RESULT_20260905.json'
VERIFY = DOC/'SZ_APPROVED_CHECKPOINT_PRUNE_VERIFY_20260905.json'
BASE = '/data/chenyiteng/results/rlinf-shenzhen/'
REMOTE_AUDIT = BASE+'checkpoint-cleanup-20260905-approved'

def candidate_packet():
    manifest = json.loads((DOC/'SZ_CHECKPOINT_CLEANUP_DISCUSSION_MANIFEST_20260905.json').read_text(encoding='utf-8'))
    allowed = ('/grpo/', '/ppo/', '/pi05/', '/fastwam-grpo/')
    smoke_names = ('/pi0-adjust-bottle-smoke32x1-b1024-u10-eval16x2-gpu6-20260905-v7/', '/pi0-bc-dvac32x1-b1024-u10-gpu7-smoke2-20260905-v1/')
    selected = [dict(x) for x in manifest['candidate_files'] if x['path'].startswith(BASE) and (any(k in x['path'] for k in allowed) or any(k in x['path'] for k in smoke_names))]
    probes = [
        ('sft-leaf-wrap-nativeopt-local-orig-false-20260905/checkpoint/model_state_dict/full_weights.pt',8065002471),
        ('sft-leaf-wrap-nativeopt-local-orig-false-20260905/checkpoint/local_shard_checkpoint/checkpoint_rank_0.pt',10390434174),
        ('sync-probe-orig-false-20260905/checkpoint/dcp_checkpoint/__0_0.distcp',9338808918),
        ('sync-probe-orig-false-20260905/checkpoint/model_state_dict/full_weights.pt',8065002471),
    ]
    selected += [{'path':BASE+'online-bc/'+p,'bytes':b,'run_root':None,'generation':'component_probe'} for p,b in probes]
    assert len(selected)==107 and len({x['path'] for x in selected})==107
    roots={x['run_root'] for x in selected if x['run_root']}
    keeps=[{'root':r['root'],'step':r['latest_number_only'],'path':r['root']+'/global_step_'+str(r['latest_number_only'])} for r in manifest['runs'] if r['root'] in roots]
    return {'candidate_files':selected,'keeps':keeps,'audit_path':REMOTE_AUDIT}

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('plan','execute','verify'));a=p.parse_args()
    packet=candidate_packet() if a.mode=='plan' else json.loads(PLAN.read_text(encoding='utf-8'))
    if a.mode=='execute':
        allowed={x['path'] for x in candidate_packet()['candidate_files']}
        assert packet['passed'] and len(packet['candidate_files'])==packet['count']
        assert {x['path'] for x in packet['candidate_files']} <= allowed
        assert not RESULT.exists(), 'Execution result already exists; never automatically replay deletion.'
    packet['mode']=a.mode
    code=Path('local_scripts/checkpoint_prune_remote_20260905.py').read_text(encoding='utf-8')
    source='import json,sys\nPACKET=json.load(sys.stdin)\n'+code
    os.environ['SEETA_SSH_PASSWORD']=getpass.getpass('SSH password: ')
    client=ssh.connect(argparse.Namespace(host='120.241.223.9',port=22,user='chenyiteng',timeout=20,host_key_sha256='qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY'))
    try:
        _,out,err=client.exec_command('id; date -Is');print(out.read().decode(),flush=True);assert out.channel.recv_exit_status()==0
        command="env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' nice -n 19 ionice -c3 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -c "+shlex.quote(source)
        stream,out,err=client.exec_command(command)
        stream.write(json.dumps(packet));stream.flush();stream.channel.shutdown_write()
        data=out.read().decode();error=err.read().decode();rc=out.channel.recv_exit_status()
        target={'plan':PLAN,'execute':RESULT,'verify':VERIFY}[a.mode]
        if data.strip():target.write_text(data,encoding='utf-8')
        if error:print(error[-6000:],flush=True)
        assert rc==0, 'Scoped command failed; NOT replayed. Inspect audit before further action.'
        value=json.loads(data)
        print(json.dumps({k:v for k,v in value.items() if k not in ('candidate_files','keeps','protected_files','small_files','references','active_processes','deleted','retained','deferred')},ensure_ascii=False),flush=True)
    finally:
        client.close();os.environ.pop('SEETA_SSH_PASSWORD',None)

if __name__=='__main__':main()
