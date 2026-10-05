"""Derive the scoped repair after an authorized profile-name-only tidy."""
from pathlib import Path
P=Path(__file__).resolve().parent
s=(P/'repair_return.py').read_text()
s=s.replace("ROOT=D/'rlt-return-repair-v1'", "ROOT=D/'rlt-return-repair-v2'")
s=s.replace("env=H.read(oldpre/'environment.json')", "env=read(oldpre/'environment.json')")
function='''
def refreshed_scope(plan):
    old_manifest=Path(plan['scope_manifest']);old_activation=Path(plan['scope_activation'])
    m=read(old_manifest);a=read(old_activation)
    assert sha(old_manifest)==a['manifest_sha256'] and a['status']=='active'
    assert m['physical_gpus']==[4,5,6,7] and m['cpu_full_mask_target']==4
    backup=Path('/data/chenyiteng/deployment-20261005/management-tidy-v1')
    changes={}
    def renamed_only(path,old_hash):
        current=Path(path);prior=backup/(current.name+'.before')
        assert current.stat().st_uid==20001 and not current.is_symlink()
        assert sha(prior)==old_hash
        value=read(current);normalized=copy.deepcopy(value);names=[]
        for rule in normalized['rules']:
            profile=rule['profile'];assert set(profile)=={'name','settings'}
            names.append(profile['name']);rule['profile']=profile['settings']
        assert len(names)==len(set(names)) and normalized==read(prior)
        changes[str(current)]={'prior_sha256':old_hash,'sha256':sha(current),'rules_unchanged':True,'backup':str(prior)}
        return sha(current)
    m['profile_sha256']=renamed_only(m['profile_path'],m['profile_sha256'])
    for path,item in m['existing_profiles'].items():
        if sha(path)!=item['sha256']:
            assert Path(path).name=='00-opendw-g4-opendw20261003v4.json'
            item['sha256']=renamed_only(path,item['sha256']);item['size']=Path(path).stat().st_size
    scope=ROOT/'scope';scope.mkdir();mp=scope/'scope.json';ap=scope/'activation.json';ep=scope/'environment-fragment.json'
    bootstrap=scope/'bootstrap';bootstrap.mkdir();(scope/'receipts').mkdir()
    assert sha(m['runtime_path'])==m['runtime_sha256'] and sha(m['bootstrap_path'])==m['bootstrap_sha256']
    runtime=bootstrap/'graphics_scope_runtime.py';runtime.write_bytes(Path(__file__).with_name('graphics_scope_runtime.py').read_bytes());runtime.chmod(0o600)
    site=bootstrap/'sitecustomize.py';site.write_bytes(Path(m['bootstrap_path']).read_bytes());site.chmod(0o600)
    m['runtime_path']=str(runtime);m['runtime_sha256']=sha(runtime)
    m['bootstrap_path']=str(site);m['bootstrap_sha256']=sha(site);m['receipts_dir']=str(scope/'receipts')
    save(mp,m)
    a['derived_from']={'path':str(old_activation),'sha256':sha(old_activation),'reason':'Only profile names changed; matching rules and GPU masks verified identical.'}
    a['manifest']=str(mp);a['manifest_sha256']=sha(mp);a['profile_sha256']=m['profile_sha256']
    a['runtime_path']=str(runtime);a['runtime_sha256']=sha(runtime)
    a['activation_receipt']=str(ap)
    a['environment_fragment']['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST']=str(mp)
    oldbootstrap=str(Path(read(old_manifest)['bootstrap_path']).parent)
    tail=[x for x in a['environment_fragment']['PYTHONPATH'].split(':') if x and x!=oldbootstrap]
    a['environment_fragment']['PYTHONPATH']=':'.join([str(bootstrap),*tail])
    save(ep,a['environment_fragment']);a['environment_fragment_file']=str(ep);a['environment_fragment_sha256']=sha(ep)
    save(ap,a);save(scope/'semantic-equivalence.json',changes)
    return mp,ap,changes

'''
s=s.replace('\ndef main():',function+'\ndef main():')
s=s.replace("q=copy.deepcopy(p);q['time']=H.now();q['runs']['gpu4'].update", "mp,ap,scope_changes=refreshed_scope(p)\n        q=copy.deepcopy(p);q['scope_manifest']=str(mp);q['scope_activation']=str(ap);q['time']=H.now();q['runs']['gpu4'].update")
s=s.replace("q['frozen_files'].update(evidence)", "q['frozen_files'].update(evidence)\n        q['frozen_files'].update({str(f):sha(f) for f in (mp,ap,mp.parent/'environment-fragment.json',mp.parent/'semantic-equivalence.json',mp.parent/'bootstrap/graphics_scope_runtime.py',mp.parent/'bootstrap/sitecustomize.py')})\n        q['frozen_files'].update({path:item['sha256'] for path,item in scope_changes.items()})")
s=s.replace("assert not any(k in env for k in H.MASKS)", "oldbootstrap=str(Path(read(p['scope_manifest'])['bootstrap_path']).parent)\n        env['PYTHONPATH']=':'.join(x for x in env.get('PYTHONPATH','').split(':') if x and x!=oldbootstrap)\n        assert not any(k in env for k in H.MASKS)")
s=s.replace("'no_new_stop':True,'evidence':evidence", "'no_new_stop':True,'scope_profile_names_only':scope_changes,'evidence':evidence")
compile(s,'repair_return_v2.py','exec');(P/'repair_return_v2.py').write_text(s,encoding='utf-8')
print('Wrote derived return repair v2')
