"""Flatten the proven fresh wrapper into its existing supervisor, without a new process layer."""
import ast
from pathlib import Path
H=Path(__file__).resolve().parent
base=(H/'reference/opendw_owner_base.py').read_text()
fresh=(H.parent/'lift_two_gpu_20261006/fresh_owner.py').read_text()
def function(source,name):
 node=next(n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
 return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
normal=function(fresh,'normalized')
validate=function(fresh,'validate').replace('    global OWNER_ENV\n','').replace('M.','').replace("'two_gpu_b32_from0'","'two_gpu_b16_from0'").replace('sha(THIS)','sha(__file__)')
a=validate.index("    assert read(plan['smoke_result'])")
b=validate.index("    for row in plan['trials']:",a)
validate=validate[:a]+"    assert plan['skip_smoke'] is True\n"+validate[b:]
validate=validate.replace('    else:\n        OWNER_ENV = environment\n','')
for name in ['validate_config','validate_trial_order']:
 base=base.replace(function(base,name),'')
base=base.replace(function(base,'validate'),normal+'\n\n'+validate)
base=base.replace("assert one_arg(argv, '--wm-batch-size') == '32'","assert one_arg(argv, '--wm-batch-size') == '16'")
scope="""SCOPE_KEYS = {'RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST', 'PYTHONPATH', 'LD_PRELOAD',
              '__GL_APPLICATION_PROFILE', '__GL_APPLICATION_PROFILE_LOG', 'HOME', 'USER', 'LOGNAME'}
"""
base=base.replace("VISIBLE = {'actor': [['4']], 'env': [['5']], 'rollout': [['4']]}\n", "VISIBLE = {'actor': [['4']], 'env': [['5']], 'rollout': [['4']]}\n"+scope)
before="        runtime.setdefault('env_vars', {}).update({TOKEN: plan['token'], PHASE: key})\n"
after=before+"        for name in SCOPE_KEYS:\n            assert os.environ.get(name)\n            runtime['env_vars'][name] = os.environ[name]\n"
assert base.count(before)==1;base=base.replace(before,after)
before='                C.stop(cycle)\n'
after=before+"""                fragment = read(plan['graphics_fragment'])
                manifest = owned_path(fragment['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST'])
                assert sha(manifest) == plan['scope_manifest_sha256']
                assert not H.gpu_processes([4, 5])
                base_env.update(fragment)
                record(owner/'scope-activated.json', {'physical_gpus':[4,5],
                       'reused_existing_profile':True, 'manifest_sha256':sha(manifest)})
"""
assert base.count(before)==1;base=base.replace(before,after)
base=base.replace("'Service exited during smoke'","'Service exited during training'").replace("'Smoke deadline: '","'Training deadline: '").replace("'Smoke failed; do not start next N: '","'Training failed: '").replace('Four-card stop','Two-card stop').replace('complete four-card stopped receipt','complete two-card stopped receipt')
tree=ast.parse(base);doc=tree.body[0]
lines=base.splitlines(keepends=True)
base='"""Two-card lift_pot formal run: B16 WM, unchanged N64/R8 and automatic RLT return.\n\nOne supervisor entry; no smoke, adoption wrapper, or automatic WM retry.\n"""\n'+''.join(lines[doc.end_lineno:])
ast.parse(base)
# Identity/signalling, catalog, cleanup, resource checks and return loop remain
# inherited; only the scope-injection hooks above move out of the old wrapper.
prior=(H/'reference/opendw_owner_base.py').read_text()
for name in ['Catalog','cleanup','resource_snapshot','register_actors','load_lifecycle']:
 assert ast.dump(ast.parse(function(base,name)))==ast.dump(ast.parse(function(prior,name)))
(H/'owner.py').write_text(base,encoding='utf-8',newline='\n')
print('Generated one owner entry; cleanup/resource implementation preserved; no smoke stage')
