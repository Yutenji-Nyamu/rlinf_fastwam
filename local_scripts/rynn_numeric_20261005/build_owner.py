"""Derive the established single-card owner with only child and data paths changed."""
import re
from pathlib import Path
P=Path(__file__).resolve().parent
old=P.parent/'rynn_wmrl_20261005/rynn_diagnostic_owner_v2.py'
source=old.read_text(encoding='utf-8')
source=source.replace("D = S / 'rynn-diagnosis-v2'","D = S / 'rynn-numeric-v1'")
source=source.replace("PARENT = S / 'rynn-diagnosis-v1/run'","PARENT = S / 'rynn-diagnosis-v2/run'")
source=source.replace('rynn_diagnostic_owner_v2.py','rynn_numeric_owner.py').replace('rynn_native_diagnostic.py','rynn_numeric_probe.py')
source=source.replace("CASES = S / 'rynn-diagnosis-v1/prepared/cases.json'","CASES = D / 'prepared/cases.json'")
source=source.replace("SAMPLES = S / 'rynn-diagnosis-v1/prepared/samples.npz'","SAMPLES = D / 'prepared/samples.npz'")
source=source.replace("stage = D / 'prepared/rynn-diagnosis-v2-gpu4'","stage = D / 'prepared/rynn-numeric-v1-gpu4'")
expected="""    expected = {'--service-module': str(S / 'rynn-control-v2/code/rynn_success_service.py'),
        '--model-path': str(MODEL), '--manifest-path': str(MODEL / 'manifest.json'),
        '--physical-gpu': '4', '--cases-json': str(CASES), '--samples-npz': str(SAMPLES),
        '--official-inference': str(OFFICIAL), '--output': str(O / 'result.json')}
"""
source,n=re.subn(r'    expected = \{.*?\n(?=    assert len\(argv)',expected,source,flags=re.S);assert n==1
argv="""    argv = [RYNN_PYTHON, '-u', '-B', str(D / 'code/rynn_numeric_probe.py'),
        '--service-module', str(S / 'rynn-control-v2/code/rynn_success_service.py'),
        '--model-path', str(MODEL), '--manifest-path', str(MODEL / 'manifest.json'),
        '--physical-gpu', '4', '--cases-json', str(CASES), '--samples-npz', str(SAMPLES),
        '--official-inference', str(OFFICIAL), '--output', str(O / 'result.json')]
"""
source,n=re.subn(r'    argv = \[RYNN_PYTHON,.*?\n(?=    child = read\(stage)',argv,source,flags=re.S);assert n==1
files="""    files = [SCRIPT, D / 'code/rynn_numeric_probe.py', Path(argv[5]), MODEL / 'manifest.json',
        CASES, SAMPLES, OFFICIAL, OFFICIAL.with_name('plot_utils.py'), D / 'prepared/cpu-tests.json',
        stage / 'plan.json', catalog, Path(scope['activation_receipt']), Path(active['manifest']), Path(active['runtime_path'])]
"""
source,n=re.subn(r'    files = \[SCRIPT,.*?\n(?=    source = dict\(child)',files,source,flags=re.S);assert n==1
source=source.replace('GPU4 now comes from diagnostic-v2 single-card lifecycle','GPU4 now comes from numeric-v1 single-card lifecycle')
source=source.replace('Native diagnostic','Numeric diagnostic')
compile(source,'rynn_numeric_owner.py','exec')
(P/'rynn_numeric_owner.py').write_text(source,encoding='utf-8')
print('Created numeric owner from the validated single-card parent contract')
