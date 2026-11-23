"""Check retry selection using a temporary config and an existing full RLT checkpoint."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from omegaconf import OmegaConf

b=Path(__file__).resolve().parent
sys.path.insert(0,json.loads((b/'return-g4/wait.json').read_text())['queue_ops'])
import rlt_retry

def test():
    results={}
    for gpu in (4,7):
        for kind in ('return','next'):
            name=f'{kind}-g{gpu}'
            plan=json.loads((b/name/'plan.json').read_text())
            cfg=OmegaConf.load(b/name/'config/formal.yaml')
            with TemporaryDirectory(dir=b) as tmp:
                path=Path(tmp);(path/'config').mkdir()
                OmegaConf.save(cfg,path/'config/formal.yaml')
                log=path/'formal.log';log.write_text('Unrelated config error')
                assert not rlt_retry.prepare_retry(path,plan,log)
                log.write_text('TimeoutError: RoboTwin simulator RPC exceeded 900s')
                with patch.object(rlt_retry,'gpus',return_value={gpu:{}}), patch.object(rlt_retry,'admission_card',return_value={'processes':[]}):
                    assert rlt_retry.prepare_retry(path,plan,log)
                recovered=OmegaConf.load(path/'config/formal.yaml').runner.resume_dir
                results[name]=int(Path(recovered).name[12:])
    expected=json.loads((b/'cpu-verified.json').read_text())['rlt_rounds']
    assert results==expected
    (b/'retry-selection.json').write_text(json.dumps(results))
    print(results)
if __name__=='__main__':
    test()
