"""Pinned clean50 data preparation; downloads overlap, conversions are serial."""
import concurrent.futures
import datetime
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
import types
import zipfile

ROOT = Path('/data/chenyiteng/datasets/robotwin2')
ST = Path('/data/chenyiteng/deployment-20261023/endpoint-multitask-sz1-g45/data')
REV = '9dc9299c163db059931898a9f0852098a61155a1'
DATE = '20261023'


def now():
    return datetime.datetime.now().astimezone().isoformat()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def download(task):
    from huggingface_hub import hf_hub_download
    archive = ST / 'downloads' / REV / f'dataset/{task}/aloha-agilex_clean_50.zip'
    if not archive.is_file():
        archive = Path(hf_hub_download(
            repo_id='TianxingChen/RoboTwin2.0', repo_type='dataset', revision=REV,
            filename=f'dataset/{task}/aloha-agilex_clean_50.zip',
            local_dir=str(ST / 'downloads' / REV)))
    save(ST / f'{task}-download.json', {'time': now(), 'archive': str(archive),
                                      'bytes': archive.stat().st_size})
    print(json.dumps({'event': 'download_complete', 'task': task, 'time': now()}), flush=True)
    return archive


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def convert(task, archive):
    import numpy as np
    dest = ROOT / 'canonical'
    repo_id = task.replace('_', '-') + '-aloha-clean50-' + DATE
    output = dest / repo_id
    if (output / 'prepared.json').is_file():
        info = json.loads((output / 'meta/info.json').read_text())
        assert info['total_episodes'] == 50
        return json.loads((output / 'prepared.json').read_text())
    assert not output.exists(), f'Partial canonical requires explicit review: {output}'
    raw_parent = ROOT / 'raw' / REV / task / ('clean50-' + DATE)
    raw_parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
        source.extractall(raw_parent)
    raw = raw_parent / 'aloha-agilex_clean_50'
    assert len(list((raw / 'data').glob('episode*.hdf5'))) == 50
    intermediate = ROOT / 'intermediate' / REV / task / ('clean50-' + DATE)
    if not intermediate.exists():
        process = module('raw_to_aloha', ST / 'converter/scripts/process_data.py')
        process.data_transform(str(raw), 50, str(intermediate))
    os.environ['HF_LEROBOT_HOME'] = str(dest)
    source = (ST / 'converter/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py').read_text()
    source = source.replace('from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME',
                            f'HF_LEROBOT_HOME = Path({str(dest)!r})')
    source = source.replace('from lerobot.common.datasets.lerobot_dataset import LeRobotDataset',
                            'from lerobot.datasets.lerobot_dataset import LeRobotDataset')
    assert source.count('dataset.add_frame(frame)') == 1
    source = source.replace('dataset.add_frame(frame)', "dataset.add_frame(frame, task=frame.pop('task'))")
    converter = types.ModuleType('aloha_to_lerobot')
    sys.modules[converter.__name__] = converter
    exec(compile(source, 'pinned_converter_compat', 'exec'), converter.__dict__)
    random.seed(0)
    np.random.seed(0)
    files = sorted(intermediate.rglob('episode_*.hdf5'), key=lambda p: int(p.stem.split('_')[-1]))
    assert len(files) == 50
    dataset = converter.create_empty_dataset(
        repo_id, robot_type='aloha', mode='image',
        has_effort=converter.has_effort(files), has_velocity=converter.has_velocity(files),
        dataset_config=converter.DEFAULT_DATASET_CONFIG)
    converter.populate_dataset(dataset, files, task=task.replace('_', ' '), episodes=list(range(50)))
    info = json.loads((output / 'meta/info.json').read_text())
    assert info['total_episodes'] == 50
    receipt = {'complete': True, 'task': task, 'time': now(), 'path': str(output),
               'episodes': info['total_episodes'], 'frames': info['total_frames'],
               'dataset_revision': REV, 'archive': str(archive)}
    save(output / 'prepared.json', receipt)
    save(ST / f'{task}-complete.json', receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


def run(tasks):
    lock = (ST / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(ST / 'worker-started.json', {'pid': os.getpid(), 'time': now(), 'tasks': tasks})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {task: executor.submit(download, task) for task in tasks}
        for task in tasks:
            archive = futures[task].result()
            print(json.dumps({'event': 'conversion_start', 'task': task, 'time': now()}), flush=True)
            # A fresh process per dataset releases converter/image-writer memory.
            subprocess.run([sys.executable, '-u', '-B', __file__, '--convert', task, str(archive)], check=True)
    save(ST / 'all-complete.json', {'time': now(), 'tasks': tasks})


if __name__ == '__main__':
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    try:
        if sys.argv[1] == '--convert':
            convert(sys.argv[2], Path(sys.argv[3]))
        else:
            run(sys.argv[1:])
    except BaseException:
        if sys.argv[1] != '--convert':
            save(ST / 'worker-error.json', {'time': now(), 'error': traceback.format_exc()})
        raise
