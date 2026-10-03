import multiprocessing,runpy

def run():
    runpy.run_path('/data/chenyiteng/projects/expo-ft-sz2-20261001/gpu4567-fix-20261003/native-probe-v5.py',run_name='__main__')

if __name__=='__main__':
    p=multiprocessing.get_context('spawn').Process(target=run)
    p.start();p.join(150)
    if p.is_alive():
        p.terminate();p.join(10)
    assert p.exitcode==0,p.exitcode
