"""Publish formal-start scalar evidence on the existing branch."""
import inspect
import make_publication_rpc as base
OLD=base.PRIOR
base.PRIOR='47097c225d0a86e7029ea548760fbf4c9804415f'
base.RELEASE='lift-wmrl-progress-20261006-v1'
base.MANIFEST=base.PREFIX+'status_manifest_1754.json'
base.LIGHT=base.PREFIX+'status_light_1754.json'
base.IDENTITIES=base.IDENTITIES.replace(OLD,base.PRIOR)
base.FILES=base.FILES+(base.CODE+'make_progress_publication_rpc.py',base.CODE+'progress_remote.py',base.PREFIX+'status_1754.md',base.PREFIX+'status_published_1531.json',base.LIGHT)
source=inspect.getsource(base.main)
source=source.replace('PREFIX + "results_published.json"','PREFIX + "status_published_1531.json"')
source=source.replace('publication/task-reward-results-20261006-v1/published.json','publication/lift-wmrl-status-20261006-v1/published.json')
source=source.replace('Integrate converged lift-pot reward model with existing WMRL pipeline','Record first four formal lift-pot WMRL rounds')
source=source.replace('"publication_stage_remote.py"','"progress_stage_remote.py"').replace('"publication_push_remote.py"','"progress_push_remote.py"')
exec(compile(source,__file__,'exec'),base.__dict__)
if __name__=='__main__':base.main()
