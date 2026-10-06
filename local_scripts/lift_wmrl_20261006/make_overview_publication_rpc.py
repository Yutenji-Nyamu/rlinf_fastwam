"""Publish CP10 evaluation and resource-layout evidence."""
import inspect
import make_publication_rpc as base
OLD=base.PRIOR
base.PRIOR='a20171f32f5e1340daba2b3c2bbb89de7360249d'
base.RELEASE='lift-wmrl-overview-20261006-v1'
base.MANIFEST=base.PREFIX+'overview_manifest_2055.json'
base.LIGHT=base.PREFIX+'overview_light_2055.json'
base.IDENTITIES=base.IDENTITIES.replace(OLD,base.PRIOR)
base.FILES=base.FILES+(base.CODE+'make_overview_publication_rpc.py',base.CODE+'overview_remote.py',base.CODE+'overview_detail_remote.py',base.PREFIX+'overview_2055.md',base.PREFIX+'overview_detail_2055.json',base.PREFIX+'status_published_1754.json',base.LIGHT)
source=inspect.getsource(base.main)
source=source.replace('PREFIX + "results_published.json"','PREFIX + "status_published_1754.json"')
source=source.replace('publication/task-reward-results-20261006-v1/published.json','publication/lift-wmrl-progress-20261006-v1/published.json')
source=source.replace('Integrate converged lift-pot reward model with existing WMRL pipeline','Record lift-pot CP10 native evaluation and four-GPU layout')
source=source.replace('"publication_stage_remote.py"','"overview_stage_remote.py"').replace('"publication_push_remote.py"','"overview_push_remote.py"')
exec(compile(source,__file__,'exec'),base.__dict__)
if __name__=='__main__':base.main()
