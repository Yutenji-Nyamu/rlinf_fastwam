"""Publish a dated live-status summary on the existing isolated branch."""
import inspect
import make_publication_rpc as base

OLD=base.PRIOR
base.PRIOR='5888b9e9cbdbf802244643db673b9c8a57052cff'
base.RELEASE='lift-wmrl-status-20261006-v1'
base.MANIFEST=base.PREFIX+'status_manifest_1531.json'
base.LIGHT=base.PREFIX+'status_light_1531.json'
base.IDENTITIES=base.IDENTITIES.replace(OLD,base.PRIOR)
base.FILES=base.FILES+(
 base.CODE+'make_status_publication_rpc.py',base.CODE+'audit_remote.py',
 base.PREFIX+'status_1531.md',base.PREFIX+'integration_published.json',base.LIGHT)
source=inspect.getsource(base.main)
source=source.replace('PREFIX + "results_published.json"','PREFIX + "integration_published.json"')
source=source.replace('publication/task-reward-results-20261006-v1/published.json','publication/lift-wmrl-integration-20261006-v1/published.json')
source=source.replace('Integrate converged lift-pot reward model with existing WMRL pipeline','Record live lift-pot baseline and WMRL queue status')
source=source.replace('"publication_stage_remote.py"','"status_stage_remote.py"').replace('"publication_push_remote.py"','"status_push_remote.py"')
exec(compile(source,__file__,'exec'),base.__dict__)
if __name__=='__main__':base.main()
