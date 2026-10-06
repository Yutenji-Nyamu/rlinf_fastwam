"""Serialize the completed native RM result update on the existing branch."""
import inspect
import make_publication_rpc as base

base.PRIOR='c79d02877ccb4e7457fe182d2bccf197ceb760b1'
base.RELEASE='task-reward-results-20261006-v1'
base.MANIFEST=base.PREFIX+'results_manifest.json'
base.FILES=base.FILES+(
 base.PREFIX+'results_20261006.md',base.PREFIX+'performance_results.json',
 base.PREFIX+'implementation_published.json',base.CODE+'make_results_publication_rpc.py')
source=inspect.getsource(base.main)
source=source.replace("PREFIX + 'published.json'", "PREFIX + 'implementation_published.json'")
source=source.replace('publication/wmrl-task-reward-plan-20261006-v1/published.json','publication/task-reward-implementation-20261006-v1/published.json')
source=source.replace('Add task reward classifier training and native capture pipeline','Record lift-pot reward results and resource measurements')
source=source.replace("'publication_stage_remote.py'", "'results_stage_remote.py'")
source=source.replace("'publication_push_remote.py'", "'results_push_remote.py'")
exec(compile(source,__file__, 'exec'),base.__dict__)
if __name__=='__main__':base.main()
