"""Publish read-only memory evidence and two-GPU discussion; no deployment."""
import inspect
import make_publication_rpc as base
OLD = base.PRIOR
base.PRIOR = '8d9b46f484a1e4c191c0898eaf729082c47513dc'
base.RELEASE = 'lift-wmrl-memory-20261006-v1'
base.MANIFEST = base.PREFIX + 'memory_manifest_2128.json'
base.LIGHT = base.PREFIX + 'memory_light_2128.json'
base.IDENTITIES = base.IDENTITIES.replace(OLD, base.PRIOR)
base.FILES = base.FILES + (
    base.CODE + 'make_memory_publication_rpc.py',
    base.CODE + 'memory_layout_remote.py',
    base.PREFIX + 'two_gpu_discussion_2128.md',
    base.PREFIX + 'overview_published_2055.json', base.LIGHT,
)
source = inspect.getsource(base.main)
source = source.replace('PREFIX + "results_published.json"', 'PREFIX + "overview_published_2055.json"')
source = source.replace('publication/task-reward-results-20261006-v1/published.json', 'publication/lift-wmrl-overview-20261006-v1/published.json')
source = source.replace('Integrate converged lift-pot reward model with existing WMRL pipeline', 'Record WMRL GPU memory and two-GPU options')
source = source.replace('"publication_stage_remote.py"', '"memory_stage_remote.py"').replace('"publication_push_remote.py"', '"memory_push_remote.py"')
source = source.replace('Lift-pot RM convergence, error review metadata, reset/native seeds, B16 integration and lifecycle preparation.', 'Read-only GPU memory snapshot and measured B16 timing; two-GPU feasibility discussion without execution.')
exec(compile(source, __file__, 'exec'), base.__dict__)
if __name__ == '__main__':
    base.main()
