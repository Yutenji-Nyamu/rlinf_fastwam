"""Instrument only isolated probes; charge nested calls to the innermost operation."""
from functools import wraps
from time import perf_counter
def instrument(loop):
    stack=[]
    for obj,name,label in ((loop.env,'sim_step','physics'),(loop.env,'process_control_info','interpolation'),(loop.env.reward_manager,'step','reward'),(loop.env,'is_episode_end','termination'),(loop.env.obs_manager,'render_for_capture','render')):
        original=getattr(obj,name)
        def wrap(fn,tag):
            @wraps(fn)
            def measured(*a,**kw):
                start=perf_counter();stack.append(0.)
                try:return fn(*a,**kw)
                finally:
                    elapsed=perf_counter()-start
                    exclusive=elapsed-stack.pop()
                    loop.trial_profile[tag]=loop.trial_profile.get(tag,0.)+exclusive
                    if stack:stack[-1]+=elapsed
            return measured
        setattr(obj,name,wrap(original,label))

