"""Use RLinf's normal update and evaluator; isolate the two post-update evaluations."""
import sys,time
from pathlib import Path
def install(Runner,d,phase):
    sys.path.insert(0,str(d))
    from comparison import evaluation,write
    original_run=Runner.run
    def score(self, arm):
        start=time.time()
        self.update_rollout_weights()
        metrics=self.evaluate()
        rows=evaluation(d,start)
        write(d/f"{arm}-eval.json",dict(arm=arm,round=self.global_step,rows=rows,metrics={k:float(v) for k,v in metrics.items()}))
    if phase=="reference_eval":
        def run(self):
            assert self.global_step==1
            score(self,"a")
            return self._finish_run()
        Runner.run=run
    elif phase=="candidate_one":
        original_finish=Runner._finish_run
        def finish(self):
            assert self.global_step==1
            score(self,"b")
            return original_finish(self)
        def run(self):
            assert self.global_step==0 and self.max_steps==200
            self.max_steps=1
            return original_run(self)
        # Same CP1 save as the reference. Evaluation follows step logging.
        def save_step(self,step):
            assert self.global_step==1
            self._save_checkpoint()
            return {}
        Runner._maybe_eval_and_checkpoint=save_step
        Runner._finish_run=finish
        Runner.run=run

