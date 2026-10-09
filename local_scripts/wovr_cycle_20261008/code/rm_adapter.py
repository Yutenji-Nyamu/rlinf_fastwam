"""lift_pot reward adapter; each offline stage supplies its new checkpoint."""
from rm_inference import SingleTaskReward, sha256


class LiftPotReward(SingleTaskReward):
    def __init__(self, checkpoint_path):
        super().__init__(checkpoint_path=checkpoint_path)
        if (self.metadata['task_name'],self.metadata.get('task_config')) != ('lift_pot','demo_clean'):
            raise ValueError('Require the lift_pot/demo_clean reward checkpoint')
        self.float().eval().requires_grad_(False)
        self.checkpoint_sha256=sha256(checkpoint_path)

    def deployment_metadata(self):
        return dict(reward_architecture='resnet18_mlp256',reward_task='lift_pot',
            reward_task_config='demo_clean',reward_checkpoint_sha256=self.checkpoint_sha256,
            reward_threshold_owner='RLinf environment',
            reward_instructions='accepted; ignored by the single-task model')
