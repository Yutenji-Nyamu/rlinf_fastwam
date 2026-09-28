"""Focused checks: task/path substitutions and exactly one N4-to-N8 delta."""
def diff(old, new, path=''):
    if isinstance(old, dict) and isinstance(new, dict):
        assert set(old) == set(new), ('Config key-set changed', path)
        return [d for key in old for d in diff(old[key], new[key], path + '.' + key if path else key)]
    if isinstance(old, list) and isinstance(new, list):
        assert len(old) == len(new), ('Config list length changed', path)
        return [d for i, (a, b) in enumerate(zip(old, new)) for d in diff(a, b, f'{path}[{i}]')]
    return [] if old == new else [{'key': path, 'old': old, 'new': new}]


def validate(role, old, new, task):
    common = {'runner.logger.log_path', 'runner.logger.experiment_name'}
    formal = {'env.train.total_num_envs', 'rollout.rlt_feature_model.model_path',
              'rollout.rlt_feature_model.openpi_data.default_prompt'}
    formal |= {f'env.{phase}.{suffix}' for phase in ('train', 'eval')
               for suffix in ('task_config.task_name', 'seeds_path', 'task_config.save_path',
                              'video_cfg.video_base_dir')}
    stage1 = {'data.train_data_paths[0].dataset_path', 'actor.model.openpi_data.default_prompt'}
    changes = diff(old, new)
    allowed = common | (stage1 if role == 'stage1-full' else formal)
    assert all(row['key'] in allowed for row in changes), ('Unexpected config delta', changes)
    numeric = [row for row in changes if not isinstance(row['old'], str) or not isinstance(row['new'], str)]
    if role == 'stage1-full':
        assert not numeric, numeric
        assert new['runner']['max_steps'] == 2000
        assert new['actor']['global_batch_size'] == 32
        assert new['actor']['micro_batch_size'] == 16
    else:
        assert numeric == [{'key': 'env.train.total_num_envs', 'old': 4, 'new': 8}], numeric
        assert new['runner']['max_steps'] == new['runner']['max_epochs'] == 3000
        assert new['env']['train']['rollout_epoch'] == 1
        assert new['env']['eval']['total_num_envs'] == 4
        assert new['env']['eval']['rollout_epoch'] == 5
        assert new['algorithm']['group_size'] == 1
        assert new['algorithm']['dv_observe'] is True
        for phase in ('train', 'eval'):
            env = new['env'][phase]
            assert env['task_config']['task_name'] == task
            assert env['max_steps_per_rollout_epoch'] == env['max_episode_steps'] == env['task_config']['step_lim'] == 200
    return changes
