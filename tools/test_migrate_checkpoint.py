"""Run only on approved server; stdlib migration rejection checks."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import migrate_checkpoint as m


class MigrationChecks(unittest.TestCase):
    def inputs(self):
        old = {'env': {'task_config': {'save_path': str(m.OLD) + '/run/env-data', 'step_lim': 200},
                       'video_cfg': {'video_base_dir': str(m.OLD) + '/run/video'}},
               'core': {'n_base': 8}, 'model': {'is_lora': False}, 'formal': {'max_physical_actions': 20000},
               'evaluation': {'config': {'task': 'turn_switch'}},
               'port_source_manifest': {name: 'old' for name in m.CHANGED_SOURCE | {'core.py'}}}
        new = copy.deepcopy(old)
        new['env']['task_config']['save_path'] = str(m.NEW) + '/run/env-data'
        new['env']['video_cfg']['video_base_dir'] = str(m.NEW) + '/run/video'
        new['port_source_manifest'].update({name: 'new' for name in m.REVIEWED_SOURCE})
        return old, new

    def test_only_explicit_output_routes_and_lifecycle_source_allowed(self):
        old, new = self.inputs(); m.check_inputs(old, new)

    def test_existing_control_files_may_change_but_are_still_manifest_pinned(self):
        old, new = self.inputs()
        old['port_source_manifest']['tools/expo_formal_owner.py'] = 'old-owner'
        m.check_inputs(old, new)

    def test_missing_required_control_file_is_rejected(self):
        old, new = self.inputs()
        del new['port_source_manifest']['tools/launch_expo_repair.py']
        with self.assertRaises(ValueError): m.check_inputs(old, new)

    def test_contract_uses_real_paths_through_mount_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / 'home-nvme' / 'scope'; real.mkdir(parents=True)
            real = real.resolve()
            (real / 'inputs.json').write_text('{}')
            alias = root / 'data-scope'; alias.symlink_to(real, target_is_directory=True)
            expected = m.make_contract(real, 'input-sha')
            self.assertEqual(m.make_contract(alias, 'input-sha'), expected)
            self.assertEqual(expected['run'], str(real / 'run'))
            self.assertEqual(expected['inputs_path'], str(real / 'inputs.json'))
            (real / 'replay').mkdir()
            self.assertEqual((alias / 'replay').resolve(), real / 'replay')

    def test_method_budget_eval_and_backend_mutations_rejected(self):
        for mutation in ('model', 'core', 'formal', 'evaluation', 'source', 'outside'):
            with self.subTest(mutation=mutation):
                old, new = self.inputs()
                if mutation == 'source': new['port_source_manifest']['core.py'] = 'changed'
                elif mutation == 'outside': new['env']['task_config']['save_path'] = '/tmp/foreign'
                else: new[mutation]['unreviewed'] = True
                with self.assertRaises(ValueError): m.check_inputs(old, new)

    def test_source_removal_and_extra_helper_rejected(self):
        for extra in (False, True):
            old, new = self.inputs()
            if extra: new['port_source_manifest']['second.py'] = 'x'
            else: del new['port_source_manifest']['core.py']
            with self.assertRaises(ValueError): m.check_inputs(old, new)

    def test_replay_rebind_changes_only_identity(self):
        digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        original = dict(root='/old', root_identity=[1, 2, 3], root_id='retained',
                        rng=[4, 5], contract={'demo': 50}, online_entries=[])
        frozen = copy.deepcopy(original)
        new, proof = m.rebind_replay(original, Path('/new'), [1, 4, 3], digest)
        self.assertEqual(original, frozen)
        self.assertEqual(proof['changed_keys'], ['root', 'root_identity'])
        for key in ('root_id', 'rng', 'contract', 'online_entries'): self.assertEqual(new[key], original[key])

    def test_replay_requires_real_new_identity(self):
        with self.assertRaises(ValueError):
            m.rebind_replay(dict(root='/same', root_identity=[1, 2, 3]), Path('/same'), [1, 2, 3], repr)

    def test_nonzero_or_precommitted_progress_cannot_be_migrated_as_fresh(self):
        counters = dict.fromkeys(('episodes_completed', 'physical_actions', 'warmup_actions',
            'post_warmup_actions', 'carry_actions', 'pending_calls', 'completed_calls',
            'budget_truncated_episodes'), 0)
        state = dict(hashes=dict.fromkeys(m.PAYLOAD, 'sha'), cadence={'counters': counters},
            base={'base_updates': 0}, core={'_extra_state': dict.fromkeys(
                ('update_calls', 'critic_steps', 'editor_steps', 'temperature_steps'), 0)},
            replay=dict(online_entries=[], samples_q=0, samples_fm=0),
            progress=dict(evaluation_initial=False, evaluation_final=False,
                evaluation_periodic=[], evaluation_summaries={}, online_success=0,
                stopped_episodes=0, horizon_term_precedence=0, training_seed_sha256=None))
        m.check_zero_state(state, {'reason': 'fresh-base'}, {'online_entries': []})
        for kind in ('physical_actions', 'online', 'evaluation', 'learner', 'schema'):
            with self.subTest(kind=kind):
                broken = copy.deepcopy(state)
                if kind == 'physical_actions': broken['cadence']['counters']['physical_actions'] = 1
                elif kind == 'online': broken['replay']['online_entries'] = [{'episode': 0}]
                elif kind == 'evaluation': broken['progress']['evaluation_initial'] = True
                elif kind == 'learner': broken['core']['_extra_state']['critic_steps'] = 1
                else: del broken['cadence']['counters']['physical_actions']
                with self.assertRaises(ValueError):
                    m.check_zero_state(broken, {'reason': 'fresh-base'}, {'online_entries': []})


if __name__ == '__main__': unittest.main()
