"""Reuse the seven owner contracts, plus the new normal-reborrow gate."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import rynn_formal_owner_v2 as owner
import test_rynn_owner as inherited

inherited.owner = owner


class ReturnedReborrowTests(unittest.TestCase):
    def test_only_completed_normal_return_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / 'runs/rynn-success-v1'
            parent.mkdir(parents=True)
            cycle = root / 'cycle'
            child = root / 'child'
            cycle.mkdir()
            child.mkdir()
            def write(path, value):
                path.write_text(json.dumps(value))
            write(parent / 'owner-plan.json', {})
            write(parent / 'final.json', dict(terminal_status='failed', recovery_error=None,
                rlt_borrowed=True, rlt_return_dispatched=True))
            write(parent / 'owner-identity.json', dict(pid=10, start=20, uid=20001))
            write(parent / 'cleanup.json', dict(all_stopped=True))
            write(cycle / 'plan.json', dict(children={'gpu4': dict(path=str(child))}))
            write(child / 'plan.json', dict(completed_owner=str(parent)))
            plan = dict(lifecycle_path=str(cycle), returned_reborrow=dict(parent_owner=str(parent),
                owner_plan_sha256=owner.sha(parent / 'owner-plan.json'), final_sha256=owner.sha(parent / 'final.json')))
            with patch.object(owner, 'S', root):
                self.assertTrue(owner.verify_returned_reborrow(plan, types.SimpleNamespace(same=lambda _: False)))
                with self.assertRaises(AssertionError):
                    owner.verify_returned_reborrow(plan, types.SimpleNamespace(same=lambda _: True))
                write(child / 'plan.json', dict(completed_owner=str(parent), adopted_from={'cycle': 'old'}))
                with self.assertRaises(AssertionError):
                    owner.verify_returned_reborrow(plan, types.SimpleNamespace(same=lambda _: False))


def load_tests(loader, tests, pattern):
    suite = loader.loadTestsFromTestCase(inherited.OwnerContractTests)
    suite.addTests(loader.loadTestsFromTestCase(ReturnedReborrowTests))
    return suite


if __name__ == '__main__':
    unittest.main()
