"""Push the one pending RLT-DVAC telemetry fix through the existing relay."""

import sys

sys.path.insert(0, r"E:\Codex\home\tools\autodl-ssh-py312")

import push_rlt_via_ephemeral_reverse_proxy as proxy


proxy.REPO = "/root/autodl-tmp/RLinf_rlt_teacher_dvac"
proxy.BRANCH = "codex/rlt-teacher-dvac-weighting"
proxy.EXPECTED_HEAD = "a85b101bfd905f6d1e0700ae6c3ef1e4fb0ecec4"
proxy.EXPECTED_REMOTE = "513dbcb7f31ebb639afa5267dff218028d07187b"
proxy.EXPECTED_AHEAD = 1


if __name__ == "__main__":
    proxy.main()
