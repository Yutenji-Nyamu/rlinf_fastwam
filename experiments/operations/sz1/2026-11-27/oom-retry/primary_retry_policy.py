"""Retry policy for a primary job explicitly reserved by the queue."""
import re


def terminal_reason(released):
    terminal = released.get("result", released)
    phase = terminal.get("phase", "")
    if phase in ("COMPLETE", "COMPLETE_RETURN_RLT", "TARGET_REACHED"):
        return "TARGET_REACHED"
    if phase in ("OPERATOR_STOP", "CANCELLED", "INTERRUPTED") or re.search(
        r"(?m)^\s*(?:KeyboardInterrupt|SystemExit:\s*(?:130|143))\s*$",
        str(terminal.get("error", "")),
    ):
        return "OPERATOR_STOP"
    return None


def failure_evidence(released):
    result = released.get("result", released)
    lines = str(result.get("error", "")).strip().splitlines()
    return {"kind": "unexpected_exit", "line": lines[-1] if lines else result.get("phase", "owner exited")}
