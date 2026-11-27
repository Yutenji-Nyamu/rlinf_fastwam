"""Decide whether another restart is warranted after a terminal GPU allocation error."""
import os
import re
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

# Ignore discovery contexts. A peer using at least 1 GiB can plausibly affect these H100 jobs.
MATERIAL_MIB = 1024
SIZE = r"([\d.]+)\s*(bytes?|KiB|MiB|GiB|TiB)"
PEER = re.compile(r"\bProcess\s+(\d+)\s+has\s+" + SIZE + r"\s+memory in use", re.I)
SCALE = {"byte": 1 / 1048576, "bytes": 1 / 1048576, "kib": 1 / 1024,
         "mib": 1, "gib": 1024, "tib": 1048576}


def amount(match):
    return float(match[-2]) * SCALE[match[-1].lower()]


def owner_uid(pid):
    try:
        return (Path("/proc") / str(pid)).stat().st_uid
    except OSError:
        return None


def card_snapshot(cards):
    """One bounded query at failure. A driver stuck in uninterruptible sleep must not stall the queue."""
    try:
        proc = subprocess.Popen(["nvidia-smi", "-i", ",".join(map(str, cards)), "-q", "-x"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return {}
    try:
        output, _ = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=.3)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return {}
    finally:
        proc.stdout.close()
    if proc.returncode:
        return {}
    try:
        result = {}
        for gpu in ET.fromstring(output).findall("gpu"):
            card = int(gpu.findtext("minor_number"))
            if card not in cards:
                continue
            entries = []
            for item in gpu.findall("processes/process_info"):
                size = (item.findtext("used_memory") or "").split()
                if len(size) == 2 and size[1] == "MiB" and size[0].isdigit():
                    pid = int(item.findtext("pid"))
                    entries.append((pid, owner_uid(pid), int(size[0])))
            result[card] = entries
        return result
    except (ValueError, TypeError, ET.ParseError):
        return {}


def retry_delay(evidence, cards, previous_retries, contention_streak, *, snapshot=None):
    """A delay of None hands off. Never infer a physical card from a CUDA logical device number."""
    cards = tuple(map(int, cards))
    line = evidence.get("line", "")
    native = re.search(r"native-g(\d+)(?:[-.]|$)", str(evidence.get("log", "")))
    if native and int(native[1]) in cards:
        cards = (int(native[1]),)
    peers = {int(p): amount((qty, unit)) for p, qty, unit in PEER.findall(line)}
    requested = re.search(r"Tried to allocate\s+" + SIZE, line, re.I)
    free = re.search(r"of which\s+" + SIZE + r"\s+is free", line, re.I)
    need = max(MATERIAL_MIB, amount(requested.groups()) - (amount(free.groups()) if free else 0)) if requested else MATERIAL_MIB
    rows = card_snapshot(cards) if snapshot is None else snapshot
    # The emitting rank or a peer named by PyTorch can identify the fault card
    # even when CUDA lists that card as logical device 0 on both workers.
    source_pid = re.search(r"\bpid\s*=\s*(\d+)", line, re.I)
    ids = set(peers)
    if source_pid:
        ids.add(int(source_pid[1]))
    located = [{g for g in cards if any(pid == ident for pid, _, _ in rows.get(g, []))}
               for ident in ids]
    located = [group for group in located if group]
    if located:
        shared = set.intersection(*located)
        if len(shared) == 1:
            cards = tuple(shared)
    ours = os.getuid()
    known = {p: uid for g in cards for p, uid, _ in rows.get(g, []) if uid is not None}
    # PyTorch enumerates the actual other allocations on the failing GPU. A vanished
    # PID may have freed its context before this query. Its account remains unknown.
    accounted = {p: known[p] if p in known else owner_uid(p) for p in peers}
    foreign_log = sum(v for p,v in peers.items() if accounted[p] is not None and accounted[p] != ours)
    unknown_log = sum(v for p,v in peers.items() if accounted[p] is None)
    foreign = {g:sum(m for _,uid,m in rows.get(g,[]) if uid is not None and uid != ours) for g in cards}
    if foreign_log >= need:
        external, reason = True, "other_user_allocation_in_fault_report"
    elif foreign and min(foreign.values()) >= need:
        external, reason = True, "other_user_allocation_on_fault_card" if len(cards)==1 else "other_user_allocations_on_both_cards"
    else:
        external, reason = False, "oom_without_attributed_pressure"
    # In a pair with no reliable failing-card identifier, a co-tenant on just one
    # member is a plausible lead. Give it one more chance without declaring it causal.
    plausible = unknown_log >= need or (len(cards)>1 and max(foreign.values(),default=0)>=need)
    if plausible and not external:
        reason = "unattributed_peer_or_uncertain_fault_card"
    if external:
        delay = (30,60,120)[min(max(contention_streak,0),2)]
    elif previous_retries < (2 if plausible else 1):
        delay = 30 if previous_retries == 0 else 60
    else:
        delay = None
    return delay, dict(external=external,reason=reason,foreign_mib=foreign,
                       reported_foreign_mib=round(foreign_log),reported_unattributed_mib=round(unknown_log))

