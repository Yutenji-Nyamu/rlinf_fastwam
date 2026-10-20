"""Transparent English object spans; exact text/offsets retained for reannotation.

No simulator object labels or success outcomes are used. Roles are a named
language heuristic, not ground-truth phases. Missing nouns stay invalid.
"""
import re
import numpy as np

NOUNS = (
    "alarm clock", "alarmclock", "pill bottle", "pillbottle", "playing card", "playingcard",
    "plastic box", "plasticbox", "qr code", "qrcode", "trash bin", "dustbin",
    "microwave", "stapler", "container", "cabinet", "basket", "skillet",
    "bottle", "bottles", "block", "blocks", "bowl", "bowls", "hammer", "bell",
    "bin", "bins", "roller", "microphone", "mic", "mug", "pot", "can", "cans",
    "pad", "card", "cards", "laptop", "bread", "burger", "fries", "plate",
    "shoe", "shoes", "cup", "cups", "fan", "mouse", "object", "objects",
    "scale", "stand", "phone", "stamp", "seal", "switch", "box", "tray",
    "apple", "banana", "orange", "ball", "cube", "cubes", "ring", "biscuit",
    "brush", "toy", "book", "pen", "dish", "table", "rack", "lid", "handle",
)
ADJECTIVES = re.compile(r"(?:(?:the|a|an|red|green|blue|yellow|white|black|orange|purple|pink|"
                        r"left|right|middle|small|smaller|smallest|big|bigger|biggest|large|larger|"
                        r"largest|empty|full|wooden|plastic|metal|first|second|third|top|bottom)\s+)+$", re.I)
PATTERN = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(x) for x in sorted(NOUNS, key=len, reverse=True)) + r")(?!\w)", re.I)


def annotate(row):
    full = row["full_prompt"]
    ib, ie = row["instruction_span"]
    text = full[ib:ie]
    found = []
    for match in PATTERN.finditer(text):
        noun_span = [ib + match.start(), ib + match.end()]
        left = text[:match.start()]
        modifiers = ADJECTIVES.search(left)
        phrase_span = [ib + (modifiers.start() if modifiers else match.start()), noun_span[1]]
        # A relation after the previous noun identifies a goal mention. Literal
        # role evidence is saved; later plotting can correct ambiguous phrases.
        previous_end = found[-1]["noun_span"][1] - ib if found else 0
        relation_text = text[previous_end:match.start()]
        relation = re.findall(r"\b(to|into|onto|on|in|inside|above|under|over)\b", relation_text, re.I)
        role = "goal" if found and relation else "source"
        found.append(dict(noun=match.group().lower(), noun_span=noun_span, phrase_span=phrase_span,
                          phrase=full[slice(*phrase_span)], role=role,
                          role_evidence=relation_text, role_method="surface-preposition-v1"))
    labels = ("all_noun", "source_noun", "goal_noun", "all_phrase", "source_phrase", "goal_phrase")
    spans = {}
    for label in labels:
        role, kind = label.split("_")
        spans[label] = [item[f"{kind}_span"] for item in found if role == "all" or item["role"] == role]
    matched = {label: [] for label in labels}
    crossing = {label: [] for label in labels}
    for label in labels:
        for tid, (b, e) in enumerate(row["offsets"]):
            # SentencePiece can attach leading whitespace. Ignore only that
            # whitespace, never a partial word outside the object span.
            token_text = full[b:e]
            bb, ee = b + len(token_text) - len(token_text.lstrip()), e - (len(token_text) - len(token_text.rstrip()))
            if ee <= bb:
                continue
            if any(bb >= s and ee <= t for s, t in spans[label]):
                matched[label].append(tid)
            elif any(bb < t and ee > s for s, t in spans[label]):
                crossing[label].append(tid)
    return dict(objects=found, labels=list(labels), token_indices=matched, boundary_crossing=crossing,
                valid={k: bool(v) for k, v in matched.items()}, role_is_heuristic=True,
                matching="whole-token containment, whitespace-only boundary trim; repeated mentions union",
                unmatched_instruction=not found)


def masks(metadata, key_valid):
    annotations = [annotate(row) for row in metadata["token_rows"]]
    labels = annotations[0]["labels"]
    mask = np.zeros((*key_valid.shape, len(labels)), dtype=bool)
    start = metadata["language_start"]
    for b, annotation in enumerate(annotations):
        for j, label in enumerate(labels):
            mask[b, [start + i for i in annotation["token_indices"][label]], j] = True
    mask &= key_valid[..., None]
    return mask, annotations, labels
