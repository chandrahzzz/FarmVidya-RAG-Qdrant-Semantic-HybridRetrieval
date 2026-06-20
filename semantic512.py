"""
rechunk.py
===========
Reads the new muralichunks.txt (no Chunk N === headers).

Rules:
  - A line is a QUESTION if it contains '?' (after stripping)
  - A line is a HEADING if it matches a section header pattern (numbered,
    no '?', very short or ends with ':') — skip these
  - A line is an ANSWER/CONTENT otherwise
  - One chunk = 1 question line + all immediately-following content lines
    until the next question or heading
  - Orphan question (no content follows) → discard
  - Content with no preceding question → discard

Output: slumber_cache/rechunked.json
"""

import re
import json
import os

SOURCE = "muralichunks.txt"
OUT    = "slumber_cache/rechunked.json"
os.makedirs("slumber_cache", exist_ok=True)

# ── patterns ──────────────────────────────────────────────────────────────────
# Section heading: starts with a number+dot pattern, NO '?', short or ends ':'
HEADING_RE = re.compile(
    r"^\s*\d+[\d\.]*\s+[^?]{1,120}[:]\s*$"          # "1.27. జీవామృతం"  ends with colon
    r"|^\s*\d+[\d\.]*\s+[^?]{1,80}\s*$"              # "1.ప్రస్తుత వ్యవసాయ పరిస్థితులు"
)

def is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if "?" in s:
        return False
    # must start with a number+dot
    if not re.match(r"^\d+[\d\.]*\s", s):
        return False
    # short line (≤ 12 words) with no sentence-level content → heading
    words = s.split()
    # if it's just a label line (≤ 10 words, no comma, no verb clues)
    # longer lines are actual content
    if len(words) <= 10:
        return True
    return False

def is_question(line: str) -> bool:
    s = line.strip()
    if "?" not in s:
        return False
    # Reject lines that are just a sentence fragment (no number prefix, very short)
    # e.g. "గురించి వివరించండి?" is a dangling continuation, not a full question
    words = s.split()
    if len(words) < 3:
        return False
    return True

def is_blank(line: str) -> bool:
    return not line.strip()

# ── parse ─────────────────────────────────────────────────────────────────────
def parse(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        raw_lines = f.readlines()

    lines = [l.rstrip("\n") for l in raw_lines]

    chunks   : list[str] = []
    cur_q    : str | None = None
    cur_body : list[str]  = []

    def flush():
        nonlocal cur_q, cur_body
        if cur_q and cur_body:
            body_text = "\n".join(cur_body).strip()
            if body_text:
                chunks.append(cur_q.strip() + "\n" + body_text)
        cur_q    = None
        cur_body = []

    for line in lines:
        if is_blank(line):
            continue

        if is_heading(line):
            flush()
            continue

        if is_question(line):
            flush()
            cur_q    = line.strip()
            cur_body = []
        else:
            # content / answer line
            if cur_q is not None:
                cur_body.append(line.strip())
            # else: orphan answer with no question — discard

    flush()  # handle last chunk
    return chunks

# ── main ──────────────────────────────────────────────────────────────────────
chunks = parse(SOURCE)

# sanity: drop any chunk with empty body
chunks = [c for c in chunks if len(c.split("\n")) >= 2 and len(c.split()) > 5]

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

# Stats only (no Telugu printing to avoid cp1252 errors)
lengths = [len(c.split()) for c in chunks]
bad     = [c for c in chunks if "\n" not in c]
print(f"Total Q+A chunks : {len(chunks)}")
print(f"Chunks no answer : {len(bad)}")
print(f"Words min/max/avg: {min(lengths)} / {max(lengths)} / {sum(lengths)//len(lengths)}")
print(f"Saved to {OUT}")
