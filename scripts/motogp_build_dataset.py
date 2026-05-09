"""Build training JSONL from MotoGP raw snapshots.

Walks datasets/motogp/raw/**/*.json. For each (event_id, session_id),
keeps the latest snapshot only (or the one with status='F' if present).
Emits two example types per session:

  1. Q&A pairs (pole, fastest lap, gap to leader for rider X, ...)
  2. Narrative — minified raw JSON -> a one-paragraph English summary

Output schema matches datasets/train.jsonl: {"messages": [...]}.

  python scripts/motogp_build_dataset.py \
      --raw-dir datasets/motogp/raw \
      --out datasets/motogp/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


def latest_per_session(raw_dir: Path) -> Iterable[dict]:
    """Yield one payload per (event_id, session_id), preferring status='F'."""
    best: dict[tuple[str, str], tuple[int, dict]] = {}
    for path in raw_dir.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        head = payload.get("head") or {}
        key = (str(head.get("event_id", "")), str(head.get("session_id", "")))
        # rank: status='F' > anything else; tie-break by mtime
        rank = 1 if (head.get("session_status_id") or "").upper() == "F" else 0
        score = (rank, path.stat().st_mtime_ns)
        prev = best.get(key)
        if prev is None or score > prev[0]:
            best[key] = (score, payload)
    for _, payload in best.values():
        yield payload


def msg_pair(user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def session_label(head: dict) -> str:
    circuit = head.get("circuit_name") or "the circuit"
    sname = head.get("session_name") or head.get("session_shortname") or "the session"
    event = head.get("event_tv_name") or head.get("event_shortname") or ""
    date = head.get("date") or ""
    bits = [event, sname, f"at {circuit}"]
    if date:
        bits.append(f"on {date}")
    return ", ".join(b for b in bits if b)


def riders_sorted(payload: dict) -> list[dict]:
    riders = (payload.get("rider") or {}).values()
    classified = [r for r in riders if (r.get("status_name") or "").upper() == "CL"]
    classified.sort(key=lambda r: r.get("pos") or 9999)
    return classified


def lap_time_str(r: dict) -> str:
    return r.get("lap_time") or "no time"


def rider_full(r: dict) -> str:
    name = r.get("rider_name") or ""
    surname = r.get("rider_surname") or ""
    full = f"{name} {surname}".strip().title() if name or surname else (
        r.get("rider_shortname") or "Unknown"
    )
    return full


def qa_examples(payload: dict) -> list[dict]:
    head = payload.get("head") or {}
    label = session_label(head)
    rs = riders_sorted(payload)
    out: list[dict] = []

    if not rs:
        return out

    p1 = rs[0]
    out.append(msg_pair(
        f"Who set the fastest time at {label}?",
        f"{rider_full(p1)} (#{p1.get('rider_number')}, "
        f"{p1.get('team_name')}) with {lap_time_str(p1)}.",
    ))
    out.append(msg_pair(
        f"What was the pole lap time at {label}?",
        f"{lap_time_str(p1)}, set by {rider_full(p1)} on lap {p1.get('num_lap')}.",
    ))

    podium = rs[:3]
    if len(podium) >= 3:
        lines = []
        for r in podium:
            lines.append(
                f"P{r.get('pos')}: {rider_full(r)} ({r.get('team_name')}) "
                f"{lap_time_str(r)} (gap {r.get('gap_first')})"
            )
        out.append(msg_pair(
            f"Top three at {label}?",
            "\n".join(lines),
        ))

    for r in rs[:10]:
        full = rider_full(r)
        out.append(msg_pair(
            f"What was {full}'s gap to the leader at {label}?",
            f"+{r.get('gap_first')} (P{r.get('pos')}, best lap {lap_time_str(r)}).",
        ))
        out.append(msg_pair(
            f"How many laps did {full} complete at {label}?",
            f"{r.get('num_lap')} laps; best {lap_time_str(r)} on bike "
            f"{r.get('bike_name')}.",
        ))

    teams: dict[str, list[dict]] = {}
    for r in rs:
        teams.setdefault(r.get("team_name") or "Unknown", []).append(r)
    for team, members in teams.items():
        if len(members) >= 2:
            members.sort(key=lambda r: r.get("pos") or 9999)
            summary = ", ".join(
                f"{rider_full(r)} P{r.get('pos')} ({lap_time_str(r)})" for r in members
            )
            out.append(msg_pair(
                f"How did {team} perform at {label}?",
                summary + ".",
            ))

    return out


def narrative_example(payload: dict) -> dict:
    head = payload.get("head") or {}
    label = session_label(head)
    rs = riders_sorted(payload)
    if not rs:
        return msg_pair(
            f"Summarize this MotoGP session in one paragraph:\n{json.dumps(payload, ensure_ascii=False)}",
            "Session has no classified riders to summarize.",
        )

    p1 = rs[0]
    p2 = rs[1] if len(rs) > 1 else None
    p3 = rs[2] if len(rs) > 2 else None
    most_laps = max(rs, key=lambda r: r.get("num_lap") or 0)

    parts = [
        f"At {label}, {rider_full(p1)} ({p1.get('team_name')}, {p1.get('bike_name')}) "
        f"set the fastest lap of {lap_time_str(p1)} on lap {p1.get('num_lap')}.",
    ]
    if p2:
        parts.append(
            f"{rider_full(p2)} ({p2.get('team_name')}) was second, "
            f"+{p2.get('gap_first')} behind."
        )
    if p3:
        parts.append(
            f"{rider_full(p3)} ({p3.get('team_name')}) completed the top three, "
            f"+{p3.get('gap_first')}."
        )
    parts.append(
        f"{rider_full(most_laps)} logged the highest lap count with "
        f"{most_laps.get('num_lap')} laps."
    )
    parts.append(f"Total classified riders: {len(rs)}.")

    summary = " ".join(parts)
    prompt = (
        "Summarize this MotoGP session timing snapshot in one paragraph:\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    return msg_pair(prompt, summary)


def build(raw_dir: Path, out_path: Path) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_sessions = 0
    n_examples = 0
    with out_path.open("w", encoding="utf-8") as fp:
        for payload in latest_per_session(raw_dir):
            n_sessions += 1
            for ex in qa_examples(payload):
                fp.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n_examples += 1
            fp.write(json.dumps(narrative_example(payload), ensure_ascii=False) + "\n")
            n_examples += 1
    return n_sessions, n_examples


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw-dir", type=Path, default=Path("datasets/motogp/raw"))
    p.add_argument("--out", type=Path, default=Path("datasets/motogp/train.jsonl"))
    args = p.parse_args(argv)

    if not args.raw_dir.exists():
        print(f"raw dir not found: {args.raw_dir}", file=sys.stderr)
        return 1

    n_sessions, n_examples = build(args.raw_dir, args.out)
    print(f"wrote {args.out} ({n_sessions} sessions, {n_examples} examples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
