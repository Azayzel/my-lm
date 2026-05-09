"""Poll MotoGP livetiming endpoint, snapshot to disk.

Long-running. Polls every --fast-interval s when a session is active,
--slow-interval s otherwise. Dedups via (event_id, session_id, remaining).

  python scripts/motogp_poll.py --out-dir datasets/motogp/raw
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ENDPOINT = "https://api.motogp.pulselive.com/motogp/v1/timing-gateway/livetiming-lite"
UA = "my-lm/motogp-poll (research; non-redistribution)"


def fetch(session: requests.Session, timeout: float = 15.0) -> dict | None:
    try:
        r = session.get(ENDPOINT, timeout=timeout, headers={"User-Agent": UA})
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] fetch failed: {exc}",
              file=sys.stderr)
        return None
    except json.JSONDecodeError as exc:
        print(f"bad json: {exc}", file=sys.stderr)
        return None


def snapshot_key(payload: dict) -> tuple[str, str, str]:
    head = payload.get("head") or {}
    return (
        str(head.get("event_id", "")),
        str(head.get("session_id", "")),
        str(head.get("remaining", "")),
    )


def is_live(payload: dict) -> bool:
    head = payload.get("head") or {}
    status = (head.get("session_status_id") or "").upper()
    # F = Finished. Anything else (L/R/P or empty) = treat as active.
    return status != "F"


def write_snapshot(payload: dict, out_dir: Path) -> Path:
    head = payload.get("head") or {}
    date = (head.get("date") or "unknown").replace("/", "-")
    event = head.get("event_id") or "noevent"
    session = head.get("session_id") or "nosess"
    epoch = int(time.time())
    day_dir = out_dir / date
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{event}_{session}_{epoch}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=Path("datasets/motogp/raw"))
    p.add_argument("--fast-interval", type=float, default=5.0,
                   help="seconds between polls during live session")
    p.add_argument("--slow-interval", type=float, default=60.0,
                   help="seconds between polls when no live session")
    p.add_argument("--once", action="store_true", help="single fetch then exit")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    last_key: tuple[str, str, str] | None = None

    while True:
        payload = fetch(sess)
        if payload is not None:
            key = snapshot_key(payload)
            if key != last_key:
                path = write_snapshot(payload, args.out_dir)
                head = payload.get("head") or {}
                print(
                    f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] "
                    f"wrote {path.name} "
                    f"({head.get('event_shortname')}/{head.get('session_shortname')} "
                    f"status={head.get('session_status_id')} "
                    f"remaining={head.get('remaining')})"
                )
                last_key = key
            else:
                print(".", end="", flush=True)

            interval = args.fast_interval if is_live(payload) else args.slow_interval
        else:
            interval = args.slow_interval

        if args.once:
            return 0
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
