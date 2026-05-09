# OpenLibrary Ingestion

Continuous crawler that fills the BookMind `books` collection from
[Open Library](https://openlibrary.org). Walks 80+ subject catalogs, enriches
each work with metadata, computes a 384-dim embedding, and upserts into
MongoDB Atlas. Crawl progress is persisted so the daemon can be stopped and
resumed safely, and stale subjects are periodically re-crawled to pick up
newly added works.

The ingest writes the same `embedding` field that the recommender's
`vs_books_embedding` Atlas Vector Search index reads from — newly ingested
books are immediately searchable from the **Books** screen.

## Architecture

```
Open Library /subjects/<slug>.json
    │
    ▼  paginate (20/page, 0.5 s rate-limit)
fetch_work_details(/works/OLxxxW.json)        ← description, subjects, cover
    │
    ▼  classify subjects → Genres / Themes / Moods
sentence-transformers/all-MiniLM-L6-v2        ← 384-dim embedding
    │
    ▼  upsert by OlKey
MongoDB books + ol_ingest_state
```

Per-subject crawl state lives in `ol_ingest_state` (offset, total, completed,
last_updated). Existing books are skipped via `OlKey`, so re-running is
cheap.

## Files

| Path | Purpose |
|---|---|
| [scripts/ol_ingest.py](https://github.com/Azayzel/my-lm/blob/main/scripts/ol_ingest.py) | Crawler / daemon |
| [scripts/install_ol_ingest_service.ps1](https://github.com/Azayzel/my-lm/blob/main/scripts/install_ol_ingest_service.ps1) | NSSM service installer |
| [src/mylm/rag/open_library.py](https://github.com/Azayzel/my-lm/blob/main/src/mylm/rag/open_library.py) | OL API client + `GENRE_TO_OL_SUBJECT` map |
| `data/ol_ingest_status.json` | Heartbeat (status, current subject, totals) |
| `logs/ol_ingest.{out,err}.log` | Service logs (rotated at 10 MB) |

## CLI flags

```
python scripts/ol_ingest.py [options]

  --subjects "slug1,slug2"       Crawl a subset (default: all 80+)
  --per-subject N                Max new books per subject per pass (default: 200)
  --dry-run                      Log without writing to MongoDB
  --reset                        Drop ol_ingest_state and start over
  --daemon                       Loop continuously
  --sleep-hours H                Hours between daemon passes (default: 24)
  --refresh-after-days D         Re-crawl subjects whose state is older than D
                                 days, so newly added OL works are picked up
                                 (default: 30, set 0 to disable)
  --heartbeat-path PATH          Override heartbeat JSON location
  --no-heartbeat                 Disable heartbeat writing
```

## Running as a Windows service (recommended)

The installer wraps the daemon with [NSSM](https://nssm.cc) for autostart,
crash-restart, and log rotation.

### Prerequisites

```powershell
# 1. NSSM (one-time install)
winget install nssm     # or: choco install nssm  /  scoop install nssm

# 2. .env at repo root with MONGODB_URI + MONGODB_DB
```

### Install / start / stop

All actions require an **elevated** PowerShell.

```powershell
# Install (registers + configures the service, does not start it)
pwsh -File scripts\install_ol_ingest_service.ps1 -Action install

# Start
pwsh -File scripts\install_ol_ingest_service.ps1 -Action start

# Status (service state + heartbeat)
pwsh -File scripts\install_ol_ingest_service.ps1 -Action status

# Stop / uninstall
pwsh -File scripts\install_ol_ingest_service.ps1 -Action stop
pwsh -File scripts\install_ol_ingest_service.ps1 -Action uninstall
```

### Tuning at install time

```powershell
pwsh -File scripts\install_ol_ingest_service.ps1 -Action install `
    -SleepHours       1 `      # how often to wake between passes
    -PerSubject       100 `    # books per subject per pass
    -RefreshAfterDays 14       # re-crawl cadence
```

Re-running with `-Action install` re-creates the service from scratch using
the new flags.

## Tuning guide

The OL client rate-limits at 0.5 s between requests in
[open_library.py](https://github.com/Azayzel/my-lm/blob/main/src/mylm/rag/open_library.py); each new book costs
~2 OL calls (catalog page + work details), so steady-state throughput is
about **1 book / 1.5 s**. Plan from there:

| Profile | `--per-subject` | `--sleep-hours` | `--refresh-after-days` | Approx books/day |
|---|---:|---:|---:|---:|
| Conservative (default) | 200 | 24 | 30 | ~3,000 |
| **Aggressive but safe** | **100** | **1** | **14** | **~30,000** |
| Burst (one-off backfill) | 500 | n/a (no `--daemon`) | n/a | one-shot |

"Aggressive but safe" cycles through all 80+ subjects in ~3.5 h then sleeps
1 h, never exceeds OL's 0.5 s rate, and refreshes every subject every two
weeks.

## Monitoring

### Heartbeat JSON

The daemon writes `data/ol_ingest_status.json` every time it changes
subject:

```jsonc
{
  "status": "running",          // running | idle | done | stopped
  "pass_num": 4,
  "pass_started_at": "2026-05-09T14:02:11+00:00",
  "current_subject": "epic_fantasy",
  "subject_index": 17,
  "subject_total": 83,
  "totals": { "inserted": 412, "updated": 1820, "skipped": 9043, "errors": 2 },
  "timestamp": "2026-05-09T14:38:02+00:00"
}
```

The **Books** screen surfaces this live (status pill + counts), so you can
verify the service is healthy without leaving the app.

### Logs

```powershell
Get-Content -Tail 50 -Wait logs\ol_ingest.out.log
Get-Content -Tail 50 -Wait logs\ol_ingest.err.log
```

NSSM rotates files at 10 MB.

## Graceful shutdown

The daemon installs SIGINT / SIGTERM / SIGBREAK handlers and exits cleanly
between books, so:

- `Ctrl+C` from the foreground process — finishes the current book, saves
  state, exits.
- `Stop-Service BookMindOLIngest` — NSSM sends Ctrl+Break with a 30 s grace
  window before forcing termination.

State in MongoDB is always consistent — interruption never loses progress.

## Operational notes

- **Single-process by design.** Don't run two instances against the same
  database; they'd race on the `ol_ingest_state` documents. To parallelize,
  partition by `--subjects` across instances.
- **OL throttling.** Don't drop the 0.5 s rate-limit. OL will start
  returning 429s and the crawler doesn't retry — it just logs and skips.
- **Subject coverage.** [`INGEST_SUBJECTS`](https://github.com/Azayzel/my-lm/blob/main/scripts/ol_ingest.py) holds
  the ingest-only superset; [`GENRE_TO_OL_SUBJECT`](https://github.com/Azayzel/my-lm/blob/main/src/mylm/rag/open_library.py)
  is the user-genre → slug lookup the recommender uses. Add to the former to
  expand crawl coverage; add to the latter only if a user-facing genre needs
  a new mapping.
