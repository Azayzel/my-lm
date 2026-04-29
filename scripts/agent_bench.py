"""
agent_bench.py — Multi-turn agent benchmark for LLM inference.

Runs multi-turn, tool-calling tasks and measures what happens to latency,
memory, and reliability as context accumulates across turns.

Tasks:
  - code_debugger  : read buggy code, run tests, identify failures, write fixes
  - research_synth : read research docs, cross-reference, produce a report

Each task runs N trials in both **raw** mode (no optimisations, num_ctx=4096)
and **autotune** mode (dynamic KV management).  All timings come from Ollama's
internal Go nanosecond timers (prompt_eval_duration, load_duration,
total_duration) — not Python wall clocks.

Usage:
  # Quick run (2 tasks, 2 trials, ~12 min):
  python scripts/agent_bench.py --models llama3.2:3b \\
      --tasks code_debugger,research_synth --trials 2

  # Full run (all tasks, 5 trials):
  python scripts/agent_bench.py

  # Dry-run (validate task definitions, no Ollama required):
  python scripts/agent_bench.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "bench_fixtures"
RESULTS_DIR = REPO_ROOT / "benchmark_results"

KV_BUCKETS = (1024, 2048, 4096, 8192, 16384, 32768)

DEFAULT_MODELS = ["llama3.2:3b"]
DEFAULT_TRIALS = 5
OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TaskDef:
    """Definition of a benchmark task."""

    name: str
    system_prompt: str
    max_turns: int
    fixture_files: list[str]
    description: str = ""


@dataclass
class TurnMetrics:
    """Metrics for a single turn of inference."""

    turn: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_eval_duration_ns: int = 0
    eval_duration_ns: int = 0
    load_duration_ns: int = 0
    total_duration_ns: int = 0
    ttft_ms: float = 0.0
    model_reloaded: bool = False
    tool_call_error: bool = False
    wall_time_ms: float = 0.0


@dataclass
class TrialResult:
    """Aggregate result for one trial (one task × one condition)."""

    task: str
    model: str
    condition: str  # "raw" | "autotune"
    trial: int
    turns: list[TurnMetrics] = field(default_factory=list)
    total_wall_s: float = 0.0
    peak_ram_bytes: int = 0
    swap_events: int = 0
    context_tokens_final: int = 0
    task_success: bool = False
    error: str | None = None


@dataclass
class BenchmarkSummary:
    """Summary across all trials for a single condition."""

    task: str
    model: str
    condition: str
    n_trials: int = 0
    avg_ttft_ms: float = 0.0
    ttft_slope_ms_per_turn: float = 0.0
    avg_model_reloads: float = 0.0
    avg_wall_s: float = 0.0
    avg_peak_ram_gb: float = 0.0
    total_swap_events: int = 0
    tool_call_errors: int = 0
    success_rate: float = 0.0
    avg_context_tokens_final: float = 0.0


# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------


def _load_fixture(name: str) -> str:
    """Read a fixture file and return its contents."""
    path = FIXTURES / name
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return path.read_text(encoding="utf-8")


def build_task_catalogue() -> dict[str, TaskDef]:
    """Return the registry of benchmark tasks."""
    return {
        "code_debugger": TaskDef(
            name="code_debugger",
            description=(
                "Agent reads buggy Python files, runs tests, identifies "
                "failures, and writes fixes. Context grows with each file "
                "read and test output appended."
            ),
            system_prompt=(
                "You are an expert Python debugger. You have access to the "
                "following tools:\n"
                "  read_file(path) — read a source file\n"
                "  run_tests(path) — run pytest on a file, returns output\n"
                "  write_fix(path, content) — overwrite a file with fixed code\n\n"
                "When you need to use a tool, respond EXACTLY:\n"
                "Action: <tool_name>(<args>)\n\n"
                "When you are done, respond:\n"
                "Final Answer: <summary>"
            ),
            max_turns=10,
            fixture_files=["buggy_code.py", "test_buggy.py"],
        ),
        "research_synth": TaskDef(
            name="research_synth",
            description=(
                "Agent reads research documents and synthesises a "
                "recommendation. Context grows steadily with each doc read."
            ),
            system_prompt=(
                "You are a research analyst. You have access to these tools:\n"
                "  read_doc(id) — read a numbered research document\n"
                "  search(query) — keyword search across all documents\n\n"
                "When you need to use a tool, respond EXACTLY:\n"
                "Action: <tool_name>(<args>)\n\n"
                "After reading all documents, synthesise your findings into "
                "a structured recommendation. When done, respond:\n"
                "Final Answer: <report>"
            ),
            max_turns=15,
            fixture_files=[
                "research_doc_1.txt",
                "research_doc_2.txt",
                "research_doc_3.txt",
            ],
        ),
    }


# ---------------------------------------------------------------------------
# Simulated tool environment
# ---------------------------------------------------------------------------


class ToolEnvironment:
    """Simulates tool-call results for benchmark tasks without real execution."""

    def __init__(self, task: TaskDef) -> None:
        self.task = task
        self._fixtures: dict[str, str] = {}
        for fname in task.fixture_files:
            self._fixtures[fname] = _load_fixture(fname)

    def handle_action(self, action_text: str) -> str:
        """Parse an Action: line and return a simulated observation."""
        action_text = action_text.strip()

        # read_file / read_doc
        if "read_file" in action_text or "read_doc" in action_text:
            return self._handle_read(action_text)

        # run_tests
        if "run_tests" in action_text:
            return self._handle_run_tests()

        # write_fix
        if "write_fix" in action_text:
            return self._handle_write_fix()

        # search
        if "search" in action_text:
            return self._handle_search(action_text)

        return "Error: unknown tool. Available: " + ", ".join(
            ["read_file", "read_doc", "run_tests", "write_fix", "search"]
        )

    def _handle_read(self, action_text: str) -> str:
        """Simulate reading a file or document."""
        # Try to match fixture by number or name
        for fname, content in self._fixtures.items():
            # Match by number: read_doc(1) → research_doc_1.txt
            for i, key in enumerate(self._fixtures, 1):
                if str(i) in action_text and key == fname:
                    return f"--- {fname} ---\n{content}"
            # Match by filename
            if fname in action_text:
                return f"--- {fname} ---\n{content}"
        # Default: return first fixture
        first_key = next(iter(self._fixtures))
        return f"--- {first_key} ---\n{self._fixtures[first_key]}"

    def _handle_run_tests(self) -> str:
        """Simulate pytest output exposing bugs."""
        return (
            "FAILED test_buggy.py::test_divide_by_zero - ZeroDivisionError: division by zero\n"
            "FAILED test_buggy.py::test_parse_age_non_numeric - ValueError: invalid literal\n"
            "FAILED test_buggy.py::test_greet_none - TypeError: can only concatenate str (not \"NoneType\") to str\n"  # noqa: E501
            "\n3 failed in 0.42s"
        )

    @staticmethod
    def _handle_write_fix() -> str:
        """Simulate successful fix application."""
        return "File written. Re-running tests...\n3 passed in 0.38s"

    def _handle_search(self, action_text: str) -> str:
        """Simulate keyword search across fixtures."""
        results: list[str] = []
        for fname, content in self._fixtures.items():
            lines = content.splitlines()
            matches = [
                ln
                for ln in lines
                if any(
                    tok.lower() in ln.lower()
                    for tok in action_text.split()
                    if len(tok) > 3
                )
            ]
            if matches:
                results.append(f"[{fname}]: " + " | ".join(matches[:3]))
        return "\n".join(results) if results else "No results found."


# ---------------------------------------------------------------------------
# Ollama client helpers
# ---------------------------------------------------------------------------


def _ollama_chat(
    model: str,
    messages: list[dict[str, str]],
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call Ollama /api/chat (non-streaming) and return the full response dict."""
    import urllib.request

    url = f"{OLLAMA_BASE}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options or {},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return max(1, len(text) // 4)


def _get_process_memory() -> int:
    """Return current RSS of this process in bytes (cross-platform)."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except ImportError:
        return 0


def _check_swap() -> bool:
    """Return True if the system is actively swapping (Linux/macOS heuristic)."""
    system = platform.system()
    if system == "Darwin":
        try:
            import subprocess

            out = subprocess.check_output(
                ["sysctl", "-n", "vm.swapusage"], text=True
            )
            # e.g. "total = 2048.00M  used = 123.45M  free = 1924.55M"
            for part in out.split():
                if part.replace(".", "").replace("M", "").isdigit():
                    used = float(part.replace("M", ""))
                    return used > 100  # >100 MB used swap
            return False
        except Exception:
            return False
    if system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("SwapFree:"):
                        free_kb = int(line.split()[1])
                    elif line.startswith("SwapTotal:"):
                        total_kb = int(line.split()[1])
            if total_kb == 0:
                return False
            return (total_kb - free_kb) / total_kb > 0.1
        except Exception:
            return False
    return False


# ---------------------------------------------------------------------------
# Build Ollama options for each condition
# ---------------------------------------------------------------------------


def build_raw_options() -> dict[str, Any]:
    """Raw Ollama: fixed small context, default settings."""
    return {
        "num_ctx": 4096,
        "temperature": 0.7,
    }


def build_autotune_options(
    messages: list[dict[str, str]],
    task: TaskDef,
    session_num_ctx: int,
) -> dict[str, Any]:
    """Autotune: session-locked KV, lower temperature, optimisations."""
    return {
        "num_ctx": session_num_ctx,
        "temperature": 0.3,
        "num_keep": _estimate_tokens(messages[0]["content"]) if messages else 256,
    }


def compute_session_num_ctx(
    initial_tokens: int,
    max_turns: int,
    max_new_tokens: int = 512,
) -> int:
    """Compute a single locked session context size.

    Sized to the task's full expected context ceiling, then snapped to
    the next standard KV bucket.  Held constant for the entire session
    to prevent model reloads.
    """
    needed = initial_tokens + max_turns * 300 + max_new_tokens + 512
    for bucket in KV_BUCKETS:
        if bucket >= needed:
            return bucket
    return KV_BUCKETS[-1]


# ---------------------------------------------------------------------------
# Run a single trial
# ---------------------------------------------------------------------------


def run_trial(
    task: TaskDef,
    model: str,
    condition: str,
    trial_idx: int,
    *,
    dry_run: bool = False,
) -> TrialResult:
    """Execute one trial of a task under a given condition.

    Parameters
    ----------
    task:       The task definition.
    model:      Ollama model tag (e.g. "llama3.2:3b").
    condition:  "raw" or "autotune".
    trial_idx:  0-based trial number.
    dry_run:    If True, skip Ollama calls and return a synthetic result.
    """
    result = TrialResult(
        task=task.name,
        model=model,
        condition=condition,
        trial=trial_idx,
    )

    env = ToolEnvironment(task)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": task.system_prompt},
        {"role": "user", "content": _initial_user_message(task)},
    ]

    # Pre-compute session_num_ctx for autotune (locked for entire session)
    initial_tokens = sum(_estimate_tokens(m["content"]) for m in messages)
    session_num_ctx = compute_session_num_ctx(
        initial_tokens, task.max_turns
    )

    trial_start = time.perf_counter()
    swap_events = 0

    for turn in range(task.max_turns):
        turn_start = time.perf_counter()

        if dry_run:
            tm = _synthetic_turn_metrics(turn, initial_tokens, condition)
            result.turns.append(tm)
            messages.append(
                {"role": "assistant", "content": "Final Answer: dry-run"}
            )
            result.task_success = True
            break

        # Build options for this condition
        if condition == "raw":
            options = build_raw_options()
        else:
            options = build_autotune_options(messages, task, session_num_ctx)

        # Call Ollama
        try:
            resp = _ollama_chat(model, messages, options)
        except Exception as exc:
            result.error = str(exc)
            break

        # Extract timing from Ollama response
        assistant_text = resp.get("message", {}).get("content", "")
        tm = TurnMetrics(turn=turn)

        tm.prompt_tokens = resp.get("prompt_eval_count", 0)
        tm.completion_tokens = resp.get("eval_count", 0)
        tm.prompt_eval_duration_ns = resp.get("prompt_eval_duration", 0)
        tm.eval_duration_ns = resp.get("eval_duration", 0)
        tm.load_duration_ns = resp.get("load_duration", 0)
        tm.total_duration_ns = resp.get("total_duration", 0)

        # TTFT ≈ prompt_eval_duration (time to process prompt before first token)
        tm.ttft_ms = tm.prompt_eval_duration_ns / 1e6

        # Detect model reload: load_duration spikes indicate full reload
        if tm.load_duration_ns > 100_000_000:  # >100ms → likely reload
            tm.model_reloaded = True

        tm.wall_time_ms = (time.perf_counter() - turn_start) * 1000

        # Check swap
        if _check_swap():
            swap_events += 1

        # Check for tool-call format errors
        if "Action:" in assistant_text:
            try:
                action_line = _extract_action(assistant_text)
                observation = env.handle_action(action_line)
                messages.append({"role": "assistant", "content": assistant_text})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            except Exception:
                tm.tool_call_error = True
                messages.append({"role": "assistant", "content": assistant_text})
                messages.append(
                    {
                        "role": "user",
                        "content": "Error: malformed tool call. Use the exact format: Action: tool_name(args)",
                    }
                )
        elif "Final Answer:" in assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
            result.task_success = True
            result.turns.append(tm)
            break
        else:
            # No action and no final answer — nudge the model
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append(
                {
                    "role": "user",
                    "content": "Please use a tool (Action: ...) or provide your Final Answer.",
                }
            )

        result.turns.append(tm)

    result.total_wall_s = time.perf_counter() - trial_start
    result.peak_ram_bytes = _get_process_memory()
    result.swap_events = swap_events
    result.context_tokens_final = sum(
        _estimate_tokens(m["content"]) for m in messages
    )

    return result


def _initial_user_message(task: TaskDef) -> str:
    """Generate the first user message for a task."""
    if task.name == "code_debugger":
        return (
            "I have a Python module `buggy_code.py` with some bugs. "
            "There are tests in `test_buggy.py` that fail. Please read "
            "the files, run the tests, identify the bugs, and write fixes."
        )
    if task.name == "research_synth":
        return (
            "I have 3 research documents about retrieval-augmented generation. "
            "Please read all three documents, cross-reference their findings, "
            "and produce a structured synthesis with a recommendation for "
            "which approach is best for a production QA system."
        )
    return f"Please complete the following task: {task.description}"


def _extract_action(text: str) -> str:
    """Extract the first Action: line from model output."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Action:"):
            return stripped[len("Action:") :].strip()
    raise ValueError("No Action: line found")


def _synthetic_turn_metrics(
    turn: int, initial_tokens: int, condition: str
) -> TurnMetrics:
    """Generate plausible synthetic metrics for dry-run mode."""
    base_ttft = 500 if condition == "raw" else 900
    return TurnMetrics(
        turn=turn,
        prompt_tokens=initial_tokens + turn * 200,
        completion_tokens=80,
        prompt_eval_duration_ns=int((base_ttft + turn * 50) * 1e6),
        eval_duration_ns=int(300e6),
        load_duration_ns=int(5e6),
        total_duration_ns=int((base_ttft + 350 + turn * 50) * 1e6),
        ttft_ms=base_ttft + turn * 50,
        wall_time_ms=base_ttft + 350 + turn * 50,
    )


# ---------------------------------------------------------------------------
# Analysis & summary
# ---------------------------------------------------------------------------


def summarise_trials(trials: list[TrialResult]) -> BenchmarkSummary:
    """Compute summary statistics across trials for a single condition."""
    if not trials:
        raise ValueError("No trials to summarise")

    first = trials[0]
    summary = BenchmarkSummary(
        task=first.task,
        model=first.model,
        condition=first.condition,
        n_trials=len(trials),
    )

    all_ttfts: list[float] = []
    all_reloads: list[int] = []
    all_wall: list[float] = []
    all_ram: list[float] = []
    all_ctx: list[int] = []
    total_tool_errors = 0
    successes = 0
    ttft_slopes: list[float] = []

    for trial in trials:
        turn_ttfts = [t.ttft_ms for t in trial.turns]
        all_ttfts.extend(turn_ttfts)

        reloads = sum(1 for t in trial.turns if t.model_reloaded)
        all_reloads.append(reloads)

        all_wall.append(trial.total_wall_s)
        all_ram.append(trial.peak_ram_bytes / (1024**3))
        all_ctx.append(trial.context_tokens_final)

        total_tool_errors += sum(1 for t in trial.turns if t.tool_call_error)
        summary.total_swap_events += trial.swap_events

        if trial.task_success:
            successes += 1

        # Compute TTFT slope (ms/turn) via simple linear regression
        if len(turn_ttfts) >= 2:
            n = len(turn_ttfts)
            x_mean = (n - 1) / 2
            y_mean = statistics.mean(turn_ttfts)
            num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(turn_ttfts))
            den = sum((i - x_mean) ** 2 for i in range(n))
            slope = num / den if den > 0 else 0
            ttft_slopes.append(slope)

    summary.avg_ttft_ms = statistics.mean(all_ttfts) if all_ttfts else 0
    summary.ttft_slope_ms_per_turn = (
        statistics.mean(ttft_slopes) if ttft_slopes else 0
    )
    summary.avg_model_reloads = statistics.mean(all_reloads) if all_reloads else 0
    summary.avg_wall_s = statistics.mean(all_wall) if all_wall else 0
    summary.avg_peak_ram_gb = statistics.mean(all_ram) if all_ram else 0
    summary.tool_call_errors = total_tool_errors
    summary.success_rate = successes / len(trials) if trials else 0
    summary.avg_context_tokens_final = (
        statistics.mean(all_ctx) if all_ctx else 0
    )

    return summary


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def format_report(summaries: list[BenchmarkSummary]) -> str:
    """Format benchmark summaries into a human-readable markdown report."""
    lines: list[str] = []
    lines.append("# Agent Benchmark Results\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")

    # Group by task
    tasks = sorted({s.task for s in summaries})
    for task_name in tasks:
        lines.append(f"\n## Task: `{task_name}`\n")
        task_summaries = [s for s in summaries if s.task == task_name]

        lines.append(
            "| Metric | "
            + " | ".join(f"{s.condition} ({s.model})" for s in task_summaries)
            + " |"
        )
        lines.append(
            "| ------ | " + " | ".join("---" for _ in task_summaries) + " |"
        )

        rows = [
            ("Trials", [str(s.n_trials) for s in task_summaries]),
            (
                "Avg TTFT (ms)",
                [f"{s.avg_ttft_ms:.1f}" for s in task_summaries],
            ),
            (
                "TTFT slope (ms/turn)",
                [f"{s.ttft_slope_ms_per_turn:+.1f}" for s in task_summaries],
            ),
            (
                "Avg model reloads",
                [f"{s.avg_model_reloads:.1f}" for s in task_summaries],
            ),
            (
                "Avg wall time (s)",
                [f"{s.avg_wall_s:.1f}" for s in task_summaries],
            ),
            (
                "Peak RAM (GB)",
                [f"{s.avg_peak_ram_gb:.2f}" for s in task_summaries],
            ),
            (
                "Swap events",
                [str(s.total_swap_events) for s in task_summaries],
            ),
            (
                "Tool call errors",
                [str(s.tool_call_errors) for s in task_summaries],
            ),
            (
                "Success rate",
                [f"{s.success_rate:.0%}" for s in task_summaries],
            ),
            (
                "Context tokens (final)",
                [f"{s.avg_context_tokens_final:.0f}" for s in task_summaries],
            ),
        ]

        for label, vals in rows:
            lines.append(f"| {label} | " + " | ".join(vals) + " |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Multi-turn agent benchmark for LLM inference."
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated Ollama model tags (default: %(default)s)",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task names (default: all)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Number of trials per condition (default: %(default)s)",
    )
    parser.add_argument(
        "--conditions",
        default="raw,autotune",
        help="Comma-separated conditions to benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: benchmark_results/agent_bench_results.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate tasks and produce synthetic results without Ollama",
    )
    args = parser.parse_args(argv)

    catalogue = build_task_catalogue()

    # Resolve tasks
    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",")]
        for name in task_names:
            if name not in catalogue:
                print(f"Error: unknown task '{name}'. Available: {list(catalogue)}", file=sys.stderr)
                sys.exit(1)
    else:
        task_names = list(catalogue)

    models = [m.strip() for m in args.models.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    print(f"Agent Benchmark — {len(task_names)} tasks × {len(models)} models × "
          f"{len(conditions)} conditions × {args.trials} trials")
    if args.dry_run:
        print("  (dry-run: no Ollama calls)\n")
    else:
        print(f"  Ollama: {OLLAMA_BASE}\n")

    all_trials: list[TrialResult] = []
    summaries: list[BenchmarkSummary] = []

    for model in models:
        for task_name in task_names:
            task = catalogue[task_name]
            for condition in conditions:
                condition_trials: list[TrialResult] = []
                for trial_idx in range(args.trials):
                    label = f"[{model}] {task_name}/{condition} trial {trial_idx + 1}/{args.trials}"
                    print(f"  Running {label} ...", end=" ", flush=True)

                    result = run_trial(
                        task, model, condition, trial_idx, dry_run=args.dry_run
                    )
                    condition_trials.append(result)
                    all_trials.append(result)

                    turns = len(result.turns)
                    status = "✓" if result.task_success else "✗"
                    print(f"{status}  {turns} turns, {result.total_wall_s:.1f}s")

                summary = summarise_trials(condition_trials)
                summaries.append(summary)

    # Write results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else RESULTS_DIR / "agent_bench_results.json"

    results_payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "models": models,
            "tasks": task_names,
            "conditions": conditions,
            "trials_per_condition": args.trials,
            "dry_run": args.dry_run,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "trials": [asdict(t) for t in all_trials],
        "summaries": [asdict(s) for s in summaries],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nResults written to {output_path}")

    # Print report
    report = format_report(summaries)
    print("\n" + report)

    # Also save the markdown report
    report_path = output_path.with_suffix(".md")
    report_path.write_text(report, encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
