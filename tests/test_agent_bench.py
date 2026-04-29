"""Tests for the agent benchmark harness.

These tests validate the benchmark infrastructure (task definitions,
tool environment, metrics collection, report generation) without
requiring an Ollama instance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Allow import from scripts/ and src/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import agent_bench  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------


class TestTaskCatalogue:
    def test_catalogue_has_expected_tasks(self):
        cat = agent_bench.build_task_catalogue()
        assert "code_debugger" in cat
        assert "research_synth" in cat

    def test_task_fields(self):
        cat = agent_bench.build_task_catalogue()
        for name, task in cat.items():
            assert task.name == name
            assert task.max_turns > 0
            assert len(task.system_prompt) > 0
            assert len(task.fixture_files) > 0


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


class TestFixtures:
    def test_load_fixture_exists(self):
        content = agent_bench._load_fixture("buggy_code.py")
        assert "def divide" in content

    def test_load_fixture_missing(self):
        with pytest.raises(FileNotFoundError):
            agent_bench._load_fixture("nonexistent_file.txt")

    def test_all_task_fixtures_exist(self):
        cat = agent_bench.build_task_catalogue()
        for task in cat.values():
            for fname in task.fixture_files:
                content = agent_bench._load_fixture(fname)
                assert len(content) > 0


# ---------------------------------------------------------------------------
# Tool environment
# ---------------------------------------------------------------------------


class TestToolEnvironment:
    @pytest.fixture()
    def debugger_env(self):
        cat = agent_bench.build_task_catalogue()
        return agent_bench.ToolEnvironment(cat["code_debugger"])

    @pytest.fixture()
    def research_env(self):
        cat = agent_bench.build_task_catalogue()
        return agent_bench.ToolEnvironment(cat["research_synth"])

    def test_read_file(self, debugger_env):
        result = debugger_env.handle_action("read_file(buggy_code.py)")
        assert "def divide" in result

    def test_run_tests(self, debugger_env):
        result = debugger_env.handle_action("run_tests(test_buggy.py)")
        assert "FAILED" in result
        assert "3 failed" in result

    def test_write_fix(self, debugger_env):
        result = debugger_env.handle_action("write_fix(buggy_code.py, ...)")
        assert "passed" in result

    def test_search(self, research_env):
        result = research_env.handle_action("search(retrieval augmented)")
        assert len(result) > 0

    def test_unknown_tool(self, debugger_env):
        result = debugger_env.handle_action("unknown_tool()")
        assert "Error" in result


# ---------------------------------------------------------------------------
# Session context sizing
# ---------------------------------------------------------------------------


class TestSessionSizing:
    def test_compute_session_num_ctx_small(self):
        ctx = agent_bench.compute_session_num_ctx(
            initial_tokens=200, max_turns=5, max_new_tokens=512
        )
        assert ctx in agent_bench.KV_BUCKETS
        assert ctx >= 200 + 5 * 300 + 512 + 512

    def test_compute_session_num_ctx_large(self):
        ctx = agent_bench.compute_session_num_ctx(
            initial_tokens=5000, max_turns=15, max_new_tokens=1024
        )
        assert ctx in agent_bench.KV_BUCKETS
        assert ctx >= 5000 + 15 * 300 + 1024 + 512

    def test_buckets_are_powers_of_two(self):
        for b in agent_bench.KV_BUCKETS:
            assert b & (b - 1) == 0, f"{b} is not a power of 2"

    def test_compute_returns_last_bucket_for_huge(self):
        ctx = agent_bench.compute_session_num_ctx(
            initial_tokens=30000, max_turns=100
        )
        assert ctx == agent_bench.KV_BUCKETS[-1]


# ---------------------------------------------------------------------------
# Options builders
# ---------------------------------------------------------------------------


class TestOptionsBuilders:
    def test_raw_options(self):
        opts = agent_bench.build_raw_options()
        assert opts["num_ctx"] == 4096
        assert opts["temperature"] == 0.7

    def test_autotune_options_locked_ctx(self):
        messages = [{"role": "system", "content": "You are a helper."}]
        cat = agent_bench.build_task_catalogue()
        task = cat["code_debugger"]
        session_ctx = agent_bench.compute_session_num_ctx(100, task.max_turns)
        opts = agent_bench.build_autotune_options(messages, task, session_ctx)
        assert opts["num_ctx"] == session_ctx
        assert opts["temperature"] == 0.3
        assert "num_keep" in opts


# ---------------------------------------------------------------------------
# Dry-run trial
# ---------------------------------------------------------------------------


class TestDryRunTrial:
    def test_dry_run_completes(self):
        cat = agent_bench.build_task_catalogue()
        result = agent_bench.run_trial(
            cat["code_debugger"], "test-model", "raw", 0, dry_run=True
        )
        assert result.task_success is True
        assert len(result.turns) > 0
        assert result.error is None

    def test_dry_run_autotune(self):
        cat = agent_bench.build_task_catalogue()
        result = agent_bench.run_trial(
            cat["research_synth"], "test-model", "autotune", 0, dry_run=True
        )
        assert result.task_success is True


# ---------------------------------------------------------------------------
# Summary & reporting
# ---------------------------------------------------------------------------


class TestSummary:
    def _make_trial(self, condition="raw", success=True) -> agent_bench.TrialResult:
        cat = agent_bench.build_task_catalogue()
        result = agent_bench.run_trial(
            cat["code_debugger"], "test-model", condition, 0, dry_run=True
        )
        result.task_success = success
        return result

    def test_summarise_basic(self):
        trials = [self._make_trial() for _ in range(3)]
        summary = agent_bench.summarise_trials(trials)
        assert summary.n_trials == 3
        assert summary.success_rate == 1.0
        assert summary.avg_ttft_ms > 0

    def test_summarise_empty_raises(self):
        with pytest.raises(ValueError):
            agent_bench.summarise_trials([])

    def test_format_report(self):
        trials = [self._make_trial("raw"), self._make_trial("autotune")]
        summaries = [agent_bench.summarise_trials([t]) for t in trials]
        report = agent_bench.format_report(summaries)
        assert "# Agent Benchmark Results" in report
        assert "code_debugger" in report
        assert "Avg TTFT" in report


# ---------------------------------------------------------------------------
# CLI dry-run
# ---------------------------------------------------------------------------


class TestCLI:
    def test_main_dry_run(self, tmp_path):
        output_file = tmp_path / "results.json"
        agent_bench.main([
            "--tasks", "code_debugger",
            "--trials", "1",
            "--conditions", "raw",
            "--dry-run",
            "--output", str(output_file),
        ])
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "meta" in data
        assert "trials" in data
        assert "summaries" in data
        assert data["meta"]["dry_run"] is True

    def test_main_unknown_task(self):
        with pytest.raises(SystemExit):
            agent_bench.main(["--tasks", "nonexistent_task", "--dry-run"])


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    def test_estimate_tokens(self):
        assert agent_bench._estimate_tokens("hello world") >= 1
        assert agent_bench._estimate_tokens("a" * 400) == 100

    def test_estimate_empty(self):
        assert agent_bench._estimate_tokens("") == 1
