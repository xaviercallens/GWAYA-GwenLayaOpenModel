"""Tests for GCP spot GPU benchmarking tooling.

Tests cover:
1. run_on_gcp.sh --dry-run behavior (prints commands, never calls gcloud)
2. compare_runs.py correctness and gate rules
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.compare_runs import compare_runs, extract_metrics, load_results  # noqa: E402


class TestDryRunBehavior:
    """Test deploy/run_on_gcp.sh --dry-run outputs commands without executing gcloud."""

    @pytest.fixture
    def gcloud_stub(self, tmp_path: Path) -> str:
        """Create a stub gcloud script that records calls."""
        stub = tmp_path / "gcloud"
        stub.write_text(
            "#!/bin/bash\n"
            f'echo "GCLOUD_CALLED: $*" >> {tmp_path}/gcloud_calls.log\n'
            "exit 0\n"
        )
        stub.chmod(0o755)
        return str(stub)

    def test_dry_run_prints_commands_no_gcloud_calls(
        self, tmp_path: Path, gcloud_stub: str
    ) -> None:
        """Verify --dry-run prints gcloud commands without executing them."""
        # Create a fake gcloud in PATH that records calls
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_gcloud = bin_dir / "gcloud"
        fake_gcloud.write_text(
            "#!/bin/bash\n"
            f'echo "GCLOUD_CALLED: $*" >> {tmp_path}/gcloud_calls.log\n'
            "exit 0\n"
        )
        fake_gcloud.chmod(0o755)

        script_path = ROOT / "deploy" / "run_on_gcp.sh"
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")

        # Run with --dry-run (should be default)
        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
        result = subprocess.run(
            [str(script_path), "--gpu", "l4", "--n", "5", "--dry-run"],
            capture_output=True,
            text=True,
            env=env,
        )

        # Should succeed
        assert result.returncode == 0

        # Should print gcloud commands
        assert "gcloud compute instances create" in result.stdout
        assert "zones" in result.stdout.lower()
        assert "worst-case cost" in result.stdout.lower()

        # Should mention dry-run mode
        assert "DRY-RUN" in result.stdout or "dry-run" in result.stdout.lower()

        # gcloud should NOT have been called
        gcloud_calls_log = tmp_path / "gcloud_calls.log"
        if gcloud_calls_log.exists():
            calls = gcloud_calls_log.read_text().strip()
            assert not calls or calls == "", "gcloud should not have been called in dry-run"

    def test_dry_run_is_default(self, tmp_path: Path) -> None:
        """Verify that --dry-run is the default behavior."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_gcloud = bin_dir / "gcloud"
        fake_gcloud.write_text("#!/bin/bash\nexit 0\n")
        fake_gcloud.chmod(0o755)

        script_path = ROOT / "deploy" / "run_on_gcp.sh"
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")

        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"

        # Run without --dry-run flag (should still be dry-run)
        result = subprocess.run(
            [str(script_path), "--gpu", "l4", "--n", "5"],
            capture_output=True,
            text=True,
            env=env,
        )

        # Should succeed (dry-run is default)
        assert result.returncode == 0
        assert "DRY-RUN" in result.stdout or "dry-run" in result.stdout.lower()

    def test_yes_spend_requires_env_var(self, tmp_path: Path) -> None:
        """Verify --yes-spend without GWAYA_CONFIRM_SPEND=1 fails."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_gcloud = bin_dir / "gcloud"
        fake_gcloud.write_text("#!/bin/bash\nexit 0\n")
        fake_gcloud.chmod(0o755)

        script_path = ROOT / "deploy" / "run_on_gcp.sh"
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")

        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
        # Remove GWAYA_CONFIRM_SPEND if it exists
        env.pop("GWAYA_CONFIRM_SPEND", None)

        result = subprocess.run(
            [str(script_path), "--gpu", "l4", "--n", "5", "--yes-spend"],
            capture_output=True,
            text=True,
            env=env,
        )

        # Should fail without GWAYA_CONFIRM_SPEND=1
        assert result.returncode != 0
        assert "GWAYA_CONFIRM_SPEND" in result.stderr or "GWAYA_CONFIRM_SPEND" in result.stdout

    def test_prints_cost_estimate(self) -> None:
        """Verify cost estimate is calculated and printed."""
        script_path = ROOT / "deploy" / "run_on_gcp.sh"
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")

        result = subprocess.run(
            [str(script_path), "--gpu", "l4", "--n", "5", "--max-hours", "1"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        # Check for cost-related output
        output = result.stdout.lower()
        assert "cost" in output or "$" in output
        assert "usd" in output or "$" in output

    def test_prints_zones_to_try(self) -> None:
        """Verify script prints zones to try."""
        script_path = ROOT / "deploy" / "run_on_gcp.sh"
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")

        result = subprocess.run(
            [str(script_path), "--gpu", "l4", "--n", "5"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        # Should mention zones
        assert "zone" in result.stdout.lower()


class TestCompareRuns:
    """Test scripts/compare_runs.py for correctness and gate rules."""

    @pytest.fixture
    def sample_results_baseline(self) -> dict[str, Any]:
        """Sample baseline results."""
        return {
            "summary": {"total_duration_sec": 120},
            "rows": [
                {
                    "task_id": "MBPP/1",
                    "solved": True,
                    "tok_per_sec": 50.0,
                    "peak_vram_gb": 8.0,
                },
                {
                    "task_id": "MBPP/2",
                    "solved": True,
                    "tok_per_sec": 48.0,
                    "peak_vram_gb": 8.1,
                },
                {
                    "task_id": "MBPP/3",
                    "solved": False,
                    "tok_per_sec": 52.0,
                    "peak_vram_gb": 7.9,
                },
            ],
        }

    @pytest.fixture
    def sample_results_compare(self) -> dict[str, Any]:
        """Sample comparison results (improved)."""
        return {
            "summary": {"total_duration_sec": 100},
            "rows": [
                {
                    "task_id": "MBPP/1",
                    "solved": True,
                    "tok_per_sec": 60.0,
                    "peak_vram_gb": 9.0,
                },
                {
                    "task_id": "MBPP/2",
                    "solved": True,
                    "tok_per_sec": 58.0,
                    "peak_vram_gb": 8.9,
                },
                {
                    "task_id": "MBPP/3",
                    "solved": True,  # improved
                    "tok_per_sec": 62.0,
                    "peak_vram_gb": 9.2,
                },
            ],
        }

    def test_extract_metrics(self, sample_results_baseline: dict[str, Any]) -> None:
        """Test metric extraction."""
        metrics = extract_metrics(sample_results_baseline)

        # Pass rate: 2/3 solved = 66.67%
        assert 66 < metrics["pass_rate"] < 67
        assert metrics["total_problems"] == 3

        # Tokens/sec average
        assert metrics["avg_tok_per_sec"] == pytest.approx(50.0, rel=0.1)

        # VRAM
        assert metrics["avg_vram_gb"] == pytest.approx(8.0, rel=0.1)
        assert metrics["max_vram_gb"] == pytest.approx(8.1, rel=0.01)

    def test_compare_deltas(
        self,
        sample_results_baseline: dict[str, Any],
        sample_results_compare: dict[str, Any],
    ) -> None:
        """Test delta computation."""
        result = compare_runs(sample_results_baseline, sample_results_compare)

        # Pass rate improved: 66.67% -> 100%, delta = +33.33pp
        assert result["deltas"]["pass_rate_pct"] > 30

        # Tokens/sec improved
        assert result["deltas"]["tok_per_sec_delta"] > 0

        # VRAM increased (GPU might use more)
        assert result["deltas"]["vram_gb_delta"] > 0

    def test_gate_rule_pass(
        self,
        sample_results_baseline: dict[str, Any],
        sample_results_compare: dict[str, Any],
    ) -> None:
        """Test gate rule that should pass."""
        result = compare_runs(
            sample_results_baseline, sample_results_compare, gate_rule="pass_rate_pct >= -5"
        )

        # Pass rate improved significantly, gate should pass
        assert result["gate_status"] == "PASS"

    def test_gate_rule_fail(
        self,
        sample_results_baseline: dict[str, Any],
        sample_results_compare: dict[str, Any],
    ) -> None:
        """Test gate rule that should fail."""
        # Test with tok_per_sec_delta > 100 (tokens/sec improved only ~20%, not 100+)
        result = compare_runs(
            sample_results_baseline, sample_results_compare, gate_rule="tok_per_sec_delta > 100"
        )

        # Tokens/sec improved but not by 100+, should fail
        assert result["gate_status"] == "FAIL"

    def test_load_results_file(self) -> None:
        """Test loading results from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            results = {"rows": [{"solved": True}], "summary": {}}
            json.dump(results, f)
            f.flush()
            temp_path = f.name

        try:
            loaded = load_results(temp_path)
            assert loaded == results
        finally:
            os.unlink(temp_path)

    def test_load_results_missing_file(self) -> None:
        """Test loading from missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_results("/nonexistent/path/results.json")

    def test_compare_runs_with_missing_metrics(self) -> None:
        """Test comparison when some metrics are missing."""
        baseline = {"rows": [{"solved": True}]}
        compare = {"rows": [{"solved": True}]}

        result = compare_runs(baseline, compare)

        # Should still compute pass rate (both 100%)
        assert result["deltas"]["pass_rate_pct"] == 0

    def test_gate_rule_relative_percentage(
        self,
        sample_results_baseline: dict[str, Any],
        sample_results_compare: dict[str, Any],
    ) -> None:
        """Test relative percentage computation in gate rules."""
        result = compare_runs(
            sample_results_baseline, sample_results_compare, gate_rule="pass_rate_pct >= -5"
        )

        # Should have relative delta as well
        assert "pass_rate_pct_relative" in result["deltas"]
        # Pass rate improved, relative should be positive
        assert result["deltas"]["pass_rate_pct_relative"] > 0


class TestCompareRunsCLI:
    """Test compare_runs.py command-line interface."""

    def test_compare_runs_cli_basic(self) -> None:
        """Test CLI with basic usage."""
        script_path = ROOT / "scripts" / "compare_runs.py"
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")

        # Create temp result files
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            baseline = {
                "rows": [
                    {"task_id": "1", "solved": True, "tok_per_sec": 50, "peak_vram_gb": 8},
                    {"task_id": "2", "solved": False, "tok_per_sec": 52, "peak_vram_gb": 8},
                ]
            }
            compare = {
                "rows": [
                    {"task_id": "1", "solved": True, "tok_per_sec": 60, "peak_vram_gb": 9},
                    {"task_id": "2", "solved": True, "tok_per_sec": 62, "peak_vram_gb": 9},
                ]
            }

            baseline_file = tmp_path / "baseline.json"
            compare_file = tmp_path / "compare.json"
            baseline_file.write_text(json.dumps(baseline))
            compare_file.write_text(json.dumps(compare))

            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    str(baseline_file),
                    str(compare_file),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            # Should print metrics
            assert "Baseline" in result.stdout
            assert "Compare" in result.stdout
            assert "Deltas" in result.stdout

    def test_compare_runs_cli_with_gate(self) -> None:
        """Test CLI with --gate option."""
        script_path = ROOT / "scripts" / "compare_runs.py"
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            baseline = {"rows": [{"task_id": "1", "solved": True}]}
            compare = {"rows": [{"task_id": "1", "solved": True}]}

            baseline_file = tmp_path / "baseline.json"
            compare_file = tmp_path / "compare.json"
            baseline_file.write_text(json.dumps(baseline))
            compare_file.write_text(json.dumps(compare))

            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    str(baseline_file),
                    str(compare_file),
                    "--gate",
                    "pass_rate_pct >= -5",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "PASS" in result.stdout

    def test_compare_runs_cli_missing_file(self) -> None:
        """Test CLI with missing file."""
        script_path = ROOT / "scripts" / "compare_runs.py"
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "/nonexistent/file1.json",
                "/nonexistent/file2.json",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0


class TestGPUPricesJSON:
    """Test deploy/gpu_prices.json configuration."""

    def test_gpu_prices_file_exists(self) -> None:
        """Verify gpu_prices.json exists and is valid."""
        prices_file = ROOT / "deploy" / "gpu_prices.json"
        assert prices_file.exists(), f"gpu_prices.json not found at {prices_file}"

    def test_gpu_prices_structure(self) -> None:
        """Verify gpu_prices.json has required structure."""
        prices_file = ROOT / "deploy" / "gpu_prices.json"
        if not prices_file.exists():
            pytest.skip("gpu_prices.json not found")

        with open(prices_file) as f:
            cfg = json.load(f)

        # Should have gpus section
        assert "gpus" in cfg
        gpus = cfg["gpus"]

        # T4 and L4 should be defined
        assert "t4" in gpus
        assert "l4" in gpus

        # Each GPU should have required fields
        for gpu_name in ["t4", "l4"]:
            gpu_cfg = gpus[gpu_name]
            assert "name" in gpu_cfg
            assert "vram_gb" in gpu_cfg
            assert "machine_type" in gpu_cfg
            assert "price_per_hour_usd" in gpu_cfg
            assert "zones" in gpu_cfg
            assert "default_models" in gpu_cfg

        # T4 should have qwen2.5-coder:7b
        assert "qwen2.5-coder:7b" in gpus["t4"]["default_models"]

        # L4 should support both 7b and 14b
        l4_models = gpus["l4"]["default_models"]
        assert "qwen2.5-coder:7b" in l4_models or any(
            "qwen2.5-coder" in m for m in l4_models
        )
