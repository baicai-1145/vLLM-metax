import json
import subprocess
import sys
from pathlib import Path

import torch

from tools.debug.replay_deepseek_v4_plan08_probes import run_manifest


def _write_layer_stage(
    path: Path,
    *,
    positions: list[int],
    hidden_states: torch.Tensor,
    call: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "rank": 0,
            "layer_idx": 0,
            "call": call,
            "stage": "after_ffn",
            "positions": torch.tensor(positions, dtype=torch.int64),
            "input_ids": torch.tensor([100 + position for position in positions]),
            "tensors": {"hidden_states": hidden_states},
        },
        path,
    )


def _write_q_stage(
    path: Path,
    *,
    positions: list[int],
    raw_q: torch.Tensor,
    post_q: torch.Tensor,
    call: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "rank": 0,
            "layer_idx": 0,
            "call": call,
            "stage": "q_stages",
            "positions": torch.tensor(positions, dtype=torch.int64),
            "raw_q": raw_q,
            "post_q": post_q,
        },
        path,
    )


def test_run_manifest_reports_exact_and_divergent_probes(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _write_layer_stage(
        base / "rank0_layer0_call0_after_ffn.pt",
        positions=[767],
        hidden_states=torch.tensor([[1.0, 2.0]]),
    )
    _write_layer_stage(
        candidate / "rank0_layer0_call0_after_ffn.pt",
        positions=[766, 767],
        hidden_states=torch.tensor([[9.0, 9.0], [1.0, 2.0]]),
    )
    _write_q_stage(
        base / "rank0_layer0_call0_q_stages.pt",
        positions=[767],
        raw_q=torch.tensor([[[1.0, 2.0]]]),
        post_q=torch.tensor([[[3.0, 4.0]]]),
    )
    _write_q_stage(
        candidate / "rank0_layer0_call0_q_stages.pt",
        positions=[767],
        raw_q=torch.tensor([[[1.0, 2.5]]]),
        post_q=torch.tensor([[[3.0, 4.0]]]),
    )

    result = run_manifest(
        {
            "schema_version": 1,
            "probes": [
                {
                    "name": "ffn",
                    "kind": "layer_stage",
                    "base": str(base),
                    "candidate": str(candidate),
                    "position": 767,
                    "layer": 0,
                    "stage": "after_ffn",
                },
                {
                    "name": "wq_b",
                    "kind": "q_stage",
                    "base": str(base),
                    "candidate": str(candidate),
                    "position": 767,
                    "layer": 0,
                },
            ],
        }
    )

    assert result["summary"] == {
        "total": 2,
        "exact": 1,
        "divergent": 1,
        "invalid": 0,
        "missing": 0,
        "runtime_error": 0,
        "overall_pass": False,
    }
    assert result["probes"][0]["name"] == "ffn"
    assert result["probes"][0]["status"] == "exact"
    assert result["probes"][1]["name"] == "wq_b"
    assert result["probes"][1]["status"] == "divergent"
    assert result["probes"][1]["first_different_tensor"] == "raw_q"


def test_run_manifest_marks_missing_optional_probe(tmp_path):
    result = run_manifest(
        {
            "schema_version": 1,
            "probes": [
                {
                    "name": "optional_mhc",
                    "kind": "mhc_raw",
                    "corpus": str(tmp_path / "missing"),
                    "required": False,
                }
            ],
        }
    )

    assert result["summary"]["overall_pass"]
    assert result["summary"]["missing"] == 1
    assert result["probes"][0]["status"] == "missing"


def test_run_manifest_marks_invalid_o_proj_schema(tmp_path):
    capture = tmp_path / "old_o_proj.pt"
    torch.save({"legacy": torch.tensor([1])}, capture)

    result = run_manifest(
        {
            "schema_version": 1,
            "probes": [
                {
                    "name": "o_proj",
                    "kind": "o_proj",
                    "capture": str(capture),
                }
            ],
        }
    )

    assert result["summary"] == {
        "total": 1,
        "exact": 0,
        "divergent": 0,
        "invalid": 1,
        "missing": 0,
        "runtime_error": 0,
        "overall_pass": False,
    }
    assert result["probes"][0]["status"] == "invalid"
    assert "missing keys" in result["probes"][0]["error"]


def test_cli_writes_json_and_fails_when_required_probe_diverges(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    output = tmp_path / "summary.json"
    manifest_path = tmp_path / "manifest.json"
    _write_layer_stage(
        base / "rank0_layer0_call0_after_ffn.pt",
        positions=[767],
        hidden_states=torch.tensor([[1.0, 2.0]]),
    )
    _write_layer_stage(
        candidate / "rank0_layer0_call0_after_ffn.pt",
        positions=[767],
        hidden_states=torch.tensor([[1.0, 2.5]]),
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "probes": [
                    {
                        "name": "ffn",
                        "kind": "layer_stage",
                        "base": str(base),
                        "candidate": str(candidate),
                        "position": 767,
                        "layer": 0,
                        "stage": "after_ffn",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    from tools.debug.replay_deepseek_v4_plan08_probes import main

    assert main(["--manifest", str(manifest_path), "--output", str(output)]) == 1
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["summary"]["divergent"] == 1


def test_script_entrypoint_runs_from_repo_root(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "summary.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "probes": []}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "tools/debug/replay_deepseek_v4_plan08_probes.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "overall_pass=True" in completed.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["summary"]["total"] == 0
