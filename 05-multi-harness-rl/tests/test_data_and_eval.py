import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoldataenv_whitebox.tasks import stage_task

from eval import evaluate
from eval.evaluate import build_identity, summarize, write_json


def test_fixed_split_is_disjoint_and_keeps_the_training_mix():
    train = json.loads((ROOT / "data/train.json").read_text())
    test = json.loads((ROOT / "data/test.json").read_text())
    assert len(train) == 1000 and len(test) == 250
    assert sum(row["difficulty"] == "medium" for row in train) == 400
    assert sum(row["difficulty"] == "hard" for row in train) == 600
    for key in ("source", "notebook", "instruction_sha256"):
        assert not {t[key] for t in train} & {t[key] for t in test}


def test_ungraded_is_excluded_from_pass_at_one_but_visible_in_coverage():
    result = summarize(
        [
            {"correctness": 1, "tool_calls": 6},
            {"correctness": 0},
            {"correctness": None},
        ],
        4,
    )
    assert result["pass_at_1_observed"] == 0.5
    assert result["coverage"] == 0.5
    assert result["mean_tool_calls"] == 6
    assert result["tool_count_coverage"] == 1
    assert summarize([], 4)["pass_at_1_observed"] is None


def test_staging_does_not_upload_gold_or_leave_tokens_in_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-download-token")
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment/pull_bucket.py").write_text("print('download')")
    (tmp_path / "task.toml").write_text(
        '[environment.env]\nHF_TOKEN="${HF_TOKEN}"\n[verifier.env]\nEXPECTED_ANSWER="hidden-gold"\n'
    )
    writes, commands = [], []
    sandbox = SimpleNamespace(
        write_text=lambda *x: writes.append(x),
        exec=lambda *x, **kw: commands.append((x, kw)) or SimpleNamespace(exit_code=0),
    )
    stage_task(sandbox, tmp_path)
    assert "hidden-gold" not in str(writes) + str(commands)
    assert "test-download-token" not in str(writes)
    assert commands[0][1]["envs"] == {"HF_TOKEN": "test-download-token"}


def test_atomic_results_are_readable_after_replacement(tmp_path):
    target = tmp_path / "pairs" / "a.json"
    write_json(target, {"correctness": None})
    write_json(target, {"correctness": 1})
    assert json.loads(target.read_text())["correctness"] == 1
    assert not target.with_suffix(".tmp").exists()


def test_identity_records_stable_provenance_without_run_specific_fields(tmp_path):
    args = SimpleNamespace(checkpoint="checkpoint-100", model="test-model", data=str(tmp_path))
    tasks = [
        {"name": "a", "difficulty": "medium", "instruction": "first task"},
        {"name": "b", "difficulty": "hard", "instruction": "second task"},
    ]
    (tmp_path / "ready.json").write_text(
        json.dumps({"grader_sha256": "a" * 64}), encoding="utf-8"
    )
    (tmp_path / "test_manifest.json").write_text(
        json.dumps(
            {
                "source_dataset": "FineEnvs/SmolDataEnvs-harbor-test",
                "expected_hub_revision": "b" * 40,
            }
        ),
        encoding="utf-8",
    )

    identity = build_identity(args, tasks, ["whitebox"])

    assert identity["grader_sha256"] == "a" * 64
    assert identity["dataset_revisions"]["test"] == {
        "source_dataset": "FineEnvs/SmolDataEnvs-harbor-test",
        "source_revision": "b" * 40,
    }
    assert identity["task_set_sha256"] == evaluate.provenance.sha256_rows(
        tasks, ("name", "instruction", "difficulty")
    )
    assert identity["git_commit"]
    assert "run_id" not in identity
    assert "started_at" not in identity
