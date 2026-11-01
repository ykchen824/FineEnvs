from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import provenance  # noqa: E402


class RedactionTests(unittest.TestCase):
    def test_redacts_hf_token_in_text(self) -> None:
        token = "hf_" + "a" * 34
        text = f"the program printed HF_TOKEN={token} and then failed"
        redacted = provenance.redact(text)
        self.assertNotIn(token, redacted)
        self.assertIn("[redacted]", redacted)

    def test_redacts_environment_secret_values(self) -> None:
        env = {"HF_TOKEN": "forwarded-token-value", "PATH": "/usr/bin"}
        text = "stdout: forwarded-token-value"
        redacted = provenance.redact(text, environment=env)
        self.assertEqual(redacted, "stdout: [redacted]")

    def test_redacts_nested_records_and_leaves_ordinary_text(self) -> None:
        token = "hf_" + "b" * 34
        value = {"code": f"print('{token}')", "note": "ordinary output", "n": 1}
        redacted = provenance.redact(value)
        self.assertNotIn(token, json.dumps(redacted))
        self.assertEqual(redacted["note"], "ordinary output")
        self.assertEqual(redacted["n"], 1)


class TraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.asset = self.root / "grader.py"
        self.asset.write_text("REWARD = 1\n", encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _read(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_header_record_footer_and_asset_validation(self) -> None:
        trace_path = self.root / "run.jsonl"
        trace = provenance.RunTrace(
            trace_path,
            config={"model": "test-model", "dataset_revision": "abc"},
            assets=[{"path": str(self.asset), "role": "grader", "revision": "abc"}],
            repo_root=self.root,
        )
        header = trace.start()
        self.assertEqual(header["type"], "run_start")
        token = "hf_" + "c" * 34
        trace.record(task_id="t1", code=f"print('{token}')", reward=1.0)
        self.asset.write_text("REWARD = 0\n", encoding="utf-8")
        footer = trace.finish()
        self.assertFalse(footer["valid"])
        self.assertEqual(len(footer["assets_changed"]), 1)
        self.assertEqual(footer["assets_changed"][0]["path"], str(self.asset))
        records = self._read(trace_path)
        self.assertEqual([record["type"] for record in records], ["run_start", "task", "run_end"])
        self.assertNotIn(token, trace_path.read_text(encoding="utf-8"))
        self.assertEqual(records[1]["reward"], 1.0)

    def test_asset_that_does_not_change_is_valid(self) -> None:
        trace_path = self.root / "run.jsonl"
        trace = provenance.RunTrace(trace_path, assets=[str(self.asset)], repo_root=self.root)
        trace.start()
        trace.record(task_id="t1", reward=0.0)
        footer = trace.finish()
        self.assertTrue(footer["valid"])
        self.assertEqual(footer["assets_changed"], [])

    def test_missing_asset_is_recorded(self) -> None:
        trace_path = self.root / "run.jsonl"
        trace = provenance.RunTrace(
            trace_path, assets=[str(self.root / "not-there")], repo_root=self.root
        )
        trace.start()
        footer = trace.finish()
        self.assertTrue(footer["valid"])
        self.assertFalse(footer["assets"][0]["exists"])
        self.assertIsNone(footer["assets"][0]["sha256"])

    def test_record_before_start_is_an_error(self) -> None:
        trace = provenance.RunTrace(self.root / "run.jsonl", repo_root=self.root)
        with self.assertRaises(RuntimeError):
            trace.record(task_id="t1")

    def test_context_manager_finishes_even_on_error(self) -> None:
        trace_path = self.root / "run.jsonl"
        with self.assertRaises(RuntimeError):
            with provenance.RunTrace(trace_path, repo_root=self.root) as trace:
                trace.record(task_id="t1")
                raise RuntimeError("boom")
        footer = self._read(trace_path)[-1]
        self.assertEqual(footer["type"], "run_end")
        self.assertIn("boom", footer["error"])


class HelpersTests(unittest.TestCase):
    def test_sha256_rows_is_stable_across_field_order(self) -> None:
        first = provenance.sha256_rows([{"b": 2, "a": 1}], ["a", "b"])
        second = provenance.sha256_rows([{"a": 1, "b": 2}], ["b", "a"])
        self.assertEqual(first, second)
        third = provenance.sha256_rows([{"a": 1, "b": 3}], ["a", "b"])
        self.assertNotEqual(first, third)

    def test_hub_file_revision_reads_the_cache_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grader.py"
            path.write_text("x", encoding="utf-8")
            Path(str(path) + ".metadata").write_text(
                json.dumps({"commit_hash": "a" * 40}), encoding="utf-8"
            )
            self.assertEqual(provenance.hub_file_revision(path), "a" * 40)
            self.assertIsNone(provenance.hub_file_revision(Path(tmp) / "missing.py"))

    def test_docker_image_parsing(self) -> None:
        self.assertEqual(
            provenance._parse_docker_image("docker.io/owner/repo:tag"),
            ("owner/repo", "tag"),
        )
        self.assertEqual(
            provenance._parse_docker_image("ubuntu"), ("library/ubuntu", "latest")
        )
        self.assertEqual(
            provenance._parse_docker_image("owner/repo@sha256:" + "a" * 64),
            ("owner/repo", "sha256:" + "a" * 64),
        )
        self.assertIsNone(provenance.docker_image_digest(None))

    def test_git_commit_outside_a_repository_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(provenance.git_commit(tmp))


if __name__ == "__main__":
    unittest.main()
