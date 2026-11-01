from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT_DIR))
import check  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]

BASE_MANIFEST = {
    "version": 1,
    "approval_label": "audit/approved",
    "protected": {
        "tests": ["**/tests/**", "**/test_*.py"],
        "reward": ["**/rubric.py", "tools/reward_audit/**"],
    },
}


class TempRepo:
    """A throwaway git repository with one commit per call to commit()."""

    def __init__(self, with_manifest: bool = True):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Reward Audit Test")
        self.git("config", "user.email", "audit@example.com")
        self.write("project/tests/test_env.py", "def test_placeholder():\n    assert True\n")
        self.write("project/rubric.py", "REWARD = 1.0\n")
        self.write("project/notes.md", "# notes\n")
        if with_manifest:
            self.write_manifest()

    def git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        )

    def write(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_manifest(self, data: dict | None = None) -> None:
        self.write(
            "tools/reward_audit/manifest.json",
            json.dumps(data or BASE_MANIFEST, indent=2),
        )

    def commit(self, message: str) -> None:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)

    def close(self) -> None:
        self._tmp.cleanup()


def run_check(args: list[str], root: Path) -> tuple[int, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = check.main(args, cwd=root)
    return code, stdout.getvalue() + stderr.getvalue()


class ProtectedAssetGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = TempRepo()
        self.repo.commit("base")

    def tearDown(self) -> None:
        self.repo.close()

    def test_protected_change_without_label_is_blocked(self) -> None:
        self.repo.write("project/tests/test_env.py", "def test_placeholder():\n    assert 1 == 2\n")
        self.repo.commit("weaken the test")
        code, output = run_check(["--base", "HEAD~1"], self.repo.root)
        self.assertEqual(code, check.EXIT_BLOCKED, output)
        self.assertIn("project/tests/test_env.py", output)
        self.assertIn("audit/approved", output)

    def test_protected_change_with_label_passes(self) -> None:
        self.repo.write("project/rubric.py", "REWARD = 2.0\n")
        self.repo.commit("change the reward")
        code, output = run_check(
            ["--base", "HEAD~1", "--labels", "audit/approved"], self.repo.root
        )
        self.assertEqual(code, check.EXIT_OK, output)
        self.assertIn("Approved", output)

    def test_label_may_have_neighbours_and_spaces(self) -> None:
        self.repo.write("project/rubric.py", "REWARD = 3.0\n")
        self.repo.commit("change the reward")
        code, output = run_check(
            ["--base", "HEAD~1", "--labels", "docs, audit/approved, ready"], self.repo.root
        )
        self.assertEqual(code, check.EXIT_OK, output)

    def test_ordinary_change_passes(self) -> None:
        self.repo.write("project/notes.md", "# a better note\n")
        self.repo.commit("fix a typo")
        code, output = run_check(["--base", "HEAD~1"], self.repo.root)
        self.assertEqual(code, check.EXIT_OK, output)
        self.assertIn("No protected reward assets changed", output)

    def test_deleting_a_protected_test_is_blocked(self) -> None:
        (self.repo.root / "project/tests/test_env.py").unlink()
        self.repo.commit("delete the test")
        code, output = run_check(["--base", "HEAD~1"], self.repo.root)
        self.assertEqual(code, check.EXIT_BLOCKED, output)
        self.assertIn("project/tests/test_env.py", output)

    def test_manifest_itself_is_protected(self) -> None:
        narrowed = {
            "version": 1,
            "approval_label": "audit/approved",
            "protected": {"tests": ["**/tests/**"]},
        }
        self.repo.write_manifest(narrowed)
        self.repo.write("project/rubric.py", "REWARD = 9.0\n")
        self.repo.commit("narrow the manifest and change the reward")
        code, output = run_check(["--base", "HEAD~1"], self.repo.root)
        self.assertEqual(code, check.EXIT_BLOCKED, output)
        # The base manifest still protects the rubric, even though HEAD removed it.
        self.assertIn("project/rubric.py", output)
        self.assertIn("tools/reward_audit/manifest.json", output)

    def test_bootstrap_manifest_is_accepted(self) -> None:
        repo = TempRepo(with_manifest=False)
        try:
            repo.commit("base without the gate")
            repo.write_manifest()
            repo.write("project/rubric.py", "REWARD = 4.0\n")
            repo.commit("introduce the gate and change a reward")
            code, output = run_check(["--base", "HEAD~1"], repo.root)
            self.assertEqual(code, check.EXIT_OK, output)
            self.assertIn("Bootstrap", output)
        finally:
            repo.close()

    def test_validate_accepts_a_complete_manifest(self) -> None:
        code, output = run_check(["--validate"], self.repo.root)
        self.assertEqual(code, check.EXIT_OK, output)
        self.assertIn("every protected pattern matches", output)

    def test_validate_rejects_a_pattern_with_no_file(self) -> None:
        data = {
            "version": 1,
            "approval_label": "audit/approved",
            "protected": {"tests": ["**/no-such-directory/**"]},
        }
        self.repo.write_manifest(data)
        self.repo.commit("add a dead pattern")
        code, output = run_check(["--validate"], self.repo.root)
        self.assertEqual(code, check.EXIT_CONFIG, output)
        self.assertIn("**/no-such-directory/**", output)

    def test_validate_rejects_malformed_manifest(self) -> None:
        for data in (
            {"version": 2, "protected": {"tests": ["**/tests/**"]}},
            {"version": 1, "protected": {}},
            {"version": 1, "protected": {"tests": []}},
            {"version": 1, "protected": {"tests": ["/absolute/**"]}},
            {"version": 1, "protected": {"tests": ["../outside/**"]}},
        ):
            with self.subTest(data=data):
                self.repo.write_manifest(data)
                self.repo.commit("malformed manifest")
                code, output = run_check(["--validate"], self.repo.root)
                self.assertEqual(code, check.EXIT_CONFIG, output)

    def test_missing_base_is_a_gate_error(self) -> None:
        code, output = run_check(["--base", "no-such-revision"], self.repo.root)
        self.assertEqual(code, check.EXIT_CONFIG, output)
        self.assertIn("reward audit", output)


class RealManifestTests(unittest.TestCase):
    def test_known_reward_assets_are_protected(self) -> None:
        categories, label = check.load_manifest(REPO_ROOT / check.MANIFEST_REL)
        self.assertEqual(label, "audit/approved")
        expected = [
            "00-environments-101/envs/desktop/ors/server.py",
            "00-environments-101/envs/jupyter/core/tasks.py",
            "01-latex-ocr/envs/latex_ocr/server/rubric.py",
            "02-watercolour/envs/watercolour/core/gate.py",
            "03-geoguesser/env/tasks/eval_pano_v3.jsonl",
            "03-geoguesser/env/server/scoring.py",
            "04-smoldataenvs/scripts/rollout.py",
            "05-multi-harness-rl/eval/evaluate.py",
            "05-multi-harness-rl/data/test.json",
            "06-multilingual/ocr/data/eval-nayana-all-test.json",
            "06-multilingual/ocr/envs/nayana_ocr/server/rewards.py",
            "06-multilingual/ocr/envs/nayana_ocr/data/evalset.py",
            "07-simulation-environments/portsim-v1/envs/berth_planning/core/tests/test_grader_robust.py",
            "07-simulation-environments/portsim-v1/envs/berth_planning/core/berth_core/model.py",
            "tools/reward_audit/manifest.json",
            ".github/workflows/reward-audit.yml",
        ]
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(
                    any(check.matches(path, pattern) for patterns in categories.values() for pattern in patterns),
                    f"{path} is not protected by the real manifest",
                )

    def test_ordinary_files_are_not_protected(self) -> None:
        categories, _ = check.load_manifest(REPO_ROOT / check.MANIFEST_REL)
        for path in ("README.md", "CONTRIBUTING.md", "03-geoguesser/README.md"):
            with self.subTest(path=path):
                self.assertFalse(
                    any(check.matches(path, pattern) for patterns in categories.values() for pattern in patterns),
                    f"{path} should not need the audit label",
                )

    def test_real_manifest_has_no_dead_patterns(self) -> None:
        categories, _ = check.load_manifest(REPO_ROOT / check.MANIFEST_REL)
        self.assertEqual(check.unmatched_patterns(REPO_ROOT, categories), [])


if __name__ == "__main__":
    unittest.main()
