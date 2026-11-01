"""Adversarial reward tests for the audit in issue #1.

Each attack is paired with a legitimate-run control in the same class, so a
guard that rejects everything fails the control. Tests marked
``expectedFailure`` document a guard that is still open: they pass CI while the
hack works, and an unexpected success fails CI once the guard is fixed, which
is the reminder to remove the marker. The fix PRs are tracked in the audit
issue.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WATERCOLOUR = REPO_ROOT / "02-watercolour" / "envs" / "watercolour"
JUPYTER = REPO_ROOT / "00-environments-101" / "envs" / "jupyter"
DESKTOP = REPO_ROOT / "00-environments-101" / "envs" / "desktop"
OCR_DATA = REPO_ROOT / "06-multilingual" / "ocr" / "data"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _purge_core_modules(*roots: Path) -> None:
    """The 00 environments each have their own top-level ``core`` package."""
    for name in [name for name in sys.modules if name == "core" or name.startswith("core.")]:
        del sys.modules[name]
    for root in roots:
        while str(root) in sys.path:
            sys.path.remove(str(root))


def _install_sandbox_stubs() -> None:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault("dotenv", dotenv)
    for name in ("e2b_code_interpreter", "e2b_desktop"):
        module = types.ModuleType(name)
        module.Sandbox = object
        sys.modules.setdefault(name, module)


class WatercolourGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = _load_module(
            "adversarial_sketch_source", WATERCOLOUR / "core" / "sketch_source.py"
        )
        cls.fixtures = WATERCOLOUR / "core" / "fixtures"

    def report(self, name: str):
        return self.source.inspect_source((self.fixtures / name).read_text())

    def test_legitimate_painting_clears_the_gate(self) -> None:
        report = self.report("good_painting.js")
        self.assertEqual(report.external_access, [])
        self.assertEqual(report.bare_primitives, [])
        self.assertFalse(report.writes_text)
        self.assertTrue(report.painting_calls)

    def test_static_external_image_is_flagged(self) -> None:
        self.assertTrue(self.report("cheat_external_image.js").external_access)

    def test_static_text_label_is_flagged(self) -> None:
        self.assertTrue(self.report("cheat_text_label.js").writes_text)

    @unittest.expectedFailure
    def test_runtime_image_construction_is_flagged(self) -> None:
        """`new Image()` is not in the static blacklist; PR 4 must catch it."""
        self.assertTrue(self.report("cheat_runtime_image.js").external_access)

    @unittest.expectedFailure
    def test_drawing_context_text_is_flagged(self) -> None:
        """`drawingContext.fillText` bypasses the p5 `text(` check; PR 4."""
        self.assertTrue(self.report("cheat_fill_text.js").writes_text)


class JupyterRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        _install_sandbox_stubs()
        sys.path.insert(0, str(JUPYTER))
        self.env = _load_module(
            "adversarial_jupyter_skyrl", JUPYTER / "skyrl_gym" / "env.py"
        )

    def tearDown(self) -> None:
        _purge_core_modules(JUPYTER)

    def make(self, expected: str, output: str):
        env = self.env.JupyterSkyRLEnv(expected_output=expected)
        env.last_output = output
        return env

    def test_exact_answer_scores_one(self) -> None:
        self.assertEqual(self.make("y", "y")._compute_reward(), 1.0)

    @unittest.expectedFailure
    def test_incidental_character_is_not_an_answer(self) -> None:
        """``expected in last_output`` pays for any message containing 'y'."""
        self.assertEqual(self.make("y", "the system is ready")._compute_reward(), 0.0)

    @unittest.expectedFailure
    def test_incidental_number_is_not_an_answer(self) -> None:
        self.assertEqual(self.make("14", "I used 14 tools")._compute_reward(), 0.0)


class DesktopRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        _install_sandbox_stubs()
        sys.path.insert(0, str(DESKTOP))
        self.env = _load_module(
            "adversarial_desktop_skyrl", DESKTOP / "skyrl_gym" / "env.py"
        )

    def tearDown(self) -> None:
        _purge_core_modules(DESKTOP)

    def make(self, status: str):
        env = self.env.DesktopSkyRLEnv(expected_output="Example Domain")
        env._ctrl = types.SimpleNamespace(terminate_status=status)
        return env

    def test_failed_termination_scores_zero(self) -> None:
        self.assertEqual(self.make("failure")._compute_reward(True), 0.0)

    @unittest.expectedFailure
    def test_self_reported_success_without_task_evidence_scores_zero(self) -> None:
        """The reward is the agent's own terminate status; PR 4 must check state."""
        self.assertEqual(self.make("success")._compute_reward(True), 0.0)


class ReferenceHashTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_committed_mcq_answers_are_not_recoverable_from_their_hashes(self) -> None:
        """The eval file hashes the answer unsalted; PR 3 must remove the hashes."""
        data = json.loads((OCR_DATA / "eval-nayana-all-test.json").read_text())
        mcq = [task for task in data["tasks"] if task["family"] == "mcq_vqa"]
        recovered = []
        for task in mcq:
            expected = task["reference_sha256"]
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if hashlib.sha256(letter.encode()).hexdigest() == expected:
                    recovered.append((task["task_id"], letter))
                    break
        self.assertEqual(recovered, [])


if __name__ == "__main__":
    unittest.main()
