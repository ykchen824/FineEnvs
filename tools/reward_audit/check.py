#!/usr/bin/env python3
"""Fail a pull request that changes a reward-critical asset without review.

The manifest in ``tools/reward_audit/manifest.json`` lists the files that
decide an environment's reward: tests, graders, rubrics, evaluators, reference
data and the audit tooling itself.  This script compares the pull request with
its merge base and asks for the ``audit/approved`` label whenever one of those
files changed.

The base revision's manifest is used for the comparison, not the pull request's,
so a change cannot hide itself by editing the manifest in the same commit.  The
manifest's own change is protected too, and a pull request that introduces the
manifest for the first time is treated as a bootstrap (there is nothing to
compare against yet).

Usage, from the repository root::

    python3 tools/reward_audit/check.py --base origin/main
    python3 tools/reward_audit/check.py --base origin/main --labels audit/approved
    python3 tools/reward_audit/check.py --validate

The script only needs the standard library and git.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

MANIFEST_REL = "tools/reward_audit/manifest.json"
DEFAULT_LABEL = "audit/approved"

EXIT_OK = 0
EXIT_BLOCKED = 1
EXIT_CONFIG = 2


class GateError(RuntimeError):
    """The gate could not read a manifest or a git revision."""


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def repository_root(cwd: Path | None = None) -> Path:
    """Return the git toplevel for *cwd* (default: the current directory)."""
    proc = _git(Path(cwd or Path.cwd()), "rev-parse", "--show-toplevel")
    if proc.returncode:
        raise GateError(proc.stderr.strip() or "not inside a git repository")
    return Path(proc.stdout.strip())


def _validate_pattern(raw: object, category: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise GateError(f"category {category!r} has an empty or non-string pattern")
    pattern = raw.replace("\\", "/").strip()
    if pattern.startswith("/"):
        raise GateError(f"pattern {pattern!r} must be relative to the repository root")
    if ".." in Path(pattern).parts:
        raise GateError(f"pattern {pattern!r} must not contain '..'")
    return pattern


def parse_manifest(raw: object) -> tuple[dict[str, list[str]], str]:
    """Validate a decoded manifest and return (categories, approval label)."""
    if not isinstance(raw, dict):
        raise GateError("manifest must be a JSON object")
    if raw.get("version") != 1:
        raise GateError("manifest version must be 1")

    categories = raw.get("protected")
    if not isinstance(categories, dict) or not categories:
        raise GateError("manifest needs a non-empty 'protected' object")

    label = raw.get("approval_label", DEFAULT_LABEL)
    if not isinstance(label, str) or not label.strip():
        raise GateError("approval_label must be a non-empty string")

    parsed: dict[str, list[str]] = {}
    for category, patterns in categories.items():
        if not isinstance(category, str) or not category.strip():
            raise GateError("every protected category needs a name")
        if not isinstance(patterns, list) or not patterns:
            raise GateError(f"category {category!r} needs a non-empty pattern list")
        parsed[category] = [_validate_pattern(p, category) for p in patterns]
    return parsed, label.strip()


def load_manifest(path: Path) -> tuple[dict[str, list[str]], str]:
    """Load and validate a manifest from disk."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GateError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GateError(f"{path} is not valid JSON: {exc}") from exc
    return parse_manifest(raw)


def base_manifest(root: Path, base: str) -> tuple[dict[str, list[str]], str] | None:
    """Return the manifest at *base*, or None on a bootstrap change."""
    verify = _git(root, "rev-parse", "--verify", base)
    if verify.returncode:
        raise GateError(f"base revision {base!r} not found")
    proc = _git(root, "show", f"{base}:{MANIFEST_REL}")
    if proc.returncode:
        # The revision exists but has no manifest: this pull request
        # introduces the gate.  There is no baseline to compare against.
        return None
    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise GateError(f"{base}:{MANIFEST_REL} is not valid JSON: {exc}") from exc
    return parse_manifest(raw)


def changed_files(root: Path, base: str) -> list[str]:
    """Files changed between *base* and HEAD, with renames split into delete+add."""
    proc = _git(root, "diff", "--name-only", "--no-renames", f"{base}...HEAD")
    if proc.returncode:
        raise GateError(proc.stderr.strip() or f"cannot diff against {base}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def matches(path: str, pattern: str) -> bool:
    """Match a repository-relative path against one manifest pattern.

    ``fnmatch`` lets ``*`` cross directory separators, which is what the
    manifest wants for patterns such as ``**/tests/**``.  A leading ``**/`` is
    also tried without its prefix so a top-level ``tests/`` or ``test_*.py``
    matches too.
    """
    path = path.replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    pattern = pattern.replace("\\", "/")
    if fnmatch.fnmatchcase(path, pattern):
        return True
    if pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:]):
        return True
    return False


def protected_changes(
    files: Iterable[str], categories: dict[str, list[str]]
) -> list[tuple[str, str]]:
    """Return (category, path) for each changed file that the manifest protects."""
    found: list[tuple[str, str]] = []
    for path in files:
        for category, patterns in categories.items():
            if any(matches(path, pattern) for pattern in patterns):
                found.append((category, path))
                break
    return found


def parse_labels(value: str | None) -> set[str]:
    return {part.strip() for part in (value or "").split(",") if part.strip()}


def unmatched_patterns(
    root: Path, categories: dict[str, list[str]]
) -> list[tuple[str, str]]:
    """Patterns that match no tracked file, usually a typo or a deleted asset."""
    proc = _git(root, "ls-files")
    if proc.returncode:
        raise GateError(proc.stderr.strip() or "cannot list tracked files")
    tracked = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    missing: list[tuple[str, str]] = []
    for category, patterns in categories.items():
        for pattern in patterns:
            if not any(matches(path, pattern) for path in tracked):
                missing.append((category, pattern))
    return missing


def _print_manifest_summary(
    categories: dict[str, list[str]], label: str, root: Path
) -> None:
    total = sum(len(patterns) for patterns in categories.values())
    print(f"Manifest: {len(categories)} categories, {total} patterns, label {label!r}.")
    for category, patterns in categories.items():
        print(f"  {category}: {len(patterns)} patterns")
    missing = unmatched_patterns(root, categories)
    if missing:
        print("Patterns that match no tracked file (typo or deleted asset):")
        for category, pattern in missing:
            print(f"  [{category}] {pattern}")


def main(argv: list[str] | None = None, cwd: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="origin/main",
        help="base revision for the comparison (default: origin/main)",
    )
    parser.add_argument(
        "--labels",
        default=os.environ.get("PR_LABELS", ""),
        help="comma-separated pull request labels (default: $PR_LABELS)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="only validate the manifest and its patterns; no diff",
    )
    args = parser.parse_args(argv)

    try:
        root = repository_root(cwd)
        manifest_path = root / MANIFEST_REL
        categories, label = load_manifest(manifest_path)
    except GateError as exc:
        print(f"::error::reward audit: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    if args.validate:
        _print_manifest_summary(categories, label, root)
        missing = unmatched_patterns(root, categories)
        if missing:
            print(
                "::error::reward audit: the manifest has patterns that match no tracked file",
                file=sys.stderr,
            )
            return EXIT_CONFIG
        print("OK: every protected pattern matches at least one tracked file.")
        return EXIT_OK

    try:
        baseline = base_manifest(root, args.base)
    except GateError as exc:
        print(f"::error::reward audit: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    if baseline is None:
        print(
            f"Bootstrap: {args.base} has no {MANIFEST_REL}; validating the new "
            "manifest instead of diffing against it."
        )
        _print_manifest_summary(categories, label, root)
        return EXIT_OK

    base_categories, base_label = baseline

    try:
        files = changed_files(root, args.base)
    except GateError as exc:
        print(f"::error::reward audit: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    protected = protected_changes(files, base_categories)
    if not protected:
        print(f"No protected reward assets changed ({len(files)} files compared).")
        return EXIT_OK

    print("Protected reward assets changed:")
    for category, path in protected:
        print(f"  [{category}] {path}")

    labels = parse_labels(args.labels)
    if base_label in labels:
        print(f"\nApproved: the pull request carries the {base_label!r} label.")
        return EXIT_OK

    print(
        f"\n::error::This pull request changes {len(protected)} reward-critical "
        f"asset(s) but does not carry the {base_label!r} label. A maintainer "
        "applies that label after reviewing the change; the label is the audit "
        "record for this pull request. See tools/reward_audit/README.md."
    )
    return EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
