#!/usr/bin/env python3
"""Provenance records for evaluation runs.

A run writes one JSONL file: a ``run_start`` header, one ``task`` record per
graded attempt, and a ``run_end`` footer. The header pins the code revision,
the reward assets and the run configuration; the footer re-hashes the assets
and marks the run invalid if one of them changed while it was running. Task
records carry the model's program, output, prediction, reward and flags.

The module uses the standard library only. ``huggingface_hub`` is imported
lazily by the revision helpers, and every resolver that can fail returns
``None`` instead of raising so an offline run still writes a record.

See ``README.md`` in this directory for the schema and the limits.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1

# A value shorter than this is not treated as a secret to redact: replacing a
# short string would mangle ordinary prose without hiding anything.
MIN_SECRET_LENGTH = 8

_SECRET_WORDS = ("TOKEN", "SECRET", "API_KEY", "APIKEY", "PASSWORD", "CREDENTIAL")
_SECRET_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(
        r"(?i)((?:api[_-]?key|token|secret|password|credential)\s*[:=]\s*['\"]?)"
        r"[A-Za-z0-9._~+/=-]{12,}"
    ),
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | os.PathLike[str]) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def sha256_rows(rows: Iterable[Mapping[str, Any]], keys: Iterable[str]) -> str:
    """Stable digest of the selected task fields, independent of dict order."""
    payload = [{key: row.get(key) for key in sorted(keys)} for row in rows]
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str))


def _secret_values(environment: Mapping[str, str] | None = None) -> list[str]:
    environment = os.environ if environment is None else environment
    values = []
    for key, value in environment.items():
        if value and len(value) >= MIN_SECRET_LENGTH and any(
            word in key.upper() for word in _SECRET_WORDS
        ):
            values.append(value)
    return values


def redact(value: Any, environment: Mapping[str, str] | None = None) -> Any:
    """Recursively remove known tokens and secret-looking strings.

    Exact values of environment variables whose names look like credentials are
    replaced first. The regexes catch tokens that reached the trace through a
    program's stdout rather than the environment.
    """
    if isinstance(value, str):
        for secret in _secret_values(environment):
            value = value.replace(secret, "[redacted]")
        for pattern in _SECRET_PATTERNS:
            value = pattern.sub(
                lambda match: (
                    match.group(1) + "[redacted]"
                    if match.lastindex
                    else "[redacted]"
                ),
                value,
            )
        return value
    if isinstance(value, list):
        return [redact(item, environment) for item in value]
    if isinstance(value, tuple):
        return [redact(item, environment) for item in value]
    if isinstance(value, dict):
        return {str(key): redact(item, environment) for key, item in value.items()}
    return value


def git_commit(repo_root: str | os.PathLike[str] | None = None) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root or Path.cwd(),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode:
        return None
    commit = proc.stdout.strip()
    return commit or None


def hub_revision(
    repo_id: str, repo_type: str = "dataset", revision: str | None = None
) -> str | None:
    """Resolve a Hub repository's commit; ``None`` when the Hub is unavailable."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        if repo_type == "dataset":
            info = api.dataset_info(repo_id, revision=revision)
        elif repo_type == "model":
            info = api.model_info(repo_id, revision=revision)
        else:
            info = api.repo_info(repo_id, repo_type=repo_type, revision=revision)
        return getattr(info, "sha", None)
    except Exception:  # noqa: BLE001 - provenance must not break the run
        return None


def hub_file_revision(path: str | os.PathLike[str]) -> str | None:
    """Read the commit hash huggingface_hub stores next to a downloaded file."""
    metadata = Path(str(path) + ".metadata")
    try:
        data = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("commit_hash") or data.get("commit") or data.get("revision")
    return str(value) if value else None


def _parse_docker_image(image: str) -> tuple[str, str] | None:
    image = image.strip()
    if not image:
        return None
    if "@sha256:" in image:
        repo, digest = image.split("@", 1)
        repo = repo.removeprefix("docker.io/").removeprefix("index.docker.io/")
        if "/" not in repo:
            repo = "library/" + repo
        return repo, digest
    for prefix in ("docker.io/", "index.docker.io/"):
        if image.startswith(prefix):
            image = image[len(prefix) :]
    if "/" not in image:
        image = "library/" + image
    if ":" in image.rsplit("/", 1)[-1]:
        repo, tag = image.rsplit(":", 1)
    else:
        repo, tag = image, "latest"
    return repo, tag


def docker_image_digest(image: str | None, timeout: float = 8.0) -> str | None:
    """Resolve a Docker Hub image tag to its manifest digest, or ``None``."""
    if not image:
        return None
    parsed = _parse_docker_image(image)
    if parsed is None:
        return None
    repo, reference = parsed
    if reference.startswith("sha256:"):
        return reference
    try:
        token_url = (
            "https://auth.docker.io/token?service=registry.docker.io&scope="
            + urllib.parse.quote(f"repository:{repo}:pull", safe="")
        )
        with urllib.request.urlopen(token_url, timeout=timeout) as response:  # noqa: S310
            token = json.loads(response.read())["token"]
        request = urllib.request.Request(
            f"https://registry-1.docker.io/v2/{repo}/manifests/{reference}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": (
                    "application/vnd.docker.distribution.manifest.v2+json,"
                    "application/vnd.docker.distribution.manifest.list.v2+json,"
                    "application/vnd.oci.image.manifest.v1+json,"
                    "application/vnd.oci.image.index.v1+json"
                ),
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            digest = response.headers.get("Docker-Content-Digest")
            return digest or None
    except Exception:  # noqa: BLE001 - provenance must not break the run
        return None


class RunTrace:
    """Append-only JSONL provenance for one evaluation run."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        run_id: str | None = None,
        config: Mapping[str, Any] | None = None,
        assets: Iterable[Mapping[str, Any] | str] = (),
        repo_root: str | os.PathLike[str] | None = None,
    ):
        self.path = Path(path)
        self.run_id = run_id or uuid.uuid4().hex
        self.config = dict(config or {})
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.assets = [self._asset_spec(item) for item in assets]
        self._lock = threading.Lock()
        self._started = False
        self._finished = False
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.git_commit: str | None = None
        self.start_assets: list[dict[str, Any]] = []

    @staticmethod
    def _asset_spec(item: Mapping[str, Any] | str) -> dict[str, Any]:
        if isinstance(item, str):
            return {"path": item, "role": None, "revision": None}
        return {
            "path": str(item.get("path", "")),
            "role": item.get("role"),
            "revision": item.get("revision"),
        }

    def _asset_snapshot(self) -> list[dict[str, Any]]:
        records = []
        for asset in self.assets:
            path = asset["path"]
            if not path:
                continue
            size = None
            try:
                size = Path(path).stat().st_size
            except OSError:
                pass
            records.append(
                {
                    "path": path,
                    "role": asset.get("role"),
                    "revision": asset.get("revision"),
                    "sha256": sha256_file(path),
                    "bytes": size,
                    "exists": size is not None,
                }
            )
        return records

    def _write(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()

    def start(self) -> dict[str, Any]:
        if self._started:
            raise RuntimeError("RunTrace.start() called twice")
        self._started = True
        self.started_at = _now()
        self.git_commit = git_commit(self.repo_root)
        self.start_assets = self._asset_snapshot()
        header = {
            "type": "run_start",
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "git_commit": self.git_commit,
            "config": redact(self.config),
            "assets": self.start_assets,
        }
        self._write(header)
        return header

    def record(self, kind: str = "task", **fields: Any) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("RunTrace.record() before start()")
        record = {
            "type": kind,
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "at": _now(),
            **redact(fields),
        }
        self._write(record)
        return record

    def finish(self, **fields: Any) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("RunTrace.finish() before start()")
        if self._finished:
            raise RuntimeError("RunTrace.finish() called twice")
        self._finished = True
        self.finished_at = _now()
        end_assets = self._asset_snapshot()
        start_by_path = {item["path"]: item for item in self.start_assets}
        changed = [
            {
                "path": item["path"],
                "before": start_by_path.get(item["path"], {}).get("sha256"),
                "after": item["sha256"],
            }
            for item in end_assets
            if start_by_path.get(item["path"], {}).get("sha256") != item["sha256"]
            or start_by_path.get(item["path"], {}).get("exists")
            != item.get("exists")
        ]
        footer = {
            "type": "run_end",
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "finished_at": self.finished_at,
            "assets": end_assets,
            "assets_changed": changed,
            "valid": not changed,
            **redact(fields),
        }
        self._write(footer)
        return footer

    def __enter__(self) -> "RunTrace":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(error=repr(exc) if exc is not None else None)
