# Run provenance

A small, standard-library record of an evaluation run, written next to the
results it describes. It answers the questions a published number needs:
which code, which model, which dataset and grader revision, which sandbox,
what the model actually ran, and whether a reward-critical file changed while
the run was in progress.

This is the execution-trace half of the reward-hacking audit in issue #1. The
change gate lives in `tools/reward_audit/`; this directory records runs.

## Format

One JSONL file per run. Three record types, distinguished by `type`:

| Record | Written | Contents |
|---|---|---|
| `run_start` | before the first attempt | `run_id`, `started_at`, `git_commit`, `config`, and the start-time SHA-256 of every asset |
| `task` | once per attempt | `task_id`/`task`/`harness`, `instruction_sha256`, the generated `code`, `stdout`/`stderr` tails, `prediction`, `reward`, `flags`, `source`, and any error |
| `run_end` | after the last attempt | `finished_at`, the end-time asset hashes, `assets_changed`, and `valid` |

`valid` is false when any asset changed between `run_start` and `run_end`. A
code revision change, a rewritten grader or a replaced reference file makes the
run's numbers suspect, and the trace says so instead of leaving it to a later
reader to guess.

`run_id` distinguishes a fresh run from a resumed one; the trace appends, so a
resumed evaluation has a second header and footer in the same file.

## What is recorded by each harness

- `04-smoldataenvs/scripts/eval_pass1.py` writes `eval_provenance.jsonl` next to
  `eval_results.json`. It records the model, split, dataset revision, task-set
  digest, sandbox image tag and manifest digest, grader file hash and revision,
  each generated program, its output, the prediction, reward and flags. The
  script fetches this module from the repository when it runs standalone on HF
  Jobs, the same way it already fetches `rollout.py`.
- `05-multi-harness-rl/eval/evaluate.py` writes `provenance.jsonl` in the output
  directory and extends `identity.json` with the git commit, the pinned dataset
  revisions, the grader SHA-256 and the task-set digest. Harbor does not expose
  the generated program to the evaluator, so `code`/`stdout`/`stderr` are null
  there; the record says so rather than inventing them.

## Use

```python
from provenance import RunTrace

trace = RunTrace(
    "run.jsonl",
    config={"model": "checkpoint-100"},
    assets=[{"path": "grader.py", "role": "grader", "revision": "..."}],
    repo_root=".",
)
trace.start()
trace.record(task_id="t1", code="print(1)", reward=1.0)
footer = trace.finish()
assert footer["valid"]
```

The module is also usable as a context manager; `finish()` runs on the way out,
including when the body raises.

Helpers: `sha256_file`, `sha256_text`, `sha256_rows`, `git_commit`,
`hub_revision`, `hub_file_revision`, `docker_image_digest`, and `redact`.

## Redaction

`redact()` replaces the exact values of environment variables whose names look
like credentials, then matches `hf_…` tokens, `sk-…` keys, bearer tokens and
`token=…`/`api_key=…` assignments. It is applied to every field of every record.
A test feeds a known token through a program's stdout and asserts it never
reaches the file.

Redaction is defence in depth, not a licence to print secrets. The harness
never writes the environment itself, and the trace contains no token fields.

## Limits

- The trace records what the harness can see. For a Daytona/Harbor episode the
  generated program is not available to the Python evaluator; those fields are
  null and the `source` field says `harbor`.
- The start/end asset check catches a file that changed while the run was in
  progress. It is not a filesystem audit of what the agent read inside its
  sandbox; that needs provider-level egress/read logging.
- The trace is local JSONL. Moving it somewhere durable (a Hub bucket, a run
  store) is the caller's job; the file is written before the summary so an
  interrupted run still has its header and per-task records.
- `docker_image_digest` resolves Docker Hub tags only and returns `None` on any
  registry/network failure; a missing digest is recorded as `None`, not guessed.
