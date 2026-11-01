# Task indexes

`pano_v1.jsonl` is the historical demo index. It predates the contamination rule and has a
task 603 m from an eval start, so it is for playing, not for reporting a benchmark number.

The frozen 200-task **eval index is deliberately not committed**: it contains the exact
coordinates every eval task is scored against, so a public copy is an answer key. Keep it in
a private bucket or a host path outside the repository and point the environment at it:

```bash
export GEOGUESSER_TASKS_EVAL=/data/geoguesser/eval_pano_v3.jsonl
# or the legacy single-index form:
export GEOGUESSER_INDEX=/data/geoguesser/eval_pano_v3.jsonl
```

The server refuses to start with no index rather than silently serving only the demo. The
training index lives in the public bucket; only the eval answer key is private. Rebuild the
eval split with `dataset/split_tasks.py` and write the output outside this repository (or
copy it to the private bucket) — do not add it here.
