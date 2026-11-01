# /// script
# dependencies = [
#   "transformers>=4.56.0",
#   "torch>=2.4.0",
#   "torchvision",
#   "accelerate",
#   "peft",   # so a LoRA adapter repo can be evaluated directly
#   "datasets>=3.0.0",
#   "huggingface_hub>=1.22.0",
#   "math-verify",
#   "trackio",
# ]
# ///
"""Score any model on the SmolDataEnvs held-out split. Same environment as training.

    uv run eval_pass1.py                       # 144 eval tasks, Qwen3.5-2B
    MODEL=you/smoldataenvs-grpo-2b uv run eval_pass1.py
    SPLIT=test NUM_TASKS=250 uv run eval_pass1.py      # the benchmark split
    uv run eval_pass1.py --dry-run             # no GPU: prompts + grading only

Run it once before training and once after; the difference is the result. Nothing
here knows about training, so a number from this script means the same thing for
a base model, a checkpoint, or something you downloaded.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROLLOUT_URL = os.environ.get(
    "ROLLOUT_URL",
    "https://raw.githubusercontent.com/adithya-s-k/FineEnvs/main/04-smoldataenvs/scripts/rollout.py",
)
if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rollout.py")):
    with open("rollout.py", "wb") as fh:
        fh.write(urllib.request.urlopen(ROLLOUT_URL).read())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import rollout as rollout_module  # noqa: E402
from rollout import SandboxRunner, build_prompt, rollout  # noqa: E402


def _load_provenance():
    """Import the shared run-provenance module, fetching it on HF Jobs.

    Jobs runs this file standalone, so the repository's tools/ tree is not
    present next to it. The rollout script is already fetched the same way.
    """
    local = Path(__file__).resolve().parents[2] / "tools" / "run_provenance"
    if (local / "provenance.py").is_file():
        sys.path.insert(0, str(local))
    else:
        target = Path("provenance.py")
        if not target.is_file():
            url = os.environ.get(
                "PROVENANCE_URL",
                "https://raw.githubusercontent.com/adithya-s-k/FineEnvs/main/tools/run_provenance/provenance.py",
            )
            urllib.request.urlretrieve(url, target)
        sys.path.insert(0, str(target.parent))
    import provenance

    return provenance


provenance = _load_provenance()

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
DATASET = os.environ.get("DATASET", "FineEnvs/SmolDataEnvs")
SPLIT = os.environ.get("SPLIT", "eval")
NUM_TASKS = int(os.environ.get("NUM_TASKS", 0))  # 0 = the whole split
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 1024))
BATCH = int(os.environ.get("BATCH", 8))
OUT = os.environ.get("OUT", "eval_results.json")


def generate(rows) -> list[str]:
    """Greedy, one sample per task: that is what pass@1 means."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="auto")
    model.eval()

    out: list[str] = []
    for i in range(0, len(rows), BATCH):
        chunk = rows[i : i + BATCH]
        texts = [
            tok.apply_chat_template(
                build_prompt(r),
                tokenize=False,
                add_generation_prompt=True,
                # non-thinking, to match how the model is trained
                enable_thinking=False,
            )
            for r in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(ids[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"  generated {min(i + BATCH, len(rows))}/{len(rows)}", flush=True)
    return out


def main() -> None:
    from datasets import load_dataset

    rows = list(load_dataset(DATASET, split=SPLIT))
    if NUM_TASKS:
        rows = rows[:NUM_TASKS]
    dry = "--dry-run" in sys.argv
    print(f"{MODEL} on {DATASET}:{SPLIT} — {len(rows)} tasks{' (dry run)' if dry else ''}")

    if dry:
        rows = rows[: int(os.environ.get("DRY_RUN_TASKS", 2))]
        completions = [f"```python\nprint({r['answer']!r})\n```" for r in rows]
    else:
        completions = generate(rows)

    dataset_revision = provenance.hub_revision(DATASET, "dataset")
    image = getattr(SandboxRunner, "IMAGE", os.environ.get("SANDBOX_IMAGE"))
    image_digest = provenance.docker_image_digest(image)
    try:
        grader_file = rollout_module.grader_path()
    except Exception:  # noqa: BLE001 - the grader is downloaded on first use
        grader_file = None
    assets = [
        {"path": str(Path(__file__).resolve()), "role": "eval_harness"},
        {"path": str(Path(rollout_module.__file__).resolve()), "role": "rollout_harness"},
    ]
    if grader_file:
        assets.append(
            {
                "path": grader_file,
                "role": "grader",
                "revision": provenance.hub_file_revision(grader_file),
            }
        )
    trace_path = Path(
        os.environ.get("PROVENANCE_PATH", str(Path(OUT).with_name("eval_provenance.jsonl")))
    )
    trace = provenance.RunTrace(
        trace_path,
        config={
            "model": MODEL,
            "split": SPLIT,
            "dataset": DATASET,
            "dataset_revision": dataset_revision,
            "task_set_sha256": provenance.sha256_rows(
                rows,
                ("task_id", "question", "answer", "reward_mode", "atol", "rtol", "bucket_prefix", "files"),
            ),
            "num_tasks": len(rows),
            "dry_run": dry,
            "sandbox_image": image,
            "sandbox_image_digest": image_digest,
            "grader_sha256": provenance.sha256_file(grader_file) if grader_file else None,
        },
        assets=assets,
        repo_root=Path(__file__).resolve().parents[2],
    )
    trace.start()

    by_tier: dict[str, list[float]] = defaultdict(list)
    records = []
    try:
        runner = SandboxRunner()
        try:
            for row, completion in zip(rows, completions):
                res = rollout(runner, row, completion)
                if res["reward"] is not None:  # None: the sandbox died; not a wrong answer
                    by_tier[row["difficulty_tier"]].append(res["reward"])
                flags = list(res.get("flags") or [])
                if res["reward"] is None and "ungraded" not in flags:
                    flags.append("ungraded")
                instruction_sha256 = provenance.sha256_text(row["question"])
                trace.record(
                    task_id=row["task_id"],
                    tier=row["difficulty_tier"],
                    instruction_sha256=instruction_sha256,
                    prediction=res.get("prediction", ""),
                    reward=res["reward"],
                    code=res.get("code", ""),
                    stdout=(res.get("stdout") or "")[-4000:],
                    stderr=(res.get("stderr") or "")[-4000:],
                    flags=flags,
                    source="smoldataenvs",
                )
                records.append(
                    {
                        "task_id": row["task_id"],
                        "tier": row["difficulty_tier"],
                        "gold": row["answer"],
                        "prediction": res["prediction"],
                        "reward": res["reward"],
                        "instruction_sha256": instruction_sha256,
                        "flags": flags,
                    }
                )
                print(
                    f"  {row['task_id'][:24]:26} {row['difficulty_tier']:7} "
                    f"gold={row['answer'][:18]!r:22} got={res['prediction'][:18]!r:22} {res['reward']}"
                )
        finally:
            runner.close()
    finally:
        footer = trace.finish()
    if not footer["valid"]:
        print("WARNING: a reward asset changed during the run; provenance marks it invalid", file=sys.stderr)

    scored = [r["reward"] for r in records if r["reward"] is not None]
    overall = sum(scored) / len(scored) if scored else None  # nothing graded: no score, not 0
    summary = {
        "model": MODEL,
        "split": SPLIT,
        "n": len(scored),
        "ungraded": len(records) - len(scored),
        "pass@1": round(overall, 4) if overall is not None else None,
        "by_tier": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_tier.items())},
    }
    print("\n" + json.dumps(summary, indent=2))
    with open(OUT, "w") as fh:
        json.dump(
            {
                "summary": summary,
                "provenance": {
                    "path": str(trace_path),
                    "run_id": trace.run_id,
                    "git_commit": trace.git_commit,
                    "valid": footer["valid"],
                    "assets_changed": footer["assets_changed"],
                },
                "records": records,
            },
            fh,
            indent=2,
        )
    print(f"wrote {OUT}")
    print(f"wrote {trace_path}")

    if os.environ.get("TRACKIO_SPACE"):
        import trackio

        trackio.init(project="smoldataenvs", name=f"eval-{MODEL.split('/')[-1]}-{SPLIT}",
                     space_id=os.environ["TRACKIO_SPACE"])
        metrics = {f"pass@1/{k}": v for k, v in summary["by_tier"].items()}
        if overall is not None:
            metrics["pass@1"] = overall
        trackio.log(metrics)
        trackio.finish()


if __name__ == "__main__":
    main()
