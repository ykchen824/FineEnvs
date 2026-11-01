from pathlib import Path


def test_eval_answer_key_is_not_committed():
    """The 200 eval coordinates are an answer key; keep them out of the repo."""
    tasks = Path(__file__).resolve().parents[1] / "tasks"
    assert (tasks / "pano_v1.jsonl").is_file()
    assert not (tasks / "eval_pano_v3.jsonl").exists()
