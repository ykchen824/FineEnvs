"""The ORS Wordle server pins the task's word, ends on the right guess, and cannot be replayed."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("openreward")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import GuessInput, HistoryInput, WordleORS  # noqa: E402
from core.game import WordleGame  # noqa: E402


def _env(answer="crane"):
    env = WordleORS(task_spec={"answer": answer})
    env.setup()
    return env


def _expected(answer, words):
    game = WordleGame(answer=answer)
    for word in words:
        game.guess(word)
    return game.reward


def test_the_task_answer_is_the_hidden_word():
    out = _env("crane").guess(GuessInput(word="crane"))
    assert out.finished is True
    assert "Correct" in out.blocks[0].text


def test_a_guess_that_does_not_end_the_game_is_not_finished():
    out = _env().guess(GuessInput(word="slate"))
    assert out.finished is False
    assert out.reward == 0.0


def test_a_win_reports_the_game_reward():
    env = _env()
    env.guess(GuessInput(word="slate"))
    out = env.guess(GuessInput(word="crane"))
    assert out.finished is True
    assert out.reward == pytest.approx(_expected("crane", ["slate", "crane"]))


def test_six_misses_finish_with_partial_credit():
    words = ["adieu", "ghost", "lumpy", "brisk", "fjord", "waltz"]
    env = _env()
    for word in words:
        out = env.guess(GuessInput(word=word))
    assert out.finished is True
    assert out.reward == pytest.approx(_expected("crane", words))


def test_a_guess_after_a_loss_cannot_turn_it_into_a_win():
    words = ["adieu", "ghost", "lumpy", "brisk", "fjord", "waltz"]
    env = _env()
    for word in words:
        lost = env.guess(GuessInput(word=word))
    out = env.guess(GuessInput(word="crane"))
    assert out.finished is True
    assert lost.reward == pytest.approx(_expected("crane", words))
    assert out.reward is None
    assert "Correct" not in out.blocks[0].text


def test_a_guess_after_a_win_does_not_pay_the_reward_again():
    env = _env()
    won = env.guess(GuessInput(word="crane"))
    out = env.guess(GuessInput(word="slate"))
    assert won.finished is True
    assert won.reward == pytest.approx(_expected("crane", ["crane"]))
    assert out.finished is True
    assert out.reward is None


def test_tools_before_setup_report_no_game():
    env = WordleORS(task_spec={"answer": "crane"})
    assert env.guess(GuessInput(word="crane")).finished is False
    assert "No game" in env.get_history(HistoryInput()).blocks[0].text


def test_list_tasks_does_not_expose_the_answer():
    tasks = WordleORS.list_tasks("train")
    assert tasks and all("answer" not in task for task in tasks)
    assert all(task.get("task_id") for task in tasks)


def test_task_id_selects_the_pinned_answer_without_disclosing_it():
    from core.game import TASKS

    task = TASKS[0]
    env = WordleORS(task_spec={"task_id": task["task_id"]})
    env.setup()
    out = env.guess(GuessInput(word=task["answer"]))
    assert out.finished is True and "Correct" in out.blocks[0].text
