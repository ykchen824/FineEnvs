"""Answer matching shared by every Jupyter framework port.

The reward used to be ``expected in output``. That pays for any message that
contains the expected value, so with one-character answers (the Python major
version ``"3"``, the unique-character count ``"4"``, the column name ``"y"``) a
sentence or an unrelated tool line scored 1.0. The answer is the last non-empty
line the agent produced; match that line exactly.
"""

from __future__ import annotations


def last_line(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def last_line_matches(expected: str, output: str) -> bool:
    """True only when the last non-empty line is the expected answer."""
    expected = (expected or "").strip()
    if not expected:
        return False
    return last_line(output) == expected
