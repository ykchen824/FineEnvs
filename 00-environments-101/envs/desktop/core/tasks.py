"""Shared Desktop tasks. Used by every framework variant under desktop_env/.

Each task ends with a status file or artefact the verifier can inspect, so the
environment does not have to trust the agent's own ``terminate(success)``. The
verify command prints exactly ``VERIFIED`` or ``NOT_VERIFIED``.
"""

from __future__ import annotations

from typing import Any, Callable


def verify_task(task: dict, run_command: Callable[[str], Any]) -> bool:
    """Run the task's status probe; only an explicit ``VERIFIED`` line passes."""
    command = task.get("verify_command")
    if not command:
        return False
    try:
        result = run_command(command)
    except Exception:  # noqa: BLE001 - a broken probe is not a pass
        return False
    if isinstance(result, tuple):
        result = result[0]
    text = getattr(result, "stdout", None) or str(result or "")
    return any(line.strip() == "VERIFIED" for line in text.splitlines())


TASKS = [
    {
        "task": (
            "In Firefox, navigate to https://example.com and save the page to "
            "/tmp/example-domain.html (Ctrl+S; choose 'Web Page, HTML Only' if "
            "asked). The saved file must contain 'Example Domain'."
        ),
        "app": "firefox",
        "resolution": [1024, 768],
        "expected_output": "Example Domain",
        "verify_command": (
            "grep -q 'Example Domain' /tmp/example-domain.html "
            "&& echo VERIFIED || echo NOT_VERIFIED"
        ),
    },
    {
        "task": (
            "In xfce4-terminal, run `echo hello-rl > /tmp/hello-rl.txt`, then "
            "`cat /tmp/hello-rl.txt` to confirm it prints hello-rl."
        ),
        "app": "terminal",
        "resolution": [1024, 768],
        "expected_output": "hello-rl",
        "verify_command": (
            "test \"$(cat /tmp/hello-rl.txt 2>/dev/null)\" = 'hello-rl' "
            "&& echo VERIFIED || echo NOT_VERIFIED"
        ),
    },
    {
        "task": (
            "In LibreOffice Calc: enter 23 in A1, 45 in B1, =A1+B1 in C1, then "
            "save the sheet as /tmp/calc-result.csv (File > Save As, CSV). "
            "C1 must be 68."
        ),
        "app": "libreoffice-calc",
        "resolution": [1024, 768],
        "expected_output": "68",
        "verify_command": (
            "python3 -c \"import csv; rows=list(csv.reader(open('/tmp/calc-result.csv'))); "
            "assert rows and [c.strip() for c in rows[0][:3]] == ['23', '45', '68']; "
            "print('VERIFIED')\" 2>/dev/null || echo NOT_VERIFIED"
        ),
    },
]
