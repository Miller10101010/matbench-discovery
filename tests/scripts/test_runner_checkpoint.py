"""Local checkpoints reach both dependency-isolated inference runners."""

import shlex
import sys
from pathlib import Path
from typing import NoReturn

import pytest

from tests.utils import import_repo_script


@pytest.mark.parametrize("task", ["md", "diatomics"])
def test_print_command_retains_checkpoint(
    task: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checkpoint paths with spaces survive command rendering."""
    script = import_repo_script(f"checkpoint_{task}", f"models/run_{task}.py")
    checkpoint = "/tmp/local model.pt2"
    monkeypatch.setattr(
        sys, "argv", [task, "--model", "emt", "--print-cmd", "--checkpoint", checkpoint]
    )
    assert script.main() == 0
    command = shlex.split(capsys.readouterr().out)
    assert command[command.index("--checkpoint") + 1] == checkpoint


@pytest.mark.parametrize("task", ["md", "diatomics"])
def test_local_checkpoint_reaches_calculator(
    task: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The override is passed to the calculator before any inference starts."""
    script = import_repo_script(f"load_checkpoint_{task}", f"models/run_{task}.py")
    checkpoint = str(tmp_path / "local.pt2")
    captured = {}

    class CalculatorReachedError(Exception):
        """Stop before running the scientific benchmark."""

    def capture(*_args: object, **kwargs: object) -> NoReturn:
        """Record the local checkpoint and stop before evaluating the model."""
        captured.update(kwargs)
        raise CalculatorReachedError

    monkeypatch.setattr(script, "load_calculator", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            task,
            "--model",
            "emt",
            "--dry-run",
            "--checkpoint",
            checkpoint,
            "--out-dir",
            str(tmp_path / "result"),
        ],
    )
    with pytest.raises(CalculatorReachedError):
        script.main()
    assert captured["checkpoint"] == checkpoint
