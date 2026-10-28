import importlib

import pytest
from typer.testing import CliRunner

import raterbench
from raterbench.cli import app

SUBPACKAGES = [
    "data",
    "llm",
    "prompting",
    "baselines",
    "metrics",
    "perturb",
    "analysis",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name: str) -> None:
    importlib.import_module(f"raterbench.{name}")


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert raterbench.__version__ in result.output


def test_cli_run_stub_exits_nonzero() -> None:
    result = CliRunner().invoke(app, ["run", "--config", "configs/main.yaml"])
    assert result.exit_code == 1
