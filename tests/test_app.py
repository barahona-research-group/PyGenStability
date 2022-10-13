"""Test cli."""
from pathlib import Path
from click.testing import CliRunner

from pygenstability import app

DATA = Path(__file__).absolute().parent / "data"


def test_cli(tmp_path):
    runner = CliRunner()
    res = runner.invoke(
        app.cli,
        ["run", str(DATA / "edges.csv"), "--result-file", str(tmp_path / "results.pkl")],
        catch_exceptions=False,
    )
    assert res.exit_code == 0

    res = runner.invoke(
        app.cli, ["plot_scan", str(tmp_path / "results.pkl")], catch_exceptions=False
    )
    assert res.exit_code == 0

    res = runner.invoke(
        app.cli,
        ["plot_communities", str(DATA / "edges.csv"), str(tmp_path / "results.pkl")],
        catch_exceptions=False,
    )
    assert res.exit_code == 0
