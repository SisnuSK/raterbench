"""Command-line entry point: `raterbench run|analyze --config ...`."""

from pathlib import Path

import typer

from raterbench import __version__

app = typer.Typer(help="RaterBench: benchmark LLMs as expert text raters.", no_args_is_help=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"raterbench {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """RaterBench CLI."""


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to a run config YAML."),
    estimate: bool = typer.Option(
        False, "--estimate", help="Only project the cost of uncached calls; make no API calls."
    ),
) -> None:
    """Run the benchmark described by CONFIG (resumes from the response cache)."""
    import asyncio
    import json

    from dotenv import load_dotenv

    from raterbench.runner import Runner, load_run_config

    load_dotenv()
    runner = Runner(load_run_config(config))
    if estimate:
        rows = runner.estimate()
        for r in rows:
            typer.echo(json.dumps(r))
        typer.echo(f"TOTAL upper bound: ${sum(r['upper_bound_usd'] for r in rows):.4f}")
        return
    run_dir = asyncio.run(runner.run())
    typer.echo((run_dir / "summary.md").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    cost = manifest["cost"]
    typer.echo(f"Spent ${cost['spent_usd']:.4f} of ${cost['budget_usd']:.2f} budget")
    for name, u in cost["by_model"].items():
        typer.echo(
            f"  {name}: {u['calls']} new calls, {u['cached_calls']} cache hits, "
            f"${u['cost_usd']:.4f}"
        )
    if cost["halted_by_budget"]:
        typer.echo("HALTED: budget cap reached; re-run with a higher cap to resume from cache.")
        raise typer.Exit(code=2)
    typer.echo(f"Outputs: {run_dir}")


@app.command()
def analyze(
    config: Path = typer.Option(Path("configs/analyze.yaml"), "--config", "-c"),
) -> None:
    """Produce every results table and figure from saved predictions."""
    from raterbench.analysis.analyze import Analysis, AnalyzeConfig
    from raterbench.config import load_yaml

    out = Analysis(load_yaml(config, AnalyzeConfig)).run()
    typer.echo(f"Tables and figures written to {out}")


@app.command()
def perturb(config: Path = typer.Option(Path("configs/perturb.yaml"), "--config", "-c")) -> None:
    """Generate meaning-preserving perturbations and label them with each model."""
    import asyncio

    from raterbench.config import load_yaml
    from raterbench.perturb.run import PerturbConfig, run_perturbation

    out = asyncio.run(run_perturbation(load_yaml(config, PerturbConfig)))
    typer.echo((out / "perturbation_manifest.json").read_text(encoding="utf-8")[:2000])


@app.command()
def baselines(
    config: Path = typer.Option(Path("configs/baselines.yaml"), "--config", "-c"),
    majority_only: bool = typer.Option(False, "--majority-only", help="Skip encoder training."),
) -> None:
    """Run the majority-class and fine-tuned encoder baselines."""
    from raterbench.baselines.run import BaselineConfig, run_baselines
    from raterbench.config import load_yaml

    cfg = load_yaml(config, BaselineConfig)
    if majority_only:
        cfg.run_encoder = False
    out = run_baselines(cfg)
    typer.echo((out / "metrics.md").read_text(encoding="utf-8"))


@app.command("report-assets")
def report_assets(
    analysis_dir: Path = typer.Option(Path("results/analysis")),
    run_dir: Path = typer.Option(Path("results/runs/full_local")),
    baselines_dir: Path = typer.Option(Path("results/baselines")),
    out_dir: Path = typer.Option(Path("report/generated")),
) -> None:
    """Write the report's LaTeX tables and number macros, and the README results table."""
    import json

    from raterbench.analysis.report_assets import write_assets

    info = write_assets(analysis_dir, run_dir, baselines_dir, out_dir)
    typer.echo(json.dumps(info, indent=2))


errors_app = typer.Typer(help="Manual error coding for the report.", no_args_is_help=True)
app.add_typer(errors_app, name="errors")


@errors_app.command("export")
def errors_export(
    run_dir: Path = typer.Option(Path("results/runs/full_local"), help="Run with predictions."),
    out: Path = typer.Option(Path("data/error_coding/errors_to_code.csv")),
    condition: str = typer.Option("codebook_reasoning", help="Condition to sample errors from."),
    per_cell: int = typer.Option(25, help="Errors per (dataset, model)."),
) -> None:
    """Write a spreadsheet of sampled model errors to code by hand (contains dataset text)."""
    from raterbench.analysis.error_coding import export

    n = export(run_dir, Path("data/raw"), out, condition, per_cell)
    typer.echo(f"Wrote {n} errors to {out}; codes are listed in {out.parent / 'TAXONOMY.md'}")


@errors_app.command("summarize")
def errors_summarize(
    coded: Path = typer.Option(Path("data/error_coding/errors_to_code.csv")),
    out_dir: Path = typer.Option(Path("results/analysis")),
) -> None:
    """Count the coded error types into results/analysis/table9_error_taxonomy.*."""
    from raterbench.analysis.error_coding import summarize

    typer.echo(summarize(coded, out_dir))


data_app = typer.Typer(help="Download, sample, and inspect datasets.", no_args_is_help=True)
app.add_typer(data_app, name="data")

_CONFIG_OPT = typer.Option(Path("configs/main.yaml"), "--config", "-c", help="Main config YAML.")


@data_app.command("download")
def data_download(config: Path = _CONFIG_OPT, force: bool = False) -> None:
    """Download raw datasets from pinned sources and verify checksums."""
    from raterbench.config import MainConfig, load_yaml
    from raterbench.data import download, registry

    cfg = load_yaml(config, MainConfig)
    for ds in cfg.dataset_configs():
        info = registry.get(ds.name)
        paths = download.fetch(info.files, cfg.raw_dir / info.name, force=force)
        typer.echo(f"{info.name}: {len(paths)} file(s) verified in {cfg.raw_dir / info.name}")


@data_app.command("sample")
def data_sample(config: Path = _CONFIG_OPT, force: bool = False) -> None:
    """Draw the fixed eval/calibration/train roles and save their IDs to samples_dir."""
    from raterbench.config import MainConfig, load_yaml
    from raterbench.data import registry, splits

    cfg = load_yaml(config, MainConfig)
    for ds in cfg.dataset_configs():
        path = cfg.samples_dir / f"{ds.name}.jsonl"
        if path.exists() and not force:
            typer.echo(f"{path} exists; refusing to overwrite fixed samples (use --force).")
            continue
        items = registry.get(ds.name).load(cfg.raw_dir)
        roles = splits.build_splits(items, ds)
        splits.write_samples(roles, path)
        sizes = ", ".join(f"{r}={len(v)}" for r, v in roles.items())
        typer.echo(f"{ds.name}: wrote {path} ({sizes})")


@data_app.command("stats")
def data_stats(config: Path = _CONFIG_OPT) -> None:
    """Print sizes, label distributions, and subgroup counts; check documented counts."""
    from raterbench.config import MainConfig, load_yaml
    from raterbench.data import registry, splits, stats

    cfg = load_yaml(config, MainConfig)
    failed = False
    for ds in cfg.dataset_configs():
        info = registry.get(ds.name)
        items = info.load(cfg.raw_dir)
        path = cfg.samples_dir / f"{ds.name}.jsonl"
        roles = splits.read_samples(items, path) if path.exists() else None
        typer.echo(stats.report(info, items, roles) + "\n")
        failed |= bool(registry.check_documented(info, items))
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
