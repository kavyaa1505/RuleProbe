"""Typer CLI for RuleProbe."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel

from .models import ExperimentConfig, Operator
from .parser import parse_file

app = typer.Typer(
    name="ruleprobe",
    help="🔬 Mutation testing for LLM agent instruction files.",
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# ruleprobe analyze
# ---------------------------------------------------------------------------

@app.command()
def analyze(
    instruction_file: str = typer.Argument(..., help="Path to AGENTS.md / CLAUDE.md / system prompt"),
    scenarios: Optional[str] = typer.Option(None, "--scenarios", "-s", help="Path to scenarios YAML"),
    adapter: str = typer.Option("mock", "--adapter", "-a", help="Agent adapter: mock | openai | anthropic"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="LLM model name"),
    runs: int = typer.Option(5, "--runs", "-n", help="Runs per mutant"),
    budget: float = typer.Option(5.0, "--budget", "-b", help="Budget cap in USD (e.g. 5usd)"),
    operators: str = typer.Option("remove,negate", "--operators", help="Comma-separated operators"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table | json | markdown"),
    pinned: Optional[str] = typer.Option(None, "--pinned", help="Comma-separated rule IDs to pin"),
    cache_db: str = typer.Option("ruleprobe_cache.db", "--cache", help="Cache DB path"),
    materiality: float = typer.Option(2.0, "--materiality", help="Minimum effect size to report"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Analyze an instruction file and report per-rule verdicts."""
    import logging
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Parse budget string like "5usd" or "5.0"
    budget_val = float(str(budget).replace("usd", "").strip())

    op_list = []
    for op_str in operators.split(","):
        op_str = op_str.strip().lower()
        try:
            op_list.append(Operator(op_str))
        except ValueError:
            console.print(f"[yellow]Unknown operator '{op_str}' — skipping[/yellow]")

    pinned_list = [p.strip() for p in pinned.split(",")] if pinned else []

    config = ExperimentConfig(
        instruction_file=instruction_file,
        scenarios_file=scenarios,
        model=model,
        adapter=adapter,
        runs_per_mutant=runs,
        budget_usd=budget_val,
        operators=op_list,
        pinned_rules=pinned_list,
        cache_db=cache_db,
        output_format=output,
        materiality_threshold=materiality,
    )

    console.print(Panel(
        f"[bold]RuleProbe[/bold] analyzing [cyan]{instruction_file}[/cyan]\n"
        f"Adapter: [green]{adapter}[/green]  Model: [green]{model}[/green]  "
        f"Budget: [yellow]${budget_val:.2f}[/yellow]  Runs/mutant: [blue]{runs}[/blue]",
        title="🔬 RuleProbe",
    ))

    from .pipeline import RuleProbe
    rp = RuleProbe(config)
    result = asyncio.run(rp.analyze())

    console.print(result["report"])
    console.print(f"\n[dim]Elapsed: {result['elapsed_s']:.1f}s[/dim]")


# ---------------------------------------------------------------------------
# ruleprobe minimize
# ---------------------------------------------------------------------------

@app.command()
def minimize(
    instruction_file: str = typer.Argument(..., help="Instruction file to minimize"),
    scenarios: Optional[str] = typer.Option(None, "--scenarios", "-s"),
    adapter: str = typer.Option("mock", "--adapter", "-a"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m"),
    out: Optional[str] = typer.Option(None, "--out", help="Write minimal prompt to this file"),
    runs: int = typer.Option(5, "--runs", "-n"),
    budget: float = typer.Option(10.0, "--budget", "-b"),
    pass_threshold: float = typer.Option(90.0, "--threshold", help="Min % of baseline score to keep"),
):
    """Find the minimal equivalent instruction set."""
    from .minimizer import PromptMinimizer
    from .parser import parse_file, InstructionParser
    from .adapters import get_adapter
    from .models import ExperimentConfig

    config = ExperimentConfig(
        instruction_file=instruction_file,
        scenarios_file=scenarios,
        adapter=adapter,
        model=model,
        runs_per_mutant=runs,
        budget_usd=budget,
    )

    async def _run():
        from .pipeline import RuleProbe
        rp = RuleProbe(config)
        result = await rp.analyze()

        adapter_inst = get_adapter(adapter, model=model)

        async def runner_fn(prompt: str, rules) -> float:
            from .scenarios import generate_scenarios_for_rules
            from .scorer import score_run
            from .models import Run
            import uuid, time

            if not result["plan"].scenarios:
                return 0.0
            scenario = result["plan"].scenarios[0]
            agent_res = await adapter_inst.run(prompt, scenario.input, seed=42)
            run = Run(
                id=str(uuid.uuid4()),
                mutant_id="min_probe",
                scenario_id=scenario.id,
                repeat_idx=0,
                model=model,
                output=agent_res.output,
                tokens_in=agent_res.tokens_in,
                tokens_out=agent_res.tokens_out,
                cost=agent_res.cost,
                latency_ms=agent_res.latency_ms,
            )
            score = score_run(run, scenario.checks)
            return score.aggregate_score

        minimizer = PromptMinimizer(runner_fn=runner_fn, pass_threshold=pass_threshold)
        min_rules, baseline = await minimizer.minimize(
            Path(instruction_file).read_text(encoding="utf-8"),
            result["rules"],
            result["verdicts"],
        )
        return min_rules, baseline, result["rules"]

    min_rules, baseline, all_rules = asyncio.run(_run())
    orig = len(all_rules)
    mini = len(min_rules)
    pct = round((orig - mini) / orig * 100) if orig else 0

    console.print(f"\n✨ Minimal prompt: [bold]{mini}/{orig}[/bold] rules (saves [green]{pct}%[/green])\n")

    minimized_text = "\n".join(f"- {r.text}" for r in sorted(min_rules, key=lambda x: x.position))
    if out:
        Path(out).write_text(minimized_text, encoding="utf-8")
        console.print(f"[green]Written to {out}[/green]")
    else:
        console.print("[bold]Minimal instruction set:[/bold]")
        for r in sorted(min_rules, key=lambda x: x.position):
            console.print(f"  - {r.text}")


# ---------------------------------------------------------------------------
# ruleprobe diff
# ---------------------------------------------------------------------------

@app.command()
def diff(
    rev_range: str = typer.Argument(..., help="Git revision range e.g. main..HEAD"),
    file: str = typer.Option(..., "--file", "-f", help="Instruction file to diff"),
    adapter: str = typer.Option("mock", "--adapter"),
    model: str = typer.Option("gpt-4o-mini", "--model"),
    budget: float = typer.Option(2.0, "--budget"),
    output: str = typer.Option("table", "--output"),
):
    """Analyze only the rules that changed between git revisions."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "diff", rev_range, "--", file],
            capture_output=True, text=True, check=True
        )
        diff_text = result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[red]git diff failed — running full analysis instead[/red]")
        diff_text = ""

    # Parse added/modified lines from diff
    changed_lines = []
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            changed_lines.append(line[1:].strip())

    console.print(f"[dim]{len(changed_lines)} changed lines detected[/dim]")

    # Run full analysis (cache makes unchanged rules free on rerun)
    config = ExperimentConfig(
        instruction_file=file,
        adapter=adapter,
        model=model,
        budget_usd=budget,
        output_format=output,
    )

    from .pipeline import RuleProbe
    rp = RuleProbe(config)
    result_data = asyncio.run(rp.analyze())
    console.print(result_data["report"])


# ---------------------------------------------------------------------------
# ruleprobe parse (utility)
# ---------------------------------------------------------------------------

@app.command()
def parse(
    instruction_file: str = typer.Argument(..., help="Instruction file to parse"),
    show_ids: bool = typer.Option(True, "--ids/--no-ids"),
):
    """Parse and display extracted rules (dry run, no LLM calls)."""
    rules = parse_file(instruction_file)
    console.print(f"[bold]Parsed {len(rules)} rules from {instruction_file}[/bold]\n")
    for rule in rules:
        prefix = f"[dim]{rule.id}[/dim] " if show_ids else ""
        section = f" [cyan]({rule.section})[/cyan]" if rule.section else ""
        pinned = " [bold cyan]📌[/bold cyan]" if rule.pinned else ""
        console.print(f"  {prefix}{rule.text}{section}{pinned}")


# ---------------------------------------------------------------------------
# ruleprobe compare (cross-model)
# ---------------------------------------------------------------------------

@app.command()
def compare(
    instruction_file: str = typer.Argument(...),
    models: str = typer.Option(..., "--models", help="Comma-separated model names to compare"),
    scenarios: Optional[str] = typer.Option(None, "--scenarios", "-s"),
    budget: float = typer.Option(5.0, "--budget"),
    runs: int = typer.Option(3, "--runs"),
):
    """Compare verdicts across multiple models (migration analysis)."""
    from .pipeline import RuleProbe
    model_list = [m.strip() for m in models.split(",")]
    all_results = {}

    async def _run_all():
        for m in model_list:
            config = ExperimentConfig(
                instruction_file=instruction_file,
                scenarios_file=scenarios,
                adapter="openai",
                model=m,
                runs_per_mutant=runs,
                budget_usd=budget / len(model_list),
            )
            rp = RuleProbe(config)
            res = await rp.analyze()
            all_results[m] = res

    asyncio.run(_run_all())

    # Cross-model comparison report
    console.print("\n[bold]Cross-Model Comparison[/bold]\n")
    if not all_results:
        return

    rules = list(all_results.values())[0]["rules"]
    for rule in rules:
        verdicts_by_model = {}
        for m, res in all_results.items():
            verdict_map = {v.rule_id: v for v in res["verdicts"]}
            if rule.id in verdict_map:
                verdicts_by_model[m] = verdict_map[rule.id].verdict.value
        if len(set(verdicts_by_model.values())) > 1:
            console.print(f"[yellow]⚡ Divergent:[/yellow] {rule.text[:50]}")
            for m, v in verdicts_by_model.items():
                console.print(f"   {m}: {v}")


if __name__ == "__main__":
    app()
