"""Rich terminal and markdown reporter for RuleProbe results."""

from __future__ import annotations

import json
from typing import Sequence

from .models import Rule, RuleVerdict, Verdict

VERDICT_COLORS = {
    Verdict.load_bearing: "bold green",
    Verdict.redundant: "yellow",
    Verdict.ignored: "red",
    Verdict.harmful: "bold red",
    Verdict.order_sensitive: "magenta",
    Verdict.untested: "dim",
    Verdict.pinned: "bold cyan",
    Verdict.unverified: "dim yellow",
}

VERDICT_EMOJI = {
    Verdict.load_bearing: "✅",
    Verdict.redundant: "🗑️ ",
    Verdict.ignored: "❌",
    Verdict.harmful: "⚠️ ",
    Verdict.order_sensitive: "↕️ ",
    Verdict.untested: "⬜",
    Verdict.pinned: "📌",
    Verdict.unverified: "❓",
}


def format_verdict_table(
    rules: list[Rule],
    verdicts: list[RuleVerdict],
    conflicts: list[tuple[str, str, str]] | None = None,
    minimal_rules: list[Rule] | None = None,
    baseline_score: float | None = None,
    spent_usd: float = 0.0,
) -> str:
    """Format results as a rich terminal table."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
        import io

        rule_map = {r.id: r for r in rules}
        verdict_map = {(v.rule_id, v.operator): v for v in verdicts}

        table = Table(title="RuleProbe Analysis", show_lines=True)
        table.add_column("ID", style="dim", width=12)
        table.add_column("Rule Text", width=45)
        table.add_column("Verdict", width=16)
        table.add_column("Effect", justify="right", width=8)
        table.add_column("95% CI", width=16)
        table.add_column("p_adj", justify="right", width=7)

        # Group verdicts by rule
        rule_verdicts: dict[str, RuleVerdict] = {}
        for v in verdicts:
            # Prefer 'remove' operator verdict as primary
            if v.rule_id not in rule_verdicts or v.operator.value == "remove":
                rule_verdicts[v.rule_id] = v

        for rule in rules:
            v = rule_verdicts.get(rule.id)
            if v is None:
                table.add_row(
                    rule.id, rule.text[:45], "untested", "-", "-", "-"
                )
                continue

            color = VERDICT_COLORS.get(v.verdict, "white")
            emoji = VERDICT_EMOJI.get(v.verdict, "")
            verdict_str = f"{emoji} {v.verdict.value}"
            effect_str = f"{v.effect_size:+.1f}"
            ci_str = f"[{v.ci_low:+.1f}, {v.ci_high:+.1f}]"
            p_str = f"{v.p_adj:.3f}"

            table.add_row(
                rule.id,
                rule.text[:45] + ("…" if len(rule.text) > 45 else ""),
                Text(verdict_str, style=color),
                effect_str,
                ci_str,
                p_str,
            )

        buf = io.StringIO()
        console = Console(file=buf, width=120)
        console.print(table)

        output = buf.getvalue()

        if conflicts:
            output += "\n⚠️  Conflicts detected:\n"
            for a, b, desc in conflicts:
                output += f"   {a} ↔ {b}: {desc}\n"

        if minimal_rules is not None:
            orig_count = len(rules)
            min_count = len(minimal_rules)
            saved = orig_count - min_count
            pct = round(saved / orig_count * 100) if orig_count else 0
            output += f"\n✨ Minimal prompt: {min_count} of {orig_count} rules (-{pct}% rules)\n"

        output += f"\n💰 Analysis cost: ${spent_usd:.4f}\n"
        return output

    except ImportError:
        return _plain_table(rules, verdicts, conflicts, minimal_rules, spent_usd)


def _plain_table(rules, verdicts, conflicts, minimal_rules, spent_usd) -> str:
    """Plain text fallback when rich is not available."""
    lines = ["RuleProbe Analysis", "=" * 80]
    lines.append(f"{'ID':<15} {'Rule':<40} {'Verdict':<16} {'Effect':>8} {'CI':>18}")
    lines.append("-" * 80)

    rule_verdicts = {v.rule_id: v for v in verdicts}
    for rule in rules:
        v = rule_verdicts.get(rule.id)
        if v:
            lines.append(
                f"{rule.id:<15} {rule.text[:40]:<40} {v.verdict.value:<16} "
                f"{v.effect_size:>+8.1f} [{v.ci_low:>+6.1f},{v.ci_high:>+6.1f}]"
            )
        else:
            lines.append(f"{rule.id:<15} {rule.text[:40]:<40} {'untested':<16}")

    if conflicts:
        lines.append("\nConflicts:")
        for a, b, desc in conflicts:
            lines.append(f"  {a} <-> {b}: {desc}")

    if minimal_rules is not None:
        pct = round((len(rules) - len(minimal_rules)) / len(rules) * 100) if rules else 0
        lines.append(f"\nMinimal prompt: {len(minimal_rules)}/{len(rules)} rules (-{pct}%)")

    lines.append(f"Cost: ${spent_usd:.4f}")
    return "\n".join(lines)


def format_markdown_pr_comment(
    rules: list[Rule],
    verdicts: list[RuleVerdict],
    changed_rule_ids: list[str] | None = None,
    conflicts: list[tuple[str, str, str]] | None = None,
    minimal_rules: list[Rule] | None = None,
    spent_usd: float = 0.0,
) -> str:
    """Generate Markdown suitable for a GitHub PR comment."""
    lines = [
        "## 🔬 RuleProbe Analysis",
        "",
        "| Rule | Text | Verdict | Effect | 95% CI | p_adj |",
        "|------|------|---------|--------|--------|-------|",
    ]

    rule_verdicts = {v.rule_id: v for v in verdicts}
    for rule in rules:
        changed_marker = " 🔄" if changed_rule_ids and rule.id in changed_rule_ids else ""
        v = rule_verdicts.get(rule.id)
        emoji = VERDICT_EMOJI.get(v.verdict, "") if v else "⬜"
        verdict_str = f"{emoji} {v.verdict.value}" if v else "⬜ untested"
        effect_str = f"{v.effect_size:+.1f}" if v else "-"
        ci_str = f"[{v.ci_low:+.1f}, {v.ci_high:+.1f}]" if v else "-"
        p_str = f"{v.p_adj:.3f}" if v else "-"
        text_short = rule.text[:50] + ("…" if len(rule.text) > 50 else "")

        lines.append(f"| `{rule.id}`{changed_marker} | {text_short} | {verdict_str} | {effect_str} | {ci_str} | {p_str} |")

    if conflicts:
        lines += ["", "### ⚠️ Conflicts detected", ""]
        for a, b, desc in conflicts:
            lines.append(f"- `{a}` ↔ `{b}`: {desc}")

    if minimal_rules is not None:
        orig = len(rules)
        mini = len(minimal_rules)
        pct = round((orig - mini) / orig * 100) if orig else 0
        lines += ["", f"### ✨ Minimal prompt: **{mini}/{orig} rules** (saves {pct}% of rules)", ""]

    lines += ["", f"*Analysis cost: ${spent_usd:.4f}*"]
    return "\n".join(lines)


def format_json(rules: list[Rule], verdicts: list[RuleVerdict], **meta) -> str:
    """JSON output."""
    return json.dumps(
        {
            "rules": [r.model_dump() for r in rules],
            "verdicts": [v.model_dump() for v in verdicts],
            **meta,
        },
        indent=2,
        default=str,
    )
