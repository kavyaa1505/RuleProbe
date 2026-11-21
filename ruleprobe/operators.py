"""Mutation operators that transform prompts for ablation experiments."""

from __future__ import annotations

import hashlib
import random
import re
from typing import Protocol, runtime_checkable

from .models import Mutant, Operator, Rule
from .parser import InstructionParser


def _make_mutant_id(operator: Operator, rule_ids: list[str]) -> str:
    raw = f"{operator}:{'_'.join(rule_ids)}"
    return f"M_{hashlib.sha1(raw.encode()).hexdigest()[:10]}"


def _prompt_without_rule(original_prompt: str, rule: Rule) -> str:
    """Remove a rule's text from the prompt."""
    # Try to remove the bullet line
    escaped = re.escape(rule.text)
    # Remove bullet variants: "- text", "* text", "1. text"
    pattern = rf"^[ \t]*[-*+\d.]+\s+{escaped}\s*\n?"
    result = re.sub(pattern, "", original_prompt, flags=re.MULTILINE)
    if result == original_prompt:
        # Try plain text removal
        result = original_prompt.replace(rule.text, "").strip()
    return result.strip()


def _prompt_with_rule_at(original_prompt: str, rule: Rule, position: str) -> str:
    """Move rule to start, middle, or end of prompt."""
    cleaned = _prompt_without_rule(original_prompt, rule)
    lines = [l for l in cleaned.splitlines() if l.strip()]
    rule_line = f"- {rule.text}"

    if position == "start":
        lines.insert(0, rule_line)
    elif position == "end":
        lines.append(rule_line)
    elif position == "middle":
        mid = len(lines) // 2
        lines.insert(mid, rule_line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Operator implementations
# ---------------------------------------------------------------------------

class RemoveOperator:
    """Remove a single rule from the prompt."""

    name = Operator.remove

    def apply(self, prompt: str, rule: Rule, all_rules: list[Rule]) -> Mutant:
        mutant_prompt = _prompt_without_rule(prompt, rule)
        mid = _make_mutant_id(self.name, [rule.id])
        return Mutant(
            id=mid,
            operator=self.name,
            rule_ids=[rule.id],
            prompt_text=mutant_prompt,
        )


class NegateOperator:
    """Negate a rule's meaning in the prompt."""

    name = Operator.negate

    def apply(self, prompt: str, rule: Rule, all_rules: list[Rule]) -> Mutant:
        negated_text = InstructionParser.negate_rule_text(rule.text)
        mutant_prompt = prompt.replace(rule.text, negated_text)
        if mutant_prompt == prompt:
            mutant_prompt = prompt + f"\n- {negated_text}"
        mid = _make_mutant_id(self.name, [rule.id])
        return Mutant(
            id=mid,
            operator=self.name,
            rule_ids=[rule.id],
            prompt_text=mutant_prompt,
        )


class MoveOperator:
    """Move a rule to a different position (start, middle, end)."""

    name = Operator.move

    def __init__(self, position: str = "start"):
        assert position in ("start", "middle", "end")
        self.position = position

    def apply(self, prompt: str, rule: Rule, all_rules: list[Rule]) -> Mutant:
        mutant_prompt = _prompt_with_rule_at(prompt, rule, self.position)
        mid = _make_mutant_id(self.name, [rule.id])
        return Mutant(
            id=mid,
            operator=self.name,
            rule_ids=[rule.id],
            prompt_text=mutant_prompt,
        )


SYNONYM_MAP = {
    "always": "at all times",
    "never": "under no circumstances",
    "ensure": "make sure",
    "avoid": "refrain from",
    "use": "employ",
    "prefer": "prioritize",
    "keep": "maintain",
    "provide": "offer",
    "include": "incorporate",
    "must": "is required to",
    "should": "is expected to",
    "friendly": "warm and approachable",
    "concise": "brief and to the point",
    "clear": "straightforward",
    "cite": "reference",
    "reveal": "disclose",
}


class ParaphraseOperator:
    """Reword a rule using simple synonym substitution (no LLM needed)."""

    name = Operator.paraphrase

    def apply(self, prompt: str, rule: Rule, all_rules: list[Rule]) -> Mutant:
        paraphrased = rule.text
        for word, synonym in SYNONYM_MAP.items():
            pattern = r"\b" + re.escape(word) + r"\b"
            paraphrased = re.sub(pattern, synonym, paraphrased, flags=re.IGNORECASE, count=1)
        mutant_prompt = prompt.replace(rule.text, paraphrased)
        mid = _make_mutant_id(self.name, [rule.id])
        return Mutant(
            id=mid,
            operator=self.name,
            rule_ids=[rule.id],
            prompt_text=mutant_prompt,
        )


class DuplicateOperator:
    """Duplicate a rule (test over-emphasis effects)."""

    name = Operator.duplicate

    def apply(self, prompt: str, rule: Rule, all_rules: list[Rule]) -> Mutant:
        rule_line = f"- {rule.text}"
        mutant_prompt = prompt.replace(rule.text, f"{rule.text}\n{rule_line}")
        mid = _make_mutant_id(self.name, [rule.id])
        return Mutant(
            id=mid,
            operator=self.name,
            rule_ids=[rule.id],
            prompt_text=mutant_prompt,
        )


class PairRemoveOperator:
    """Remove two rules at once — detects interactions and conflicts."""

    name = Operator.pair_remove

    def apply(self, prompt: str, rule: Rule, partner: Rule) -> Mutant:
        mutant_prompt = _prompt_without_rule(prompt, rule)
        mutant_prompt = _prompt_without_rule(mutant_prompt, partner)
        mid = _make_mutant_id(self.name, [rule.id, partner.id])
        return Mutant(
            id=mid,
            operator=self.name,
            rule_ids=[rule.id, partner.id],
            prompt_text=mutant_prompt,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OPERATOR_REGISTRY: dict[str, object] = {
    "remove": RemoveOperator(),
    "negate": NegateOperator(),
    "move_start": MoveOperator("start"),
    "move_middle": MoveOperator("middle"),
    "move_end": MoveOperator("end"),
    "paraphrase": ParaphraseOperator(),
    "duplicate": DuplicateOperator(),
}


def get_operator(name: str):
    return OPERATOR_REGISTRY.get(name)
