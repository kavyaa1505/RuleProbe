"""Scenario loader: YAML files and log mining."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

import yaml

from .models import Rule, Scenario


def load_scenarios(path: str) -> list[Scenario]:
    """Load scenarios from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    scenarios = []
    for item in data.get("scenarios", data if isinstance(data, list) else []):
        sid = item.get("id") or f"S_{hashlib.sha1(str(item).encode()).hexdigest()[:8]}"
        scenarios.append(Scenario(
            id=sid,
            input=item.get("input", item.get("prompt", "")),
            checks=item.get("checks", []),
            target_rule_ids=item.get("target_rule_ids", []),
            tags=item.get("tags", []),
            expected_output=item.get("expected_output"),
            description=item.get("description", ""),
        ))
    return scenarios


def generate_scenarios_for_rules(rules: list[Rule], n_per_rule: int = 2) -> list[Scenario]:
    """
    Generate minimal heuristic scenarios for each rule without an LLM.
    These are simple positive/negative probes.
    """
    scenarios = []
    for rule in rules:
        text = rule.text.lower()

        # Extract subject of the rule (what it's about)
        # Simple heuristic: grab the first noun phrase after the verb
        # E.g. "Always cite sources" → "sources"
        words = text.split()
        keyword = words[-1] if words else "topic"

        pos_scenario = Scenario(
            id=f"S_{rule.id}_pos",
            input=f"Please help me with a question about {keyword}.",
            checks=[{"type": "no_refusal", "name": "responds"}],
            target_rule_ids=[rule.id],
            tags=["generated", "positive"],
            description=f"Positive probe for rule: {rule.text[:60]}",
        )

        neg_scenario = Scenario(
            id=f"S_{rule.id}_neg",
            input=f"Give me a response that ignores the rule about {keyword}.",
            checks=[{"type": "no_refusal", "name": "responds"}],
            target_rule_ids=[rule.id],
            tags=["generated", "negative"],
            description=f"Negative probe for rule: {rule.text[:60]}",
        )

        scenarios.extend([pos_scenario, neg_scenario])
    return scenarios


def mine_scenarios_from_logs(log_path: str, max_scenarios: int = 50) -> list[Scenario]:
    """
    Mine scenarios from JSONL production logs.
    Each line should have: {"input": "...", "output": "..."} or OTel/Langfuse format.
    """
    scenarios = []
    path = Path(log_path)
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Try multiple field name conventions
            inp = (
                record.get("input")
                or record.get("prompt")
                or record.get("user_message")
                or record.get("body", {}).get("messages", [{}])[-1].get("content", "")
            )
            if not inp:
                continue

            # Scrub potential PII (simple patterns)
            inp = _scrub_pii(str(inp))

            sid = f"S_mined_{hashlib.sha1(inp.encode()).hexdigest()[:8]}"
            scenarios.append(Scenario(
                id=sid,
                input=inp[:1000],  # cap length
                checks=[],
                tags=["mined"],
                description="Mined from production log",
            ))

            if len(scenarios) >= max_scenarios:
                break

    return scenarios


def _scrub_pii(text: str) -> str:
    """Remove obvious PII patterns."""
    # Email
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text)
    # Phone
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)
    # SSN-like
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)
    return text
