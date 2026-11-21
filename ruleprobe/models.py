"""Pydantic data models for RuleProbe."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    load_bearing = "load_bearing"
    redundant = "redundant"
    ignored = "ignored"
    harmful = "harmful"
    order_sensitive = "order_sensitive"
    untested = "untested"
    pinned = "pinned"
    unverified = "unverified"
    inconclusive = "inconclusive"


class Operator(str, Enum):
    remove = "remove"
    negate = "negate"
    move = "move"
    paraphrase = "paraphrase"
    duplicate = "duplicate"
    pair_remove = "pair_remove"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class Rule(BaseModel):
    id: str
    text: str
    section: str = ""
    position: int = 0
    hash: str = ""
    pinned: bool = False

    @model_validator(mode="after")
    def _set_hash(self) -> "Rule":
        if not self.hash:
            raw = f"{self.position}:{self.text}"
            self.hash = hashlib.sha1(raw.encode()).hexdigest()[:12]
        if not self.id:
            self.id = f"R{self.position}_{self.hash[:6]}"
        return self


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class Scenario(BaseModel):
    id: str
    input: Any
    checks: list[dict[str, Any]] = Field(default_factory=list)
    target_rule_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    expected_output: str | None = None
    description: str = ""


class Mutant(BaseModel):
    id: str
    operator: Operator
    rule_ids: list[str]
    prompt_text: str
    prompt_hash: str = ""

    @model_validator(mode="after")
    def _set_hash(self) -> "Mutant":
        if not self.prompt_hash:
            self.prompt_hash = hashlib.sha256(self.prompt_text.encode()).hexdigest()[:16]
        return self


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None


class Run(BaseModel):
    id: str
    mutant_id: str
    scenario_id: str
    repeat_idx: int
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    output: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    trace: dict[str, Any] | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    cache_hit: bool = False
    error: str | None = None


class BehavioralFingerprint(BaseModel):
    tool_call_sequence: list[str] = Field(default_factory=list)
    step_count: int = 0
    tokens_used: int = 0
    refusal: bool = False
    latency_ms: float = 0.0


class Score(BaseModel):
    run_id: str
    check_scores: dict[str, bool] = Field(default_factory=dict)
    judge_score: float | None = None
    judge_rationale: str = ""
    aggregate_score: float = 0.0
    fingerprint: BehavioralFingerprint = Field(default_factory=BehavioralFingerprint)
    judge_confidence: float = 1.0


class RuleVerdict(BaseModel):
    rule_id: str
    operator: Operator
    effect_size: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    p_value: float = 1.0
    p_adj: float = 1.0
    verdict: Verdict = Verdict.untested
    supporting_scenarios: list[str] = Field(default_factory=list)
    compliance_rate: float | None = None
    token_cost: int = 0
    is_stable: bool = True


class ExperimentConfig(BaseModel):
    instruction_file: str
    scenarios_file: str | None = None
    model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    adapter: str = "openai"
    runs_per_mutant: int = 5
    budget_usd: float = 5.0
    concurrency: int = 5
    materiality_threshold: float = 5.0
    operators: list[Operator] = Field(default_factory=lambda: [Operator.remove, Operator.negate])
    pinned_rules: list[str] = Field(default_factory=list)
    cache_db: str = "ruleprobe_cache.db"
    output_format: str = "table"


class Experiment(BaseModel):
    id: str
    config: ExperimentConfig
    budget: float
    spent: float = 0.0
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    commit_sha: str = ""
    verdicts: list[RuleVerdict] = Field(default_factory=list)
    minimal_rules: list[str] | None = None
    token_savings_pct: float | None = None


class ExperimentPlan(BaseModel):
    experiment_id: str
    runs: list[tuple[str, str, str, int]] = Field(default_factory=list)
    # (mutant_id, scenario_id, model, repeat_idx)
    estimated_cost: float = 0.0
    estimated_tokens: int = 0
    mutants: list[Mutant] = Field(default_factory=list)
    scenarios: list[Scenario] = Field(default_factory=list)
