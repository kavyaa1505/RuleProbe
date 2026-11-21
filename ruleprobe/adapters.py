"""Agent adapter interface and built-in adapters."""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None


@dataclass
class AgentResult:
    output: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    trace: dict | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0


@runtime_checkable
class AgentAdapter(Protocol):
    """The only contract an agent must satisfy to work with RuleProbe."""

    async def run(
        self,
        instructions: str,
        scenario_input: Any,
        *,
        seed: int | None = None,
    ) -> AgentResult: ...


# ---------------------------------------------------------------------------
# Mock adapter — deterministic, no API key needed
# ---------------------------------------------------------------------------

class MockAdapter:
    """
    Fake adapter for testing. Checks whether specific rule text appears
    in the instructions and simulates compliance / non-compliance.
    """

    def __init__(self, compliance_rate: float = 0.9, latency_ms: float = 50.0, **kwargs):
        self.compliance_rate = compliance_rate
        self.base_latency = latency_ms

    async def run(
        self,
        instructions: str,
        scenario_input: Any,
        *,
        seed: int | None = None,
    ) -> AgentResult:
        rng = random.Random(seed)
        await asyncio.sleep(self.base_latency / 1000)

        input_str = str(scenario_input)
        # Score based on how many rule-like lines are in instructions
        rule_lines = [l for l in instructions.splitlines() if l.strip().startswith("-")]
        n_rules = len(rule_lines)

        # Simulate that more rules → slightly better structured output
        score_base = min(0.5 + n_rules * 0.01, 0.95)
        compliant = rng.random() < (score_base * self.compliance_rate)

        output = (
            f"Response to: {input_str[:80]}. "
            + ("All guidelines followed." if compliant else "Some guidelines may not apply here.")
        )

        tokens_in = len(instructions.split()) + len(input_str.split())
        tokens_out = len(output.split())
        cost = (tokens_in * 0.15 + tokens_out * 0.60) / 1_000_000  # GPT-4o-mini pricing

        return AgentResult(
            output=output,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            latency_ms=self.base_latency + rng.uniform(0, 20),
        )


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------

class OpenAIAdapter:
    """Calls OpenAI chat completions API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise RuntimeError("openai package not installed. Run: uv add openai")
        return self._client

    async def run(
        self,
        instructions: str,
        scenario_input: Any,
        *,
        seed: int | None = None,
    ) -> AgentResult:
        client = self._get_client()
        input_str = str(scenario_input)

        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": input_str},
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed

        t0 = time.perf_counter()
        response = await client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        choice = response.choices[0]
        output = choice.message.content or ""
        usage = response.usage

        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        # GPT-4o-mini pricing (approximate)
        cost = (tokens_in * 0.15 + tokens_out * 0.60) / 1_000_000

        return AgentResult(
            output=output,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

class AnthropicAdapter:
    """Calls Anthropic messages API."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise RuntimeError("anthropic package not installed. Run: uv add anthropic")
        return self._client

    async def run(
        self,
        instructions: str,
        scenario_input: Any,
        *,
        seed: int | None = None,
    ) -> AgentResult:
        client = self._get_client()
        input_str = str(scenario_input)

        t0 = time.perf_counter()
        response = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=instructions,
            messages=[{"role": "user", "content": input_str}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        output = response.content[0].text if response.content else ""
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        # Claude Haiku pricing (approximate)
        cost = (tokens_in * 0.25 + tokens_out * 1.25) / 1_000_000

        return AgentResult(
            output=output,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_adapter(name: str, **kwargs) -> AgentAdapter:
    from .benchmark import BenchmarkAdapter
    adapters = {
        "mock": MockAdapter,
        "openai": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "benchmark": BenchmarkAdapter,
    }
    cls = adapters.get(name)
    if cls is None:
        raise ValueError(f"Unknown adapter '{name}'. Choose from: {list(adapters)}")
    return cls(**kwargs)
