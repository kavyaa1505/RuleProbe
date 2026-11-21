"""Smoke test for NvidiaAdapter (skipped if NVIDIA_API_KEY is unset)."""

import asyncio
import os
import pytest

from ruleprobe.adapters import NvidiaAdapter, BudgetExceededError


@pytest.mark.skipif(
    not os.environ.get("NVIDIA_API_KEY"),
    reason="NVIDIA_API_KEY environment variable not set",
)
def test_nvidia_adapter_smoke():
    model = os.environ.get("AGENT_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    adapter = NvidiaAdapter(model=model, max_concurrency=2, timeout=30.0)

    async def _test():
        res = await adapter.run(
            instructions="You are a helpful test assistant. Reply strictly with the single word PONG.",
            scenario_input="PING",
        )
        assert isinstance(res.output, str)
        assert len(res.output.strip()) > 0
        assert res.tokens_in > 0
        assert res.tokens_out > 0
        assert res.latency_ms > 0
        assert adapter.total_calls >= 1

    asyncio.run(_test())


@pytest.mark.skipif(
    not os.environ.get("NVIDIA_API_KEY"),
    reason="NVIDIA_API_KEY environment variable not set",
)
def test_nvidia_budget_cap():
    model = os.environ.get("AGENT_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    adapter = NvidiaAdapter(model=model, max_calls=1)

    async def _test():
        # First call succeeds
        await adapter.run(
            instructions="Say OK.",
            scenario_input="Hi",
        )
        # Second call hits hard cap
        with pytest.raises(BudgetExceededError):
            await adapter.run(
                instructions="Say OK.",
                scenario_input="Hi again",
            )

    asyncio.run(_test())
