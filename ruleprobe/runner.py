"""Async runner: executes experiment plans with caching, retries, and concurrency."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from typing import AsyncIterator, Callable

from .adapters import AgentAdapter, AgentResult
from .models import (
    ExperimentPlan,
    Mutant,
    Run,
    Scenario,
)
from .scorer import score_run

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory + SQLite cache
# ---------------------------------------------------------------------------

class RunCache:
    """Content-addressed cache keyed on (prompt_hash, scenario_hash, model, seed)."""

    def __init__(self, db_path: str = "ruleprobe_cache.db"):
        self.db_path = db_path
        self._mem: dict[str, Run] = {}
        self._db = None

    async def _get_db(self):
        if self._db is None:
            try:
                import aiosqlite
                self._db = await aiosqlite.connect(self.db_path)
                await self._db.execute("""
                    CREATE TABLE IF NOT EXISTS run_cache (
                        cache_key TEXT PRIMARY KEY,
                        run_json TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                """)
                await self._db.commit()
            except ImportError:
                logger.warning("aiosqlite not installed — cache disabled")
                self._db = None
        return self._db

    @staticmethod
    def make_key(prompt_hash: str, scenario_id: str, model: str, seed: int | None, adapter_name: str = "") -> str:
        raw = f"{adapter_name}:{prompt_hash}:{scenario_id}:{model}:{seed}"
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    async def get(self, key: str) -> Run | None:
        if key in self._mem:
            return self._mem[key]
        db = await self._get_db()
        if db:
            async with db.execute("SELECT run_json FROM run_cache WHERE cache_key = ?", (key,)) as cur:
                row = await cur.fetchone()
                if row:
                    run = Run.model_validate_json(row[0])
                    run.cache_hit = True
                    self._mem[key] = run
                    return run
        return None

    async def set(self, key: str, run: Run) -> None:
        self._mem[key] = run
        db = await self._get_db()
        if db:
            await db.execute(
                "INSERT OR REPLACE INTO run_cache VALUES (?, ?, ?)",
                (key, run.model_dump_json(), time.time()),
            )
            await db.commit()

    async def close(self):
        if self._db:
            await self._db.close()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Async runner that executes an ExperimentPlan.
    - Semaphore-limited concurrency (default 5)
    - Content-addressed cache
    - Exponential backoff retries (3 attempts)
    - Budget guard
    """

    def __init__(
        self,
        adapter: AgentAdapter,
        concurrency: int = 5,
        max_retries: int = 3,
        cache: RunCache | None = None,
        budget_usd: float = float("inf"),
        progress_cb: Callable[[Run], None] | None = None,
    ):
        self.adapter = adapter
        self.semaphore = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.cache = cache or RunCache()
        self.budget_usd = budget_usd
        self.progress_cb = progress_cb
        self._spent: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def spent(self) -> float:
        return self._spent

    async def _budget_ok(self) -> bool:
        async with self._lock:
            return self._spent < self.budget_usd

    async def _charge(self, cost: float) -> None:
        async with self._lock:
            self._spent += cost

    async def _run_one(
        self,
        mutant: Mutant,
        scenario: Scenario,
        model: str,
        repeat_idx: int,
    ) -> Run:
        """Execute one (mutant, scenario, repeat) with caching and retries."""
        seed = 1000 + repeat_idx
        adapter_name = self.adapter.__class__.__name__
        cache_key = RunCache.make_key(mutant.prompt_hash, scenario.id, model, seed, adapter_name=adapter_name)

        # Cache hit?
        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        # Budget guard
        if not await self._budget_ok():
            return Run(
                id=str(uuid.uuid4()),
                mutant_id=mutant.id,
                scenario_id=scenario.id,
                repeat_idx=repeat_idx,
                model=model,
                error="budget_exceeded",
            )

        # Execute with retries
        last_error = ""
        for attempt in range(self.max_retries):
            try:
                async with self.semaphore:
                    result = await self.adapter.run(
                        instructions=mutant.prompt_text,
                        scenario_input=scenario.input,
                        seed=seed,
                    )
                run = Run(
                    id=str(uuid.uuid4()),
                    mutant_id=mutant.id,
                    scenario_id=scenario.id,
                    repeat_idx=repeat_idx,
                    model=model,
                    output=result.output,
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    cost=result.cost,
                    latency_ms=result.latency_ms,
                )
                await self.cache.set(cache_key, run)
                await self._charge(result.cost)
                return run

            except Exception as e:
                last_error = str(e)
                wait = 2 ** attempt
                logger.warning(f"Run attempt {attempt + 1} failed: {e}. Retrying in {wait}s")
                await asyncio.sleep(wait)

        return Run(
            id=str(uuid.uuid4()),
            mutant_id=mutant.id,
            scenario_id=scenario.id,
            repeat_idx=repeat_idx,
            model=model,
            error=last_error,
        )

    async def run_experiment(self, plan: ExperimentPlan) -> list[Run]:
        """Execute all runs in the plan concurrently."""
        # Build lookup maps
        mutant_map = {m.id: m for m in plan.mutants}
        scenario_map = {s.id: s for s in plan.scenarios}

        tasks = []
        for mutant_id, scenario_id, model, repeat_idx in plan.runs:
            mutant = mutant_map[mutant_id]
            scenario = scenario_map[scenario_id]
            tasks.append(self._run_one(mutant, scenario, model, repeat_idx))

        results = await asyncio.gather(*tasks, return_exceptions=False)
        runs = list(results)

        if self.progress_cb:
            for run in runs:
                self.progress_cb(run)

        await self.cache.close()
        return runs
