"""SQLite storage backend for runs, scores, verdicts, and experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class Storage:
    """Async SQLite storage for RuleProbe experiments."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        config_json TEXT,
        budget REAL,
        spent REAL DEFAULT 0,
        started_at TEXT,
        finished_at TEXT,
        commit_sha TEXT DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS rules (
        id TEXT PRIMARY KEY,
        experiment_id TEXT,
        text TEXT,
        section TEXT,
        position INTEGER,
        hash TEXT,
        pinned INTEGER DEFAULT 0,
        FOREIGN KEY(experiment_id) REFERENCES experiments(id)
    );
    CREATE TABLE IF NOT EXISTS scenarios (
        id TEXT PRIMARY KEY,
        experiment_id TEXT,
        input_json TEXT,
        checks_json TEXT,
        target_rule_ids_json TEXT,
        tags_json TEXT
    );
    CREATE TABLE IF NOT EXISTS mutants (
        id TEXT PRIMARY KEY,
        experiment_id TEXT,
        operator TEXT,
        rule_ids_json TEXT,
        prompt_hash TEXT
    );
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        mutant_id TEXT,
        scenario_id TEXT,
        repeat_idx INTEGER,
        model TEXT,
        output TEXT,
        tokens_in INTEGER DEFAULT 0,
        tokens_out INTEGER DEFAULT 0,
        cost REAL DEFAULT 0,
        latency_ms REAL DEFAULT 0,
        cache_hit INTEGER DEFAULT 0,
        error TEXT
    );
    CREATE TABLE IF NOT EXISTS scores (
        run_id TEXT PRIMARY KEY,
        check_scores_json TEXT,
        judge_score REAL,
        judge_rationale TEXT,
        aggregate_score REAL,
        FOREIGN KEY(run_id) REFERENCES runs(id)
    );
    CREATE TABLE IF NOT EXISTS verdicts (
        id TEXT PRIMARY KEY,
        experiment_id TEXT,
        rule_id TEXT,
        operator TEXT,
        effect_size REAL,
        ci_low REAL,
        ci_high REAL,
        p_value REAL,
        p_adj REAL,
        verdict TEXT,
        supporting_scenarios_json TEXT,
        compliance_rate REAL
    );
    """

    def __init__(self, db_path: str = "ruleprobe.db"):
        self.db_path = db_path
        self._db = None

    async def init(self):
        try:
            import aiosqlite
            self._db = await aiosqlite.connect(self.db_path)
            self._db.row_factory = aiosqlite.Row
            for stmt in self.SCHEMA.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    await self._db.execute(stmt)
            await self._db.commit()
            logger.info(f"Storage initialized at {self.db_path}")
        except ImportError:
            logger.warning("aiosqlite not installed — storage disabled")

    async def close(self):
        if self._db:
            await self._db.close()

    async def save_run(self, run) -> None:
        if not self._db:
            return
        await self._db.execute(
            """INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run.id, run.mutant_id, run.scenario_id, run.repeat_idx,
                run.model, run.output, run.tokens_in, run.tokens_out,
                run.cost, run.latency_ms, int(run.cache_hit), run.error,
            ),
        )
        await self._db.commit()

    async def save_verdict(self, experiment_id: str, verdict) -> None:
        if not self._db:
            return
        import uuid
        await self._db.execute(
            """INSERT OR REPLACE INTO verdicts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(uuid.uuid4()), experiment_id, verdict.rule_id,
                verdict.operator, verdict.effect_size, verdict.ci_low,
                verdict.ci_high, verdict.p_value, verdict.p_adj,
                verdict.verdict.value,
                json.dumps(verdict.supporting_scenarios),
                verdict.compliance_rate,
            ),
        )
        await self._db.commit()

    async def get_verdicts(self, experiment_id: str) -> list[dict]:
        if not self._db:
            return []
        async with self._db.execute(
            "SELECT * FROM verdicts WHERE experiment_id = ?", (experiment_id,)
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]
