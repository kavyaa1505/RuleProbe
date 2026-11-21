"""FastAPI service for RuleProbe (optional — requires 'api' extras)."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Optional

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel as _BM
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from .models import ExperimentConfig, Operator
from .pipeline import RuleProbe

if HAS_FASTAPI:
    app = FastAPI(
        title="RuleProbe API",
        description="Mutation testing for LLM agent instruction files",
        version="0.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # In-memory experiment store (replace with Postgres in production)
    _experiments: dict[str, dict] = {}

    class AnalyzeRequest(_BM):
        instruction_text: str
        scenarios_yaml: Optional[str] = None
        adapter: str = "mock"
        model: str = "gpt-4o-mini"
        runs_per_mutant: int = 3
        budget_usd: float = 2.0
        operators: list[str] = ["remove", "negate"]
        output_format: str = "json"

    class ExperimentStatus(_BM):
        experiment_id: str
        status: str
        result: Optional[dict] = None

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.2.0"}

    @app.post("/analyze", response_model=ExperimentStatus)
    async def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks):
        exp_id = str(uuid.uuid4())
        _experiments[exp_id] = {"status": "pending"}

        async def _run():
            import tempfile, os
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
                f.write(req.instruction_text)
                inst_path = f.name

            sc_path = None
            if req.scenarios_yaml:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
                    f.write(req.scenarios_yaml)
                    sc_path = f.name

            try:
                ops = []
                for op_str in req.operators:
                    try:
                        ops.append(Operator(op_str))
                    except ValueError:
                        pass

                config = ExperimentConfig(
                    instruction_file=inst_path,
                    scenarios_file=sc_path,
                    adapter=req.adapter,
                    model=req.model,
                    runs_per_mutant=req.runs_per_mutant,
                    budget_usd=req.budget_usd,
                    operators=ops,
                    output_format="json",
                )
                rp = RuleProbe(config)
                result = await rp.analyze()

                _experiments[exp_id] = {
                    "status": "done",
                    "result": {
                        "report": result["report"],
                        "spent_usd": result["spent_usd"],
                        "elapsed_s": result["elapsed_s"],
                        "verdicts": [v.model_dump() for v in result["verdicts"]],
                        "rules": [r.model_dump() for r in result["rules"]],
                    },
                }
            except Exception as e:
                _experiments[exp_id] = {"status": "error", "error": str(e)}
            finally:
                os.unlink(inst_path)
                if sc_path:
                    os.unlink(sc_path)

        background_tasks.add_task(_run)
        return ExperimentStatus(experiment_id=exp_id, status="pending")

    @app.get("/experiments/{experiment_id}", response_model=ExperimentStatus)
    async def get_experiment(experiment_id: str):
        if experiment_id not in _experiments:
            raise HTTPException(status_code=404, detail="Experiment not found")
        exp = _experiments[experiment_id]
        return ExperimentStatus(
            experiment_id=experiment_id,
            status=exp["status"],
            result=exp.get("result"),
        )

    @app.get("/experiments")
    async def list_experiments():
        return [
            {"id": k, "status": v["status"]}
            for k, v in _experiments.items()
        ]

else:
    # Stub so imports don't break
    app = None  # type: ignore
