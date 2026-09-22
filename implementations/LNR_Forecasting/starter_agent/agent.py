"""Starter agent for LNR.TO stock-return forecasting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentPredictor, ContinuousAgentForecastOutput, build_adk_agent
from aieng.forecasting.methods.agentic.agent_factory import AgentConfig, CodeExecutionConfig, ContextRetrievalConfig
from aieng.forecasting.models import LITE_MODEL
from pydantic import BaseModel


_SKILLS_ROOT = Path(__file__).parent / "skills"
_FORECASTING_SKILL = _SKILLS_ROOT / "forecasting"
_RESEARCH_SKILL = _SKILLS_ROOT / "research-playbook"
_CODE_ANALYSIS_SKILL = _SKILLS_ROOT / "code-analysis-playbook"


class LnrStarterPromptBuilder(BaseModel):
    """Serialize recent LNR return history and optional covariate snapshots."""

    model_config = {"extra": "forbid"}

    history: int = 64
    covariate_series_ids: list[str] = []

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        frame = context.get_series(task.target_series_id).tail(self.history)
        rows = ["date,log_return"] + [
            f"{pd.Timestamp(ts).date()},{float(value):.6f}" for ts, value in zip(frame["timestamp"], frame["value"])
        ]
        snapshot: dict[str, float] = {}
        for series_id in self.covariate_series_ids:
            try:
                covariate = context.get_series(series_id)
            except Exception:  # noqa: BLE001
                continue
            if not covariate.empty:
                snapshot[series_id] = round(float(covariate["value"].iloc[-1]), 6)
        payload: dict[str, Any] = {
            "task": task.task_id,
            "as_of": str(context.as_of)[:10],
            "horizons": list(task.horizons),
            "standard_quantiles": list(STANDARD_QUANTILES),
            "target_summary": {
                "ticker": "LNR.TO",
                "last_log_return": float(frame["value"].iloc[-1]),
                "last_date": str(pd.Timestamp(frame["timestamp"].iloc[-1]).date()),
                "n_obs": int(len(frame)),
            },
            "target_history_csv": "\n".join(rows),
        }
        if snapshot:
            payload["covariate_snapshot"] = snapshot
        return json.dumps(payload, indent=2)


def _starter_instruction() -> str:
    return (
        "You are a transparent equity-market analyst focused on Linamar Corporation (LNR.TO), "
        "a Canadian industrial and auto-supply company. Consider rates, Canadian and US auto "
        "production, tariffs, industrial demand, CAD/USD, commodities, and broad risk sentiment. "
        "For conversation, answer directly and concisely. When asked for a structured forecast, "
        "produce a calibrated probabilistic forecast from the supplied evidence and do not invent facts."
    )


_CONTEXT_RETRIEVAL_INSTRUCTION = """\
You are a cutoff-aware research specialist for LNR.TO. Search for recent evidence about Linamar,
Canadian industrial demand, auto production, tariffs, rates, commodities, currency, and market risk.
Use only facts from retrieved results, respect the stated cutoff, and say when evidence is insufficient.
"""


def build_starter_agent_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    *,
    enable_search: bool = True,
    enable_code_exec: bool = True,
) -> AgentConfig:
    """Build the toggle-driven LNR starter-agent configuration."""
    skills_dirs: list[Path] = [_FORECASTING_SKILL]
    if enable_search:
        skills_dirs.append(_RESEARCH_SKILL)
    if enable_code_exec:
        skills_dirs.append(_CODE_ANALYSIS_SKILL)
    context_retrieval = (
        ContextRetrievalConfig(
            enabled=True,
            instruction=_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
        )
        if enable_search
        else ContextRetrievalConfig()
    )
    return AgentConfig(
        name="lnr_starter_agent",
        model=model,
        instruction=_starter_instruction(),
        max_output_tokens=16_384 if enable_code_exec else None,
        context_retrieval=context_retrieval,
        code_execution=CodeExecutionConfig(enabled=enable_code_exec),
        skills_dirs=skills_dirs,
    )


class _StarterForecastPromptBuilder:
    def __init__(self, inner: Callable[..., str], output_schema_json: str) -> None:
        self._inner = inner
        self._schema_json = output_schema_json

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        payload = json.loads(self._inner(task=task, context=context))
        payload["instructions"] = (
            "Produce a calibrated probabilistic forecast and return it by calling `set_model_response` "
            "with a `json_response` string matching `output_schema` exactly."
        )
        payload["output_schema"] = self._schema_json
        return json.dumps(payload, indent=2)


def build_starter_agent_predictor(
    config: AgentConfig,
    *,
    covariate_series_ids: list[str] | None = None,
) -> AgentPredictor:
    """Wrap an LNR starter configuration as a continuous forecast predictor."""
    return AgentPredictor(
        agent_config=config,
        prompt_builder=_StarterForecastPromptBuilder(
            LnrStarterPromptBuilder(covariate_series_ids=covariate_series_ids or []),
            ContinuousAgentForecastOutput.prompt_schema_json(),
        ),
        output_schema=ContinuousAgentForecastOutput,
    )


def __getattr__(name: str) -> Any:
    """Expose ``root_agent`` lazily for ADK web use."""
    if name == "root_agent":
        return build_adk_agent(build_starter_agent_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
