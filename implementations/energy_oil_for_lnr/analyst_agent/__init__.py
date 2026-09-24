"""LNR LNR stock analyst agent module.

Exports the :class:`AgentConfig` factories, prompt builder, and predictor
convenience factory for the energy/oil reference implementation.
"""

from energy_oil_for_lnr.analyst_agent.agent import (
    LnrPriceForecastPromptBuilder,
    build_lnr_agent_predictor,
    build_lnr_basic_config,
    build_lnr_code_exec_config,
    build_lnr_multitask_news_config,
    build_lnr_news_config,
    build_lnr_tool_config,
    compress_history,
)


__all__ = [
    "LnrPriceForecastPromptBuilder",
    "build_lnr_agent_predictor",
    "build_lnr_basic_config",
    "build_lnr_code_exec_config",
    "build_lnr_multitask_news_config",
    "build_lnr_news_config",
    "build_lnr_tool_config",
    "compress_history",
]
