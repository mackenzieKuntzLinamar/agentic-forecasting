"""Adaptive LNR LNR stock analyst agent module.

Exports the :class:`AgentConfig` factory, prompt builder, and predictor
convenience factory for the adaptive energy/oil reference implementation.
"""

from energy_oil_for_lnr.adaptive_agent.agent import (
    LnrAdaptiveForecastPromptBuilder,
    build_lnr_adaptive_config,
    build_lnr_adaptive_predictor,
)
from energy_oil_for_lnr.adaptive_agent.skill_tools import build_skill_tools
from energy_oil_for_lnr.analyst_agent import compress_history


__all__ = [
    "LnrAdaptiveForecastPromptBuilder",
    "build_skill_tools",
    "build_lnr_adaptive_config",
    "build_lnr_adaptive_predictor",
    "compress_history",
]
