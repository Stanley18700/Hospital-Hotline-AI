"""ADK agent package for the hospital hotline triage system.

Modules in this package own the Vertex / Google ADK-backed triage
orchestration. The five-level triage schema (Red/Orange/Yellow/Green/Blue)
is loaded from ``app/data/er_triage_five_level_system.json`` and exposed
through :mod:`app.agent.triage_config`.

Heavy ADK-dependent modules (:mod:`app.agent.triage_agent`,
:mod:`app.agent.triage_runner`) are intentionally *not* imported here,
so this package stays importable in environments where ``google-adk``
is not yet installed (CI, unit tests, alembic migrations, etc.). Import
them explicitly when you need them.
"""

from app.agent.prompts import build_triage_system_prompt
from app.agent.triage_config import (
    TriageConfigError,
    get_examples_for_level,
    get_team_composition,
    get_triage_config,
    get_triage_level,
    get_triage_levels,
)
from app.agent.triage_state import TriageState

__all__ = [
    "TriageConfigError",
    "TriageState",
    "build_triage_system_prompt",
    "get_examples_for_level",
    "get_team_composition",
    "get_triage_config",
    "get_triage_level",
    "get_triage_levels",
]
