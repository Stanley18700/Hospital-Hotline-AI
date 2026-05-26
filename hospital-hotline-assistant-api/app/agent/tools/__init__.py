"""ADK tool functions exposed to the triage agent.

Tools are plain Python functions with type-annotated args and
docstrings. The ADK runtime introspects the signature and docstring to
build the function-calling schema the LLM sees, and auto-wraps the
callable as a ``FunctionTool`` when it is passed in ``tools=[...]``.

Submodules:

* :mod:`app.agent.tools.triage_tools` -- five-level triage tools:
  :func:`ask_followup`, :func:`classify_triage`, :func:`get_department_advice`.
* :mod:`app.agent.tools.pii_tools` -- side-channel PII hand-off (the
  LLM never receives PII directly).
* :mod:`app.agent.tools.alert_tools` -- notify human staff via Slack.
"""

from app.agent.tools.alert_tools import dispatch_emergency
from app.agent.tools.pii_tools import trigger_pii_collection
from app.agent.tools.triage_tools import (
    ask_followup,
    classify_triage,
    get_department_advice,
)


def get_triage_tools() -> list:
    """Return the full list of triage-agent tool callables.

    Centralised so :mod:`app.agent.triage_agent` can register them with
    a single import. Order is informational only -- the LLM picks the
    tool by name.

    Server-side helpers (not exposed to the LLM) are intentionally NOT
    in this list:

    * :func:`app.agent.tools.pii_tools.acknowledge_pii_completion`
    * :func:`app.agent.tools.alert_tools.get_demo_alert_log`
    """

    return [
        ask_followup,
        classify_triage,
        get_department_advice,
        trigger_pii_collection,
        dispatch_emergency,
    ]


__all__ = [
    "ask_followup",
    "classify_triage",
    "dispatch_emergency",
    "get_department_advice",
    "get_triage_tools",
    "trigger_pii_collection",
]
