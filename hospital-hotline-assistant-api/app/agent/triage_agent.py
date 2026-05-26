"""Triage agent construction (Google ADK).

This module owns ONLY agent construction. It does not open HTTP
connections, websocket sessions, or touch FastAPI. The
:class:`google.adk.runners.Runner` that drives this agent lives in
:mod:`app.agent.triage_runner`.

Topology:

* :class:`google.adk.agents.LlmAgent` -- the reasoning agent
  ("triage_reasoner"). Owns the five-level system prompt, the five
  triage tools, plus the built-in :func:`google.adk.tools.exit_loop`
  tool so the model can cleanly end an iteration.

* :class:`google.adk.agents.LoopAgent` -- a thin orchestration wrapper
  ("hotline_triage_agent"). Re-runs the reasoner up to
  ``settings.adk_max_tool_iterations`` times per HTTP turn or until
  the reasoner calls ``exit_loop``. The inner LlmAgent already does
  its own tool-calling micro-loop within a single run, so the outer
  LoopAgent is primarily a safety bound on worst-case work per turn.

Model selection is driven by :mod:`app.config` (``google_model_name``,
default ``gemini-2.5-flash``) so the deploy environment, not this
file, picks the model.

ADK imports are deferred inside the builder functions so this module
can be imported in environments where ``google-adk`` is not yet
installed (CI, unit tests, lint runs).
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any

from app.agent.prompts import build_triage_system_prompt
from app.agent.tools import (
    ask_followup,
    classify_triage,
    dispatch_emergency,
    get_department_advice,
    trigger_pii_collection,
)
from app.config import settings

logger = logging.getLogger(__name__)

AGENT_NAME: str = "hotline_triage_agent"
AGENT_NAME_REASONER: str = "triage_reasoner"
AGENT_DESCRIPTION: str = (
    "Hospital hotline triage assistant that classifies patient symptoms "
    "against the five-level ER triage ladder and escalates to human staff "
    "without exposing PII to the LLM."
)

# Default lower bound -- never go below 1, never above this ceiling, even
# if config is misconfigured.
_MIN_LOOP_ITERATIONS: int = 1
_MAX_LOOP_ITERATIONS_HARD_CEILING: int = 10

# Appended to the canonical triage system prompt so the model knows about
# the LoopAgent's exit_loop semantics. Lives here -- not in
# :mod:`app.agent.prompts.triage_system` -- because it is an orchestration
# detail, not a medical-conversation rule.
_LOOP_CONTROL_ADDENDUM = """
[LOOP CONTROL]
This conversation runs inside a bounded loop orchestrated by the
server. Once you have delivered your final user-facing reply for the
current turn -- whether that was a follow-up question, the
classification + department advice + emergency dispatch chain, or the
Level 1 PII handoff -- call the `exit_loop` tool (no arguments) to
end the iteration cleanly. Do not call `exit_loop` before producing
the patient-facing reply for the current turn.
""".strip()


def resolve_model_name() -> str:
    """Return the Vertex Gemini model name to use for the reasoner.

    Centralised so model selection stays a one-line config change.
    Defaults to a low-latency Gemini Flash model suited to back-and-forth
    voice triage; the deploy environment can override via
    ``GOOGLE_MODEL_NAME``.
    """

    name = (settings.google_model_name or "").strip()
    return name or "gemini-2.5-flash"


def resolve_loop_iterations() -> int:
    """Return the LoopAgent ``max_iterations`` value, clamped to a safe range."""

    requested = int(settings.adk_max_tool_iterations or _MIN_LOOP_ITERATIONS)
    return max(_MIN_LOOP_ITERATIONS, min(requested, _MAX_LOOP_ITERATIONS_HARD_CEILING))


def _configure_vertex_environment() -> None:
    """Force the ADK / google-genai runtime onto the Vertex AI backend.

    ADK's underlying ``google.genai.Client(...)`` defaults to the
    public Gemini API backend (which expects ``GOOGLE_API_KEY``).
    For this project we authenticate through a Vertex AI service
    account, so we set the trio of env vars Google's SDK looks at:

    * ``GOOGLE_GENAI_USE_VERTEXAI=true`` -- switches client mode
    * ``GOOGLE_CLOUD_PROJECT`` -- which project to bill / run in
    * ``GOOGLE_CLOUD_LOCATION`` -- which region to call

    We also resolve a relative ``GOOGLE_APPLICATION_CREDENTIALS``
    path to absolute the same way :mod:`app.services.google_stt`
    does, so launching uvicorn from a different cwd still works.

    Called from the reasoner builder so the env is set BEFORE the
    ``LlmAgent`` constructs its internal ``Client``. We use
    ``setdefault`` so an operator who has already exported these
    vars wins.
    """

    if settings.google_cloud_project:
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", settings.google_cloud_project)
        os.environ.setdefault(
            "GOOGLE_CLOUD_LOCATION", settings.google_cloud_location or "us-central1"
        )

    if settings.google_application_credentials:
        cred_path = settings.google_application_credentials
        if not pathlib.Path(cred_path).is_absolute():
            cred_path = str((pathlib.Path.cwd() / cred_path).resolve())
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", cred_path)

    logger.debug(
        "adk vertex env configured project=%s location=%s use_vertexai=%s",
        os.environ.get("GOOGLE_CLOUD_PROJECT"),
        os.environ.get("GOOGLE_CLOUD_LOCATION"),
        os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"),
    )


def _domain_tools() -> list:
    """The five triage-domain tools registered with the reasoner."""

    return [
        ask_followup,
        classify_triage,
        get_department_advice,
        trigger_pii_collection,
        dispatch_emergency,
    ]


def build_triage_reasoner(*, input_mode: str = "text") -> Any:
    """Construct the inner LlmAgent (the reasoner).

    Exposed separately from :func:`build_triage_agent` so unit tests
    and evals can exercise the LLM without the LoopAgent wrapper.
    """

    # Must run BEFORE the ADK imports below trigger client construction
    # (some ADK builds eagerly probe credentials at import-adjacent
    # time). Idempotent and cheap, so we always call it.
    _configure_vertex_environment()

    try:
        from google.adk.agents import LlmAgent
        from google.adk.tools import exit_loop
    except ImportError as exc:  # pragma: no cover - exercised in install-less envs
        raise RuntimeError(
            "google-adk is not installed. Run `pip install -e .` or "
            "`uv sync` inside hospital-hotline-assistant-api/."
        ) from exc

    instruction = (
        build_triage_system_prompt(input_mode=input_mode).rstrip()
        + "\n\n"
        + _LOOP_CONTROL_ADDENDUM
        + "\n"
    )

    return LlmAgent(
        name=AGENT_NAME_REASONER,
        description=AGENT_DESCRIPTION,
        model=resolve_model_name(),
        instruction=instruction,
        tools=[*_domain_tools(), exit_loop],
    )


def build_triage_agent(*, input_mode: str = "text") -> Any:
    """Construct the public triage agent (LoopAgent wrapping the reasoner).

    :param input_mode: ``"voice"`` to apply the voice-mode prompt
        addendum (terser, single-sentence replies); anything else uses
        the default text-mode prompt.
    :returns: A :class:`google.adk.agents.LoopAgent` ready to hand to
        :class:`google.adk.runners.Runner`. Typed as :class:`Any` so
        this module imports cleanly in environments without
        ``google-adk`` installed.
    :raises RuntimeError: if ``google-adk`` is not importable.
    """

    try:
        from google.adk.agents import LoopAgent
    except ImportError as exc:  # pragma: no cover - exercised in install-less envs
        raise RuntimeError(
            "google-adk is not installed. Run `pip install -e .` or "
            "`uv sync` inside hospital-hotline-assistant-api/."
        ) from exc

    reasoner = build_triage_reasoner(input_mode=input_mode)
    return LoopAgent(
        name=AGENT_NAME,
        description=AGENT_DESCRIPTION,
        sub_agents=[reasoner],
        max_iterations=resolve_loop_iterations(),
    )


__all__ = [
    "AGENT_DESCRIPTION",
    "AGENT_NAME",
    "AGENT_NAME_REASONER",
    "build_triage_agent",
    "build_triage_reasoner",
    "resolve_loop_iterations",
    "resolve_model_name",
]
