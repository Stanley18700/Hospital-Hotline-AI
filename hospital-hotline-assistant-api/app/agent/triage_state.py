"""Session-level state machine for the triage agent.

The hotline conversation has three coarse phases that the rest of the
backend (FastAPI handler, DB writes, alert routing) needs to reason about:

* :attr:`TriageState.TRIAGE` -- the agent is gathering symptoms and
  assigning a triage level (the default during the call).
* :attr:`TriageState.PII_COLLECT` -- a Level 1 (Red) emergency was
  detected and the system is in the secure PII hand-off window. The
  LLM never sees PII; collection happens out-of-band via a human or a
  dedicated form. The agent stays paused here.
* :attr:`TriageState.DONE` -- triage is complete (routed to a
  department, handed off to staff, or the session was explicitly ended).

Keep this enum tiny on purpose: any branching beyond these three phases
belongs in a downstream service, not in the conversation state.

Storage model
-------------

We piggy-back on the existing ``sessions.metadata`` JSONB column rather
than introducing a new column or table. The phase lives at
``metadata.triage_phase`` and an audit timestamp lives at
``metadata.triage_phase_updated_at``. This keeps the change additive --
old rows that pre-date this feature simply read back as
:attr:`TriageState.TRIAGE` (the default) without any migration.

The helpers below are intentionally pure (no DB I/O) so they can be
called from the FastAPI handler, unit tests, or admin tooling without
coupling to ``asyncpg``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


SESSION_PHASE_METADATA_KEY: str = "triage_phase"
"""Metadata key used to persist the current :class:`TriageState`."""

PHASE_UPDATED_AT_METADATA_KEY: str = "triage_phase_updated_at"
"""Metadata key used to persist the audit timestamp of the last phase change."""


class TriageState(StrEnum):
    """High-level conversation phase for a hotline session.

    Using :class:`enum.StrEnum` so values serialise cleanly to JSON / DB
    (``"triage"``, ``"pii_collect"``, ``"done"``) without custom encoders.
    """

    TRIAGE = "triage"
    PII_COLLECT = "pii_collect"
    DONE = "done"


def get_session_phase(metadata: dict[str, Any] | None) -> TriageState:
    """Read the current :class:`TriageState` out of a session metadata dict.

    Old rows that pre-date this feature (no ``triage_phase`` key)
    default to :attr:`TriageState.TRIAGE`. Unknown or malformed values
    also fall back to :attr:`TriageState.TRIAGE` so we fail open --
    callers can safely treat the result as authoritative.
    """

    if not metadata:
        return TriageState.TRIAGE
    raw = metadata.get(SESSION_PHASE_METADATA_KEY)
    if not isinstance(raw, str):
        return TriageState.TRIAGE
    try:
        return TriageState(raw)
    except ValueError:
        return TriageState.TRIAGE


def with_session_phase(
    metadata: dict[str, Any] | None,
    phase: TriageState,
    *,
    now: str | None = None,
) -> dict[str, Any]:
    """Return a *new* metadata dict with the phase + audit timestamp set.

    The caller is responsible for the ``UPDATE sessions SET metadata =
    ...`` write. We never mutate the input dict so callers can keep a
    reference to the previous state if they need to compare or roll
    back.

    ``now`` is overridable so tests can pin a deterministic timestamp;
    in production we use the current UTC time in ISO-8601.
    """

    base: dict[str, Any] = dict(metadata or {})
    base[SESSION_PHASE_METADATA_KEY] = phase.value
    base[PHASE_UPDATED_AT_METADATA_KEY] = now or datetime.now(timezone.utc).isoformat()
    return base


def is_in_pii_collection(metadata: dict[str, Any] | None) -> bool:
    """True iff the session is currently in :attr:`TriageState.PII_COLLECT`."""

    return get_session_phase(metadata) is TriageState.PII_COLLECT


__all__ = [
    "PHASE_UPDATED_AT_METADATA_KEY",
    "SESSION_PHASE_METADATA_KEY",
    "TriageState",
    "get_session_phase",
    "is_in_pii_collection",
    "with_session_phase",
]
