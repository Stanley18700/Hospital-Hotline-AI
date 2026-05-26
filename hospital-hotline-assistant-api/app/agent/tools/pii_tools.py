"""Secure PII collection signalling for the triage agent.

The LLM is strictly forbidden from collecting, repeating, storing, or
echoing patient PII (name, phone, address, national ID, passport
number, email). When the case is triaged as a Level 1 (Red) emergency
and staff need a way to dispatch resources, the agent calls
:func:`trigger_pii_collection` to *signal* the application to start a
secure, out-of-band collection flow.

How PII actually gets collected (no LLM involvement):

1. The agent calls :func:`trigger_pii_collection(session_id=...)`.
2. The tool returns a structured action -- no PII fields, just the
   declaration that the system should switch into PII-collection mode
   for that session.
3. The orchestrator/FastAPI handler sees ``action == "collect_pii"``,
   flips the session state to :attr:`TriageState.PII_COLLECT`, and
   instructs the frontend to render a **secure form** (or routes the
   call to a human staff member). The form posts directly to a
   dedicated backend endpoint that writes PII into the DB through an
   encrypted column / restricted role -- bypassing the agent and the
   model context entirely.
4. When PII collection completes, the backend calls
   :func:`acknowledge_pii_completion` (a server-side helper, NOT a tool
   exposed to the LLM) so the session can move on.

This module never accepts PII as a parameter, never logs raw PII, and
never returns PII. That is the whole point.

This tool is intended **only** for confirmed Level 1 emergencies. Do
not call it for Level 2-5 cases.
"""

from __future__ import annotations

from typing import Any, Final, Literal

from app.agent.triage_state import TriageState

# The fixed set of fields the secure form will collect. The model never
# sees the values; it only knows which fields the form will ask for. If
# the medical team approves additional fields later, edit this tuple --
# do not let the LLM influence it.
PII_FIELDS: Final[tuple[str, ...]] = ("name", "phone", "address")


def trigger_pii_collection(session_id: str) -> dict[str, Any]:
    """Signal the application to start secure PII collection for a session.

    **Use only for confirmed Level 1 (Red) emergencies** where staff
    must be dispatched and need patient identifying details. Do not
    call this for any other level.

    This tool does NOT collect, accept, or return PII. It only declares
    that the application should switch this session into a secure
    PII-collection workflow. The actual name / phone / address are
    collected by the frontend through a secure form (or by a human
    staff member on the phone) and stored by the backend through a
    dedicated, encrypted code path that the LLM never touches.

    :param session_id: The hotline session UUID (as a string). The
        orchestrator uses this to flip the session's state and to
        correlate the secure-form submission that follows.
    :returns: Stable structured signal::

            {
                "status": "ok",
                "action": "collect_pii",
                "fields": ["name", "phone", "address"],
                "session_id": "...",
                "next_state": "pii_collect",
                "patient_pii_included": false,
                "notes": "Secure form handled by frontend; LLM is not involved."
            }

        When ``session_id`` is empty or non-string, ``status`` is
        ``"error"`` and the action is not authorised. Even in the
        error case no PII is returned.
    """

    if not isinstance(session_id, str) or not session_id.strip():
        return {
            "status": "error",
            "action": "collect_pii",
            "error": "session_id must be a non-empty string",
            "fields": list(PII_FIELDS),
            "patient_pii_included": False,
        }

    return {
        "status": "ok",
        "action": "collect_pii",
        "fields": list(PII_FIELDS),
        "session_id": session_id.strip(),
        "next_state": TriageState.PII_COLLECT.value,
        "patient_pii_included": False,
        "notes": (
            "Secure form is handled by the frontend / backend, not the LLM. "
            "Do not include any patient PII in subsequent tool calls or replies."
        ),
    }


# ---------------------------------------------------------------------------
# Server-side helper: NOT registered as an ADK tool.
#
# Once the secure form (or staff member) finishes collecting PII for a Level 1
# session, the backend route that received it should call
# ``acknowledge_pii_completion`` so the orchestrator can move the session out
# of the PII_COLLECT state and resume (e.g. handing the patient over to staff
# or closing the call). DB persistence is intentionally out of scope here --
# this helper just produces the structured signal the orchestrator will act on.
# ---------------------------------------------------------------------------


PiiCollectionOutcome = Literal["completed", "declined", "timed_out", "failed"]


def acknowledge_pii_completion(
    session_id: str,
    *,
    outcome: PiiCollectionOutcome = "completed",
    note: str | None = None,
) -> dict[str, Any]:
    """Record that secure PII collection has finished for a session.

    Server-side only -- this is **not** exposed as an LLM tool. The
    backend secure-form endpoint (or the staff-handoff webhook) calls
    this helper after the actual PII has been stored through the
    dedicated, encrypted code path. The return value is a small
    structured signal the orchestrator can use to flip session state
    out of ``pii_collect``.

    No DB writes happen here yet -- that lands in a later migration
    step alongside the secure-form endpoint. Today this helper exists
    so the rest of the code can be wired against a stable signature.

    :param session_id: The hotline session UUID this acknowledgement
        applies to.
    :param outcome: How the collection ended. ``"completed"`` is the
        happy path; the others are for accurate auditing.
    :param note: Optional non-identifying note (for example,
        ``"handed off to staff in person"``). Must NOT contain PII --
        the caller is responsible for keeping it clean.
    :returns: Structured ack signal with the next session state. PII is
        never returned.
    """

    if not isinstance(session_id, str) or not session_id.strip():
        return {
            "status": "error",
            "action": "pii_collection_acknowledged",
            "error": "session_id must be a non-empty string",
            "patient_pii_included": False,
        }

    next_state = (
        TriageState.DONE.value if outcome == "completed" else TriageState.TRIAGE.value
    )

    payload: dict[str, Any] = {
        "status": "ok",
        "action": "pii_collection_acknowledged",
        "session_id": session_id.strip(),
        "outcome": outcome,
        "next_state": next_state,
        "patient_pii_included": False,
    }
    if note:
        payload["note"] = note.strip()
    return payload


__all__ = [
    "PII_FIELDS",
    "PiiCollectionOutcome",
    "acknowledge_pii_completion",
    "trigger_pii_collection",
]
