"""Voice-channel PII guard helpers and FUTURE-WORK placeholders.

This module exists for two reasons:

1.  **Centralised guard rule.** :func:`voice_input_is_allowed` is the
    single source of truth for "may we transcribe / accept audio for
    this session right now?". The ``/stt`` endpoint consults it; any
    future voice-direct chat endpoint should do the same. Keeping the
    rule in one place means there is no risk of the audio surface
    drifting away from the text surface's phase-guard semantics.

2.  **FUTURE WORK marker for voice PII capture.** We deliberately do
    NOT implement field-by-field voice capture of patient name /
    phone / address in this build. The reasons are documented below.
    The placeholder class :class:`VoicePiiFieldCapture` is the
    integration shape we would adopt if/when we revisit this; it
    raises :class:`NotImplementedError` so any accidental wiring is
    loud rather than silent.

Why we are NOT doing voice PII capture today
--------------------------------------------

*   **Transcript accuracy on names, phone numbers, and street
    addresses is poor.** Cloud STT is tuned for free-flowing
    conversational language. Single proper nouns and digit strings
    drop in accuracy fast, especially with non-Thai accents in our
    target audience. A misheard phone number on an ambulance dispatch
    is a safety failure.

*   **The secure form already covers the use case.** Once the agent
    confirms Level 1 and the session transitions to
    :attr:`app.agent.triage_state.TriageState.PII_COLLECT`, the
    frontend renders the secure PII form and the patient (or a
    bystander) types the values. This is the recommended path.

*   **Voice PII would broaden the model exposure surface.** Even if
    we kept the transcript backend-only, the audio still passes
    through Cloud STT, which is an external dependency. Keeping
    patient identifiers off that path keeps the threat model small.

If we ever revisit this, the placeholder below shows the integration
shape: a session-scoped capture state machine, one method per field,
each method backed by its own narrow recogniser (e.g. digit-only STT
for phone, alpha-grammar STT for names). It would emit the same
:class:`app.services.pii_handoff.EmergencyPiiEvent` so the rest of
the pipeline (Slack dispatch, redacted receipt, session-phase write)
is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.agent.triage_state import TriageState


# ---------------------------------------------------------------------------
# Guard rule (used today)
# ---------------------------------------------------------------------------


def voice_input_is_allowed(phase: TriageState) -> bool:
    """True iff audio input may be transcribed / forwarded for this phase.

    The rule mirrors the text-chat phase guard exactly:

    * :attr:`TriageState.TRIAGE` -- allowed (normal triage gathering).
    * :attr:`TriageState.PII_COLLECT` -- NOT allowed. The patient
      should use the secure PII form, not the microphone, while a
      Level 1 case is being handed off. This keeps any
      patient-spoken identifiers out of Cloud STT entirely.
    * :attr:`TriageState.DONE` -- allowed. Re-engagement after a
      completed session is treated as a fresh triage interaction.
    """

    return phase is not TriageState.PII_COLLECT


@dataclass(frozen=True)
class VoiceGuardDecision:
    """Structured outcome the FastAPI handler returns to the caller.

    ``allowed=True`` means the handler should proceed with STT.
    ``allowed=False`` means the handler should short-circuit with a
    409 and surface ``next_action`` + ``reason`` to the frontend so
    it can switch UI mode (e.g. show the secure form).
    """

    allowed: bool
    next_action: Literal["transcribe", "collect_pii"]
    reason: str


def evaluate_voice_guard(phase: TriageState) -> VoiceGuardDecision:
    """Wrap :func:`voice_input_is_allowed` in a structured outcome."""

    if voice_input_is_allowed(phase):
        return VoiceGuardDecision(
            allowed=True,
            next_action="transcribe",
            reason="ok",
        )
    return VoiceGuardDecision(
        allowed=False,
        next_action="collect_pii",
        reason="pii_collection_active",
    )


# ---------------------------------------------------------------------------
# FUTURE WORK: voice PII field-by-field capture
# ---------------------------------------------------------------------------
#
# Do NOT enable this in production. The class signature below is the
# planned integration shape and is intentionally non-functional so any
# accidental wiring fails loudly at import / call time.


_VoicePiiField = Literal["name", "phone", "address", "notes"]


class VoicePiiFieldCapture:
    """**FUTURE WORK** -- session-scoped voice capture of PII fields.

    Planned integration shape (not implemented):

    1.  ``begin(session_id)`` -- start the capture state machine.
    2.  ``capture_field(field, audio_bytes, language)`` -- run a
        narrow per-field recogniser (e.g. digit-only grammar for
        ``phone``, alpha grammar for ``name``).
    3.  ``finalize()`` -- emit an :class:`app.services.pii_handoff
        .EmergencyPiiEvent` so the existing dispatch + receipt
        pipeline stays unchanged.

    See the module docstring for why this is deferred. Today the
    frontend renders the secure form instead, which is more accurate
    and reduces the model / external-dependency surface.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "Voice PII field-by-field capture is intentionally not "
            "implemented. Use the secure form via "
            "POST /sessions/{id}/emergency-pii instead. See "
            "app/services/voice_pii.py for the deferral rationale."
        )

    async def capture_field(
        self,
        *,
        field: _VoicePiiField,
        audio_bytes: bytes,
        language: str,
    ) -> str:
        raise NotImplementedError(
            "Voice PII capture is deferred. See module docstring."
        )


__all__ = [
    "VoiceGuardDecision",
    "VoicePiiFieldCapture",
    "evaluate_voice_guard",
    "voice_input_is_allowed",
]
