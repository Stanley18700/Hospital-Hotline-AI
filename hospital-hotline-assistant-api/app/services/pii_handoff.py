"""Secure emergency-PII hand-off pipeline.

This module owns the data path that the patient's name, phone, and
address flow through *after* arriving at the dedicated
``POST /sessions/{session_id}/emergency-pii`` endpoint.

Storage / dispatch summary
--------------------------

After successful submission, exactly two things happen with the PII:

1.  **It is shipped to a human-notifying sink.**
    In the current build that is :class:`SlackPiiHandoffSink`, which
    wraps :class:`app.services.slack_notifier.SlackNotifier`. The
    Slack channel is the production hand-off to the human dispatcher
    (ambulance / admin desk). When ``SLACK_WEBHOOK_URL`` is unset the
    Slack sink reports ``notification_sent=False`` and the demo
    falls back to the in-memory storage sink only.

2.  **It is kept in process memory for the demo dashboard.**
    :class:`InMemoryPiiHandoffSink` appends the event to a module-
    local list so the admin demo can render the case timeline. This
    is the **TEMPORARY** persistence layer; production deployments
    swap it for a ``PostgresPiiHandoffSink`` that writes to an
    encrypted ``emergency_pii`` table (see ``ADK_MIGRATION_PLAN.md``).

Both sinks run in parallel via :class:`CompositePiiHandoffSink`, and
the merged :class:`PiiHandoffOutcome` is what the FastAPI endpoint
uses to populate the response's ``alert_sent`` flag.

Critical privacy rules enforced here
------------------------------------

1.  **PII never reaches the LLM.**
    The ADK runner (``app.agent.triage_runner``) is NOT imported in
    this file and is NOT called from any function below. The endpoint
    that consumes this module also does not invoke the runner.

2.  **PII does not reach Postgres in the current demo build.**
    Raw name / phone / address are handed to the sinks and then drop
    out of scope when the request returns. The only thing written to
    the database is a redacted :class:`EmergencyPiiReceipt` --
    ``case_id`` + populated field *names* + timestamp. The receipt is
    safe to store in the existing ``sessions.metadata`` JSONB column
    without a schema change.

3.  **Logs are redacted.**
    :class:`InMemoryPiiHandoffSink` logs only the field *names* that
    were populated; field *values* never appear in a logger call.
    :class:`SlackPiiHandoffSink` does forward PII to Slack because
    that is its dispatch channel, but it logs only the case_id.

Production migration
--------------------

The :class:`PiiHandoffSink` protocol is the seam for replacing the
demo placeholder. The endpoint contract does not change -- only the
concrete sink wired into ``app.state.pii_sink`` does.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from app.services.slack_notifier import SlackNotifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmergencyPiiPayload:
    """The raw secure-form payload as accepted by the API.

    Held only for the duration of a single ``submit_emergency_pii``
    call; handed to the sink as part of an :class:`EmergencyPiiEvent`
    and then dropped from the function's scope when control returns
    to the FastAPI runtime.

    The dataclass is frozen so handlers cannot accidentally mutate or
    re-bind the fields before passing them on.
    """

    name: str
    phone: str
    address: str
    notes: str | None = None


@dataclass(frozen=True)
class EmergencyPiiEvent:
    """Event posted to the notification sink.

    This is the on-the-wire shape a production dispatcher (paging /
    ambulance / control desk) consumes. It carries the PII because
    that is the dispatcher's job, but it is built once, handed to
    exactly one :class:`PiiHandoffSink` instance, and never written
    to a logger / database / response body in this module.
    """

    case_id: str
    session_id: str
    language: str
    triage_level: int | None
    triage_color: str | None
    symptoms_summary: str | None
    received_at: datetime
    payload: EmergencyPiiPayload


@dataclass(frozen=True)
class EmergencyPiiReceipt:
    """Audit-safe summary safe to persist in Postgres / show in admin UI.

    Strictly redacted: only the names of populated fields plus the
    case identifier and the receipt timestamp. Suitable as a value
    for ``sessions.metadata`` and as a return shape from the API.
    """

    case_id: str
    session_id: str
    received_at: datetime
    received_fields: list[str]
    notification_dispatched: bool


# ---------------------------------------------------------------------------
# Sink contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PiiHandoffOutcome:
    """Structured result of one PII hand-off across one or more sinks.

    * ``notification_sent`` -- ``True`` iff a sink that notifies a
      human (Slack / pager / SMS) reported success. This is the
      boolean the API surfaces back to the patient as ``alert_sent``.
    * ``persisted`` -- ``True`` iff a storage sink (in-memory demo or
      future Postgres) reported success. Useful for the admin
      dashboard / audit trail.
    * ``sink_results`` -- per-sink success map keyed by the sink's
      ``name`` attribute. Surfaced in logs and the admin dashboard.
    """

    notification_sent: bool
    persisted: bool
    sink_results: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "PiiHandoffOutcome":
        return cls(notification_sent=False, persisted=False, sink_results={})

    def merge(self, other: "PiiHandoffOutcome") -> "PiiHandoffOutcome":
        merged = dict(self.sink_results)
        merged.update(other.sink_results)
        return PiiHandoffOutcome(
            notification_sent=self.notification_sent or other.notification_sent,
            persisted=self.persisted or other.persisted,
            sink_results=merged,
        )


@runtime_checkable
class PiiHandoffSink(Protocol):
    """Interface every PII destination must implement.

    Implementations:

    * :class:`InMemoryPiiHandoffSink` -- TEMPORARY demo storage.
    * :class:`SlackPiiHandoffSink` -- wraps :class:`SlackNotifier` to
      ping the hospital's Slack channel with the dispatch payload.
    * Future ``PostgresPiiHandoffSink`` -- writes the encrypted PII
      payload into a dedicated table (planned).
    * Future ``PagerDutyPiiHandoffSink`` / SMS / fax integrations.

    Contract: :meth:`dispatch` must NEVER raise. Each sink reports
    its own outcome via :class:`PiiHandoffOutcome`. The endpoint
    keeps the case_id valid either way so the patient is never told
    "please try again" because of a backend hiccup.
    """

    name: str

    async def dispatch(self, event: EmergencyPiiEvent) -> PiiHandoffOutcome: ...


# ---------------------------------------------------------------------------
# Temporary in-memory placeholder (storage role)
# ---------------------------------------------------------------------------


class InMemoryPiiHandoffSink:
    """**TEMPORARY** demo placeholder for the secure PII channel.

    DO NOT USE IN PRODUCTION. Stores events in a process-local list so
    the demo admin dashboard can render the case timeline. The raw
    values live only in this Python process's memory and are not
    persisted across restarts.

    Production replacement plan (see ``ADK_MIGRATION_PLAN.md``):

    * Provision an encrypted ``emergency_pii`` table (column-level
      encryption + row-level RBAC).
    * Implement ``PostgresPiiHandoffSink`` writing into that table
      inside the same transaction that ``_set_session_phase`` opens.
    * Swap the sink at ``app.state.pii_sink`` -- the endpoint code
      does not change.
    """

    name: str = "in_memory_storage"

    def __init__(self) -> None:
        self._events: list[EmergencyPiiEvent] = []
        self._lock = asyncio.Lock()

    async def dispatch(self, event: EmergencyPiiEvent) -> PiiHandoffOutcome:
        try:
            async with self._lock:
                self._events.append(event)
        except Exception:  # noqa: BLE001 - the contract forbids raising
            logger.exception(
                "emergency-pii in-memory dispatch failed unexpectedly "
                "case_id=%s",
                event.case_id,
            )
            return PiiHandoffOutcome(
                notification_sent=False,
                persisted=False,
                sink_results={self.name: False},
            )

        # IMPORTANT: log only field *names*, never field *values*.
        populated_fields = [
            field_name
            for field_name, value in (
                ("name", event.payload.name),
                ("phone", event.payload.phone),
                ("address", event.payload.address),
                ("notes", event.payload.notes),
            )
            if value
        ]
        logger.info(
            "emergency-pii stored in_memory case_id=%s session_id=%s "
            "triage_level=%s fields=%s",
            event.case_id,
            event.session_id,
            event.triage_level,
            populated_fields,
        )
        return PiiHandoffOutcome(
            notification_sent=False,
            persisted=True,
            sink_results={self.name: True},
        )

    def snapshot(self) -> list[EmergencyPiiEvent]:
        """Defensive copy for the demo admin dashboard.

        The production sink will NOT expose this -- the dashboard will
        read a redacted projection from Postgres instead.
        """

        return list(self._events)

    def clear(self) -> None:
        """Drop all stored events (intended for tests / demo resets)."""

        self._events.clear()


# ---------------------------------------------------------------------------
# Slack dispatch (notification role)
# ---------------------------------------------------------------------------


class SlackPiiHandoffSink:
    """Notification sink that forwards the dispatch payload to Slack.

    Wraps :class:`SlackNotifier.send_emergency_dispatch` so the Slack
    payload format lives in one place. PII flows from this sink
    straight into the Slack webhook -- that is its purpose. The LLM
    is not on this path.

    If ``SLACK_WEBHOOK_URL`` is unset, the notifier returns ``False``
    without making an HTTP call, and this sink simply reports
    ``notification_sent=False``. The endpoint still completes
    successfully so the patient is never blocked by a missing config.
    """

    name: str = "slack"

    def __init__(self, notifier: SlackNotifier | None = None) -> None:
        self._notifier = notifier or SlackNotifier()

    async def dispatch(self, event: EmergencyPiiEvent) -> PiiHandoffOutcome:
        try:
            sent = await self._notifier.send_emergency_dispatch(
                case_id=event.case_id,
                session_id=event.session_id,
                language=event.language,
                triage_level=event.triage_level,
                triage_color=event.triage_color,
                symptoms_summary=event.symptoms_summary,
                patient_name=event.payload.name,
                patient_phone=event.payload.phone,
                patient_address=event.payload.address,
                patient_notes=event.payload.notes,
            )
        except Exception:  # noqa: BLE001 - the contract forbids raising
            logger.exception(
                "slack dispatch raised unexpectedly case_id=%s", event.case_id
            )
            return PiiHandoffOutcome(
                notification_sent=False,
                persisted=False,
                sink_results={self.name: False},
            )

        # Log only the case_id outcome -- never the PII.
        logger.info(
            "emergency-pii slack dispatch case_id=%s sent=%s",
            event.case_id,
            sent,
        )
        return PiiHandoffOutcome(
            notification_sent=bool(sent),
            persisted=False,
            sink_results={self.name: bool(sent)},
        )


# ---------------------------------------------------------------------------
# Fan-out composite
# ---------------------------------------------------------------------------


class CompositePiiHandoffSink:
    """Run multiple sinks in parallel and merge their outcomes.

    This is the sink the FastAPI app actually binds at startup. It
    fans the event out to every registered sink concurrently (so a
    slow Slack webhook never blocks the in-memory log) and merges
    their per-sink :class:`PiiHandoffOutcome` results.

    Adding a new destination (Postgres, PagerDuty, SMS) is one line in
    :func:`get_default_pii_handoff_sink`.
    """

    name: str = "composite"

    def __init__(self, sinks: list[PiiHandoffSink]) -> None:
        self._sinks = list(sinks)

    @property
    def sinks(self) -> list[PiiHandoffSink]:
        return list(self._sinks)

    async def dispatch(self, event: EmergencyPiiEvent) -> PiiHandoffOutcome:
        if not self._sinks:
            return PiiHandoffOutcome.empty()

        results = await asyncio.gather(
            *(sink.dispatch(event) for sink in self._sinks),
            return_exceptions=True,
        )

        merged = PiiHandoffOutcome.empty()
        for sink, result in zip(self._sinks, results, strict=True):
            if isinstance(result, BaseException):
                # A sink violated the no-raise contract. Log + count it
                # as a failure, but keep merging the others.
                logger.exception(
                    "pii sink %s violated no-raise contract case_id=%s",
                    sink.name,
                    event.case_id,
                    exc_info=result,
                )
                merged = merged.merge(
                    PiiHandoffOutcome(
                        notification_sent=False,
                        persisted=False,
                        sink_results={sink.name: False},
                    )
                )
                continue
            merged = merged.merge(result)
        return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_case_id(*, now: datetime | None = None) -> str:
    """Generate a short, human-readable emergency case identifier.

    Format: ``EM-YYYYMMDD-XXXXXXXX`` where ``XXXXXXXX`` is 8 lowercase
    hex characters from :func:`secrets.token_hex`. Cryptographically
    random so the identifier is unguessable, but compact enough to
    print on a clipboard ticket or read aloud over a phone bridge.
    """

    now = now or datetime.now(timezone.utc)
    return f"EM-{now.strftime('%Y%m%d')}-{secrets.token_hex(4)}"


def build_redacted_receipt_metadata(
    receipt: EmergencyPiiReceipt,
    outcome: PiiHandoffOutcome | None = None,
) -> dict[str, Any]:
    """Build the ``sessions.metadata`` patch for a hand-off receipt.

    Returns a plain dict containing only redacted fields: case id,
    timestamp, populated field *names*, and per-sink dispatch outcome.
    The caller merges this into the existing session metadata.
    """

    patch: dict[str, Any] = {
        "pii_case_id": receipt.case_id,
        "pii_received_at": receipt.received_at.isoformat(),
        "pii_received_fields": list(receipt.received_fields),
        "pii_notification_dispatched": receipt.notification_dispatched,
    }
    if outcome is not None:
        patch["pii_sink_results"] = dict(outcome.sink_results)
        patch["pii_persisted"] = outcome.persisted
    return patch


def next_instruction_for_patient(
    *,
    language: str,
    case_id: str,
    alert_sent: bool,
) -> str:
    """Compose the patient-facing instruction returned by the API.

    The text is hard-coded per language and parameterised only on
    ``case_id`` + the boolean ``alert_sent`` -- it cannot leak PII and
    cannot be influenced by anything the model produces.
    """

    if language == "th":
        if alert_sent:
            return (
                f"ทีมฉุกเฉินได้รับแจ้งและกำลังเดินทางไปยังที่อยู่ของคุณ "
                f"หมายเลขกรณีของคุณคือ {case_id} "
                "กรุณาอยู่บนสายและเปิดโทรศัพท์เพื่อให้เจ้าหน้าที่ติดต่อคุณได้"
            )
        return (
            f"ข้อมูลที่ปลอดภัยได้รับเรียบร้อย หมายเลขกรณีของคุณคือ {case_id} "
            "เจ้าหน้าที่จะติดต่อกลับโดยเร็วที่สุด กรุณาอยู่บนสาย"
        )

    if alert_sent:
        return (
            f"The emergency team has been notified and is on the way. "
            f"Your case ID is {case_id}. Please stay on the line and keep "
            "your phone available so responders can reach you."
        )
    return (
        f"Your secure information has been received. Your case ID is "
        f"{case_id}. A staff member will reach out to you as soon as "
        "possible -- please stay on the line."
    )


# ---------------------------------------------------------------------------
# Process-wide singleton (lazy)
# ---------------------------------------------------------------------------


_DEFAULT_SINK: CompositePiiHandoffSink | None = None
_DEFAULT_IN_MEMORY: InMemoryPiiHandoffSink | None = None


def _build_default_composite() -> CompositePiiHandoffSink:
    """Construct the production-shaped composite sink.

    Fans out to:

    * :class:`InMemoryPiiHandoffSink` -- demo storage / admin
      timeline. **TEMPORARY**; swap for Postgres in production.
    * :class:`SlackPiiHandoffSink` -- notification dispatch. Auto-
      disables when ``SLACK_WEBHOOK_URL`` is unset.

    Returning a composite (instead of a single sink) makes adding a
    third channel -- e.g. ``PostgresPiiHandoffSink`` -- a one-line
    change here.
    """

    global _DEFAULT_IN_MEMORY
    if _DEFAULT_IN_MEMORY is None:
        _DEFAULT_IN_MEMORY = InMemoryPiiHandoffSink()
    return CompositePiiHandoffSink(
        sinks=[
            _DEFAULT_IN_MEMORY,
            SlackPiiHandoffSink(),
        ]
    )


def get_default_pii_handoff_sink() -> CompositePiiHandoffSink:
    """Return the lazily-initialised process-wide composite sink.

    The FastAPI lifespan binds this onto ``app.state.pii_sink``. Tests
    can substitute their own :class:`PiiHandoffSink` by re-assigning
    ``app.state.pii_sink``; no global state is implicitly threaded
    through the request handler.
    """

    global _DEFAULT_SINK
    if _DEFAULT_SINK is None:
        _DEFAULT_SINK = _build_default_composite()
    return _DEFAULT_SINK


def get_default_in_memory_sink() -> InMemoryPiiHandoffSink:
    """Return the storage sink the composite uses.

    Useful for the admin dashboard endpoint that wants to read the
    demo case timeline. Production will replace this with a Postgres
    query.
    """

    get_default_pii_handoff_sink()  # ensure construction
    assert _DEFAULT_IN_MEMORY is not None
    return _DEFAULT_IN_MEMORY


def reset_default_pii_handoff_sink() -> None:
    """Reset the singletons. Intended for tests / demo resets."""

    global _DEFAULT_SINK, _DEFAULT_IN_MEMORY
    _DEFAULT_SINK = None
    _DEFAULT_IN_MEMORY = None


__all__ = [
    "CompositePiiHandoffSink",
    "EmergencyPiiEvent",
    "EmergencyPiiPayload",
    "EmergencyPiiReceipt",
    "InMemoryPiiHandoffSink",
    "PiiHandoffOutcome",
    "PiiHandoffSink",
    "SlackPiiHandoffSink",
    "build_redacted_receipt_metadata",
    "generate_case_id",
    "get_default_in_memory_sink",
    "get_default_pii_handoff_sink",
    "next_instruction_for_patient",
    "reset_default_pii_handoff_sink",
]
