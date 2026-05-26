"""Emergency alert dispatch tool for the triage agent.

When the agent classifies a case as Level 1 (Red) or Level 2 (Orange)
it calls :func:`dispatch_emergency`, which:

1. Builds a small :class:`EmergencyAlert` record with the case + triage
   details. The record contains NO patient PII -- only the
   non-identifying triage summary the agent produced.
2. Always appends that record to :data:`DEMO_ALERT_LOG`, an in-memory
   list that powers the demo admin dashboard. This is the demo-time
   stand-in for an `emergency_events` row -- a later migration step
   will also persist to Postgres through the existing
   :mod:`app.services.triage_service` path.
3. Attempts a Slack webhook delivery via the existing
   :class:`app.services.slack_notifier.SlackNotifier` when
   ``settings.slack_webhook_url`` is set. The Slack payload reuses the
   same builder/format that ``SlackNotifier.send_alert`` already
   produces so an admin watching the Slack channel sees one consistent
   shape regardless of which code path triggered the alert.
4. If Slack is not configured (typical for the local demo), gracefully
   falls back: the alert still lands in :data:`DEMO_ALERT_LOG` and the
   tool returns ``status="demo_pending"`` instead of raising. The
   conversation continues uninterrupted.
5. If Slack raises or returns a non-2xx, the exception is caught, the
   alert is still recorded in the in-memory log, and the tool returns
   ``status="failed"`` with a short error description -- never
   throwing back into the ADK runtime.

Demo admin dashboard hook (referenced by a later step):

    GET /admin/emergency-alerts
    -> returns ``DEMO_ALERT_LOG`` as JSON for the demo UI.

That endpoint lives in :mod:`app.main` (added in Prompt 13). This file
only owns the in-memory data source.

Design notes / constraints:

* No PII fields anywhere in this module. The tool's input does not
  accept name / phone / address, and ``symptoms_summary`` is expected
  to be PII-redacted upstream by the orchestrator. The Slack payload
  includes ONLY: ``case_id``, ``session_id``, ``triage_level``,
  ``triage_color``, ``symptoms_summary``, and the dispatch timestamp.
* No new env vars. Slack reuses ``settings.slack_webhook_url`` from
  :mod:`app.config`; nothing else.
* :mod:`app.services.slack_notifier` is reused as-is. We bypass its
  ``should_send`` cooldown check because emergencies must always fire,
  and because we don't have a DB connection in scope here.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Final, Literal

from app.config import settings
from app.services.slack_notifier import SlackNotifier

logger = logging.getLogger(__name__)


# Channel labels surfaced back to the model + admin UI.
#   slack            -> Slack webhook responded 2xx
#   admin_dashboard  -> Slack was configured but failed; admin dashboard log is the canonical record
#   demo_pending     -> Slack is not configured; demo-only environment
AlertChannel = Literal["slack", "admin_dashboard", "demo_pending"]

# Status labels.
#   sent          -> external delivery confirmed
#   queued        -> reserved for future ambulance/queue integrations
#   demo_pending  -> no external channel configured; logged only
#   failed        -> external channel attempted and errored
AlertStatus = Literal["sent", "queued", "demo_pending", "failed"]

_ALLOWED_LEVELS: Final[frozenset[int]] = frozenset({1, 2, 3, 4, 5})

_SYMPTOM_SUMMARY_MAX_LEN: Final[int] = 600


@dataclass
class EmergencyAlert:
    """In-memory record of a single emergency dispatch attempt.

    Stored verbatim in :data:`DEMO_ALERT_LOG` (after ``asdict``) for the
    demo admin dashboard. Contains no PII by construction -- the model
    never produces PII into these fields, and the orchestrator
    redactor scrubs the patient message before it reaches the agent.
    """

    case_id: str
    session_id: str
    triage_level: int
    triage_color: str
    symptoms_summary: str
    dispatched_at: str  # ISO-8601 UTC, e.g. "2026-05-26T10:42:01+00:00"
    channel: AlertChannel
    status: AlertStatus
    language: str = "en"
    error: str | None = None


# Module-level in-memory store -- the demo admin dashboard data source.
# Read by ``GET /admin/emergency-alerts`` (registered in app/main.py in a
# later step). Process-local: resets on every server restart, which is
# fine for the demo.
DEMO_ALERT_LOG: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Helpers (internal). All synchronous and pure except _send_via_slack.
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string, second precision."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_str(value: Any, *, default: str = "") -> str:
    """Trim and coerce ``value`` to a string, never raising."""

    if value is None:
        return default
    if not isinstance(value, str):
        value = str(value)
    return value.strip() or default


def _validate_inputs(
    *,
    case_id: str,
    session_id: str,
    triage_level: Any,
    triage_color: str,
) -> str | None:
    """Return an error string when inputs are invalid, ``None`` otherwise."""

    if not case_id:
        return "case_id must be a non-empty string"
    if not session_id:
        return "session_id must be a non-empty string"
    if isinstance(triage_level, bool) or not isinstance(triage_level, int):
        return f"triage_level must be an int 1..5, got {type(triage_level).__name__}"
    if triage_level not in _ALLOWED_LEVELS:
        return f"triage_level must be 1..5, got {triage_level}"
    if not triage_color:
        return "triage_color must be a non-empty string"
    return None


async def _send_via_slack(
    *,
    case_id: str,
    session_id: str,
    triage_level: int,
    triage_color: str,
    symptoms_summary: str,
    language: str,
    dispatched_at: str,
) -> tuple[bool, str | None]:
    """Attempt the Slack webhook dispatch.

    Returns ``(success, error_description)``. ``success`` is ``True``
    only when the webhook returned a 2xx. Never raises -- exceptions
    are converted into an error description string.
    """

    if not settings.slack_webhook_url:
        return False, "slack_not_configured"

    notifier = SlackNotifier()
    severity_label = f"Level {triage_level} ({triage_color})"
    alert_msg = (
        f"Case {case_id} dispatched at {dispatched_at} (session {session_id})."
    )

    try:
        sent = await notifier.send_alert(
            session_id=session_id,
            language=language,
            user_message=symptoms_summary or "(no symptom summary provided)",
            severity=severity_label,
            confidence=None,
            department_name=None,
            emergency_reason=severity_label,
            alert_message=alert_msg,
        )
    except Exception as exc:  # noqa: BLE001 -- we never let this escape
        logger.exception("Slack alert dispatch raised: %s", exc)
        return False, f"slack_exception: {type(exc).__name__}: {exc}"

    if not sent:
        return False, "slack_non_2xx_response"
    return True, None


# ---------------------------------------------------------------------------
# Public ADK tool
# ---------------------------------------------------------------------------


async def dispatch_emergency(
    case_id: str,
    session_id: str,
    triage_level: int,
    triage_color: str,
    symptoms_summary: str,
    language: str = "en",
) -> dict[str, Any]:
    """Record an emergency dispatch and notify staff out-of-band.

    Call this tool after :func:`classify_triage` for any case classified
    as Level 1 (Red) or Level 2 (Orange). Do not call for Level 3-5.

    Behaviour:

    * Always appends an :class:`EmergencyAlert` record to
      :data:`DEMO_ALERT_LOG` (the demo admin dashboard's data source).
    * If ``settings.slack_webhook_url`` is configured, also dispatches
      the alert via the existing :class:`SlackNotifier`. The Slack
      payload contains only ``case_id``, ``session_id``,
      ``triage_level``, ``triage_color``, ``symptoms_summary``, and
      the dispatch timestamp -- no PII.
    * Never raises. On any failure the tool returns a structured
      ``status="failed"`` dict so the agent can continue the
      conversation.

    :param case_id: Caller-assigned case identifier (e.g. the same UUID
        the FastAPI handler stores on ``emergency_events.id`` once the
        DB write step lands). Required; non-empty.
    :param session_id: Hotline session UUID. Required; non-empty.
    :param triage_level: 1..5 (must align with the five-level JSON).
    :param triage_color: ``"Red" | "Orange" | "Yellow" | "Green" | "Blue"``.
    :param symptoms_summary: Short, NEUTRAL summary of the patient's
        symptoms -- must NOT contain name, phone, address, ID, or
        email. The orchestrator redacts the patient message before it
        reaches the agent; this tool trusts that contract.
    :param language: ``"en"`` (default) or ``"th"`` -- controls the
        confirmation message returned to the agent. Slack payload uses
        the same value.
    :returns: Stable structured dict::

            {
                "action": "emergency_dispatched",
                "case_id": "...",
                "channel": "slack" | "admin_dashboard" | "demo_pending",
                "status": "sent" | "queued" | "demo_pending" | "failed",
                "message": "...",
                "error": "..."  # only present when status == "failed"
            }
    """

    case_id_clean = _clean_str(case_id)
    session_id_clean = _clean_str(session_id)
    triage_color_clean = _clean_str(triage_color)
    symptoms_clean = _clean_str(symptoms_summary)[:_SYMPTOM_SUMMARY_MAX_LEN]
    language_clean = "th" if _clean_str(language).lower() == "th" else "en"

    validation_error = _validate_inputs(
        case_id=case_id_clean,
        session_id=session_id_clean,
        triage_level=triage_level,
        triage_color=triage_color_clean,
    )
    if validation_error is not None:
        # Do not pollute the demo log with malformed entries; just tell
        # the agent the call was malformed and keep the conversation going.
        return {
            "action": "emergency_dispatched",
            "case_id": case_id_clean or "unknown-case",
            "channel": "demo_pending",
            "status": "failed",
            "message": _confirmation_message(language_clean, "failed"),
            "error": validation_error,
        }

    dispatched_at = _now_iso()

    # Try Slack first. The channel / status outcome flows from there.
    if settings.slack_webhook_url:
        slack_ok, slack_err = await _send_via_slack(
            case_id=case_id_clean,
            session_id=session_id_clean,
            triage_level=triage_level,
            triage_color=triage_color_clean,
            symptoms_summary=symptoms_clean,
            language=language_clean,
            dispatched_at=dispatched_at,
        )
        if slack_ok:
            channel: AlertChannel = "slack"
            status: AlertStatus = "sent"
            error: str | None = None
        else:
            channel = "admin_dashboard"
            status = "failed"
            error = slack_err
    else:
        channel = "demo_pending"
        status = "demo_pending"
        error = None

    alert = EmergencyAlert(
        case_id=case_id_clean,
        session_id=session_id_clean,
        triage_level=triage_level,
        triage_color=triage_color_clean,
        symptoms_summary=symptoms_clean,
        dispatched_at=dispatched_at,
        channel=channel,
        status=status,
        language=language_clean,
        error=error,
    )
    DEMO_ALERT_LOG.append(asdict(alert))
    logger.info(
        "Emergency dispatched | case=%s session=%s level=%s color=%s channel=%s status=%s",
        alert.case_id,
        alert.session_id,
        alert.triage_level,
        alert.triage_color,
        alert.channel,
        alert.status,
    )

    result: dict[str, Any] = {
        "action": "emergency_dispatched",
        "case_id": alert.case_id,
        "channel": alert.channel,
        "status": alert.status,
        "message": _confirmation_message(language_clean, alert.status),
    }
    if alert.error:
        result["error"] = alert.error
    return result


# ---------------------------------------------------------------------------
# Demo-dashboard accessors (not exposed as ADK tools)
# ---------------------------------------------------------------------------


def get_demo_alert_log() -> list[dict[str, Any]]:
    """Snapshot of :data:`DEMO_ALERT_LOG` for the admin dashboard endpoint.

    Returns a shallow copy so the caller can safely serialise / iterate
    without racing future appends.
    """

    return list(DEMO_ALERT_LOG)


def clear_demo_alert_log() -> None:
    """Reset the in-memory log. Intended for tests."""

    DEMO_ALERT_LOG.clear()


# ---------------------------------------------------------------------------
# Confirmation copy
# ---------------------------------------------------------------------------


_MESSAGES_EN: Final[dict[AlertStatus, str]] = {
    "sent": "Emergency alert logged. Admin has been notified.",
    "queued": "Emergency alert logged. Notification queued for delivery.",
    "demo_pending": "Emergency alert logged. Admin will be notified on the dashboard.",
    "failed": (
        "Emergency alert logged. External notification could not be confirmed; "
        "the case is visible on the admin dashboard."
    ),
}

_MESSAGES_TH: Final[dict[AlertStatus, str]] = {
    "sent": "บันทึกการแจ้งเหตุฉุกเฉินแล้ว เจ้าหน้าที่ได้รับการแจ้งเตือนแล้ว",
    "queued": "บันทึกการแจ้งเหตุฉุกเฉินแล้ว การแจ้งเตือนอยู่ในคิวเพื่อจัดส่ง",
    "demo_pending": "บันทึกการแจ้งเหตุฉุกเฉินแล้ว เจ้าหน้าที่จะเห็นในแดชบอร์ดผู้ดูแล",
    "failed": (
        "บันทึกการแจ้งเหตุฉุกเฉินแล้ว ไม่สามารถยืนยันการแจ้งเตือนภายนอกได้ "
        "เคสนี้แสดงอยู่ในแดชบอร์ดผู้ดูแล"
    ),
}


def _confirmation_message(language: str, status: AlertStatus) -> str:
    table = _MESSAGES_TH if language == "th" else _MESSAGES_EN
    return table.get(status, table["demo_pending"])


__all__ = [
    "DEMO_ALERT_LOG",
    "EmergencyAlert",
    "clear_demo_alert_log",
    "dispatch_emergency",
    "get_demo_alert_log",
]
