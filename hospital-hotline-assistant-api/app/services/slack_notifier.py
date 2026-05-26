from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import asyncpg
import httpx

from app.config import settings


SEVERITY_ORDER = {
    "unknown": 0,
    "general": 1,
    "urgent": 2,
    "emergency": 3,
}


class SlackNotifier:
    async def should_send(
        self,
        connection: asyncpg.Connection,
        session_id: str,
        severity: str,
    ) -> bool:
        threshold_level = SEVERITY_ORDER.get(settings.alert_severity_threshold, 3)
        current_level = SEVERITY_ORDER.get(severity, 0)
        if current_level < threshold_level:
            return False

        if not settings.slack_webhook_url:
            return False

        row = await connection.fetchrow("SELECT metadata FROM sessions WHERE id = $1", session_id)
        if not row:
            return False

        metadata = row.get("metadata") or {}
        last_alert = metadata.get("last_alert_at")
        if not last_alert:
            return True

        try:
            last_dt = datetime.fromisoformat(last_alert.replace("Z", "+00:00"))
        except ValueError:
            return True

        delta = datetime.now(timezone.utc) - last_dt
        return delta.total_seconds() >= settings.alert_cooldown_seconds

    async def send_alert(
        self,
        *,
        session_id: str,
        language: str,
        user_message: str,
        severity: str,
        confidence: float | None,
        department_name: str | None,
        emergency_reason: str | None,
        alert_message: str | None,
    ) -> bool:
        if not settings.slack_webhook_url:
            return False

        payload: dict[str, Any] = {
            "text": f"Hospital Hotline Alert - {severity.upper()}",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"Hotline Alert: {severity.upper()}"},
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Session:*\n`{session_id}`"},
                        {"type": "mrkdwn", "text": f"*Language:*\n{language}"},
                        {"type": "mrkdwn", "text": f"*Severity:*\n{severity}"},
                        {"type": "mrkdwn", "text": f"*Confidence:*\n{confidence if confidence is not None else 'n/a'}"},
                        {"type": "mrkdwn", "text": f"*Department:*\n{department_name or 'n/a'}"},
                        {"type": "mrkdwn", "text": f"*Reason:*\n{emergency_reason or 'n/a'}"},
                    ],
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Latest User Message:*\n{user_message}"},
                },
            ],
        }
        if alert_message:
            payload["blocks"].append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Alert message:*\n{alert_message}"},
                }
            )

        async with httpx.AsyncClient(timeout=6.0) as client:
            response = await client.post(settings.slack_webhook_url, json=payload)
            return response.status_code < 300

    async def send_emergency_dispatch(
        self,
        *,
        case_id: str,
        session_id: str,
        language: str,
        triage_level: int | None,
        triage_color: str | None,
        symptoms_summary: str | None,
        patient_name: str,
        patient_phone: str,
        patient_address: str,
        patient_notes: str | None = None,
    ) -> bool:
        """Post a Level 1 ambulance / admin dispatch alert to Slack.

        Distinct from :meth:`send_alert` (which is used by the
        conversational /chat path). This variant intentionally
        includes patient identifiers because the Slack channel is the
        backend's hand-off to a human dispatcher -- the LLM is not on
        this path. The payload format is similar to ``send_alert`` so
        the same Slack channel can be used for both flows.

        Privacy note: this is the ONLY part of the codebase that
        formats PII into a Slack payload. The PII passes from the
        FastAPI request body, through :class:`SlackPiiHandoffSink`,
        directly to this method, and out to the configured Slack
        webhook. It never touches the ADK runner or Vertex.

        Returns ``True`` on a 2xx response from the Slack webhook,
        ``False`` otherwise (including when ``slack_webhook_url`` is
        unset). Never raises.
        """

        if not settings.slack_webhook_url:
            return False

        level_label = (
            f"Level {triage_level}" + (f" ({triage_color})" if triage_color else "")
            if triage_level is not None
            else (triage_color or "Level 1 (Red)")
        )

        fields = [
            {"type": "mrkdwn", "text": f"*Case:*\n`{case_id}`"},
            {"type": "mrkdwn", "text": f"*Session:*\n`{session_id}`"},
            {"type": "mrkdwn", "text": f"*Triage:*\n{level_label}"},
            {"type": "mrkdwn", "text": f"*Language:*\n{language}"},
            {"type": "mrkdwn", "text": f"*Patient:*\n{patient_name}"},
            {"type": "mrkdwn", "text": f"*Phone:*\n{patient_phone}"},
        ]

        payload: dict[str, Any] = {
            "text": f"Hospital Hotline EMERGENCY DISPATCH - {case_id}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"EMERGENCY DISPATCH - {level_label}",
                    },
                },
                {"type": "section", "fields": fields},
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Address:*\n{patient_address}",
                    },
                },
            ],
        }

        if patient_notes:
            payload["blocks"].append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Dispatcher notes:*\n{patient_notes}",
                    },
                }
            )

        if symptoms_summary:
            payload["blocks"].append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Symptoms summary:*\n{symptoms_summary}",
                    },
                }
            )

        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                response = await client.post(settings.slack_webhook_url, json=payload)
                return response.status_code < 300
        except httpx.HTTPError:
            return False
