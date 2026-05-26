"""Demo of the emergency dispatch tool under both Slack-on and Slack-off configs."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from app.agent.tools.alert_tools import (
    clear_demo_alert_log,
    dispatch_emergency,
    get_demo_alert_log,
)
from app.agent.tools import get_triage_tools
from app.config import settings


def show(title: str, payload: Any) -> None:
    print(f"--- {title} ---")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print()


async def run() -> None:
    print("--- agent tool registry ---")
    print([fn.__name__ for fn in get_triage_tools()])
    print()

    # Scenario 1: no Slack configured (typical demo).
    settings.slack_webhook_url = None
    clear_demo_alert_log()
    result = await dispatch_emergency(
        case_id="case-9001",
        session_id="3b8c9a44-7c30-4cd9-a4b1-89c2c8b3a1ee",
        triage_level=1,
        triage_color="Red",
        symptoms_summary="Caller reports unresponsive adult, not breathing after collapse.",
        language="en",
    )
    show("scenario 1: no Slack configured", result)
    show("DEMO_ALERT_LOG after scenario 1", get_demo_alert_log())

    # Scenario 2: Slack configured but webhook is unreachable -> graceful failure.
    settings.slack_webhook_url = "http://127.0.0.1:1/nonexistent-webhook"
    clear_demo_alert_log()
    result = await dispatch_emergency(
        case_id="case-9002",
        session_id="3b8c9a44-7c30-4cd9-a4b1-89c2c8b3a1ee",
        triage_level=2,
        triage_color="Orange",
        symptoms_summary="Active chest pain with shortness of breath for 20 minutes.",
        language="en",
    )
    show("scenario 2: Slack configured but webhook unreachable", result)
    show("DEMO_ALERT_LOG after scenario 2", get_demo_alert_log())

    # Scenario 3: validation error does not pollute the log.
    settings.slack_webhook_url = None
    clear_demo_alert_log()
    result = await dispatch_emergency(
        case_id="",
        session_id="abc",
        triage_level=7,        # invalid
        triage_color="",
        symptoms_summary="",
    )
    show("scenario 3: invalid input", result)
    show("DEMO_ALERT_LOG after scenario 3 (should be empty)", get_demo_alert_log())


asyncio.run(run())
