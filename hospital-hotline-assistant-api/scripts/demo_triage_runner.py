"""Demo of the TriageRunner normalisation layer.

We exercise :meth:`TriageRunner.normalize` directly with two synthetic
tool-call streams so the demo runs without ``google-adk`` installed
and without contacting Vertex AI:

1. A normal follow-up turn -- the agent only called ``ask_followup``.
2. A Level 1 emergency classification -- the agent called
   ``classify_triage``, ``get_department_advice``,
   ``trigger_pii_collection``, and ``dispatch_emergency`` in order.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from app.agent.triage_runner import TriageRunner


def show(title: str, payload: Any) -> None:
    print(f"--- {title} ---")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print()


def scenario_followup() -> None:
    tool_calls = [
        {
            "name": "ask_followup",
            "args": {"question": "How long have you had the chest pain?"},
            "response": {
                "action": "ask_followup",
                "question": "How long have you had the chest pain?",
            },
        }
    ]
    result = TriageRunner.normalize(
        tool_calls=tool_calls,
        reply="How long have you had the chest pain?",
        session_id="11111111-1111-1111-1111-111111111111",
        language="en",
        input_mode="text",
    )
    show("1) follow-up turn", asdict(result))


def scenario_level1_emergency() -> None:
    tool_calls = [
        {
            "name": "classify_triage",
            "args": {
                "symptoms_summary": "Crushing chest pain radiating to left arm, sweating, 10 minutes.",
                "level": 1,
                "reasoning": "Classic ACS presentation; meets Level 1 key question (life-threatening).",
                "key_question_answered": "Yes - life-threatening cardiac symptoms.",
            },
            "response": {
                "action": "classify_triage",
                "level": 1,
                "color": "Red",
                "response_time": "Immediate",
                "placement": "Resuscitation bay",
                "key_question": "Is the patient's life in immediate danger?",
                "key_question_answered": "Yes - life-threatening cardiac symptoms.",
                "symptoms_summary": "Crushing chest pain radiating to left arm, sweating, 10 minutes.",
                "reasoning": "Classic ACS presentation; meets Level 1 key question (life-threatening).",
                "is_emergency": True,
                "safety_check": {"triggered": False},
            },
        },
        {
            "name": "get_department_advice",
            "args": {"level": 1, "language": "en"},
            "response": {
                "department": "Emergency Department - Resuscitation",
                "interim_action": "Stay on the line. Do not move the patient. Loosen tight clothing.",
                "urgency_statement": "An emergency team is being dispatched right now.",
                "response_time": "Immediate",
                "placement": "Resuscitation bay",
            },
        },
        {
            "name": "trigger_pii_collection",
            "args": {"session_id": "22222222-2222-2222-2222-222222222222"},
            "response": {
                "action": "collect_pii",
                "fields": ["name", "phone", "address"],
                "session_id": "22222222-2222-2222-2222-222222222222",
            },
        },
        {
            "name": "dispatch_emergency",
            "args": {
                "case_id": "case-2026-0001",
                "session_id": "22222222-2222-2222-2222-222222222222",
                "triage_level": 1,
                "triage_color": "Red",
                "symptoms_summary": "Crushing chest pain radiating to left arm, sweating, 10 minutes.",
                "language": "en",
            },
            "response": {
                "action": "emergency_dispatched",
                "case_id": "case-2026-0001",
                "channel": "demo_pending",
                "status": "demo_pending",
                "message": "Emergency alert logged. Admin has been notified.",
            },
        },
    ]
    result = TriageRunner.normalize(
        tool_calls=tool_calls,
        reply="Stay on the line. An emergency team is on the way.",
        session_id="22222222-2222-2222-2222-222222222222",
        language="en",
        input_mode="voice",
    )
    show("2) level 1 emergency classification", asdict(result))


def main() -> None:
    scenario_followup()
    scenario_level1_emergency()


if __name__ == "__main__":
    main()
