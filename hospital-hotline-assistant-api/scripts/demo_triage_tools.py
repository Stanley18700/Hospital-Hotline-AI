"""One-shot demo of the triage tools. Run with `python scripts/demo_triage_tools.py`."""

import json

from app.agent.tools.triage_tools import (
    ask_followup,
    classify_triage,
    get_department_advice,
)


def show(title: str, payload: dict) -> None:
    print(f"--- {title} ---")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print()


show("ask_followup", ask_followup("Where exactly does it hurt?"))

show(
    "classify_triage (Level 1)",
    classify_triage(
        symptoms_summary="Patient is unresponsive and not breathing after collapse.",
        level=1,
        reasoning="Cardiac/respiratory arrest matches Level 1 examples.",
        key_question_answered="yes",
    ),
)

show("get_department_advice (Level 1, EN)", get_department_advice(1, "en"))
show("get_department_advice (Level 1, TH)", get_department_advice(1, "th"))

show(
    "classify_triage (Level 4)",
    classify_triage(
        symptoms_summary="Right ankle sprained two hours ago; some swelling, can bear weight.",
        level=4,
        reasoning="One resource needed (X-ray), stable vitals.",
        key_question_answered="one resource",
    ),
)

show("get_department_advice (Level 4, EN)", get_department_advice(4, "en"))

show(
    "safety-check fires when summary mentions a Level 1 example but the agent picked 3",
    classify_triage(
        symptoms_summary="Patient appears to be in cardiac arrest at home.",
        level=3,
        reasoning="Test of safety net override.",
        key_question_answered="unclear",
    ),
)

show("error: invalid level", classify_triage("foo", 7, "bar", "no"))
show("error: unsupported language", get_department_advice(2, "jp"))
