"""Triage-domain tools exposed to the ADK agent.

These functions are registered with the :class:`LlmAgent` and are
auto-wrapped by ADK as ``FunctionTool`` instances (per the official
"Function Tools" guide). They are intentionally synchronous and side
effect free -- their job is to give the model structured handles back
into the medical team's five-level JSON so the conversation grounds in
that source of truth.

Three tools:

* :func:`ask_followup` -- the agent commits to gathering more
  information by asking exactly one question.
* :func:`classify_triage` -- the agent commits to a triage level. The
  return value pulls the level's canonical metadata from the JSON and
  adds a lightweight safety check against the JSON's own Level 1
  examples to catch obvious mis-classifications.
* :func:`get_department_advice` -- the agent looks up routing,
  interim-action, and urgency wording for a chosen level. English is
  the primary language; Thai strings are provided alongside so the
  hotline can speak both.

Design notes:

* No PII handling here -- that lives in :mod:`app.agent.tools.pii_tools`.
* :mod:`app.services.rule_engine` is intentionally *not* reused: it is
  designed to evaluate triggers loaded from Postgres, but tool
  functions in the ADK runtime have no DB session in scope. The JSON
  itself is the primary safety net; reusing the rule engine would
  duplicate the source of truth.
* Returns are stable, JSON-serialisable dicts so the runner and the
  FastAPI handler can pass them through without further reshaping.
"""

from __future__ import annotations

from typing import Any, Final, Literal

from app.agent.triage_config import get_triage_level, get_triage_levels

# ---------------------------------------------------------------------------
# Per-language, per-level advice text. EN is the primary surface; TH mirrors
# the same keys so the hotline can speak both. Keep these short -- they are
# read aloud by TTS in voice mode and rendered in the chat bubble in text
# mode. Department names use the same vocabulary as ``app/data/departments.json``
# wherever possible so the admin UI stays consistent.
# ---------------------------------------------------------------------------

LanguageCode = Literal["en", "th"]

_DEPARTMENT_BY_LEVEL: Final[dict[int, dict[LanguageCode, str]]] = {
    1: {"en": "Emergency Resuscitation Bay", "th": "ห้องกู้ชีพฉุกเฉิน"},
    2: {"en": "Emergency Department", "th": "แผนกอุบัติเหตุและฉุกเฉิน"},
    3: {"en": "Emergency Department (Urgent Care)", "th": "แผนกฉุกเฉิน (เร่งด่วน)"},
    4: {"en": "Outpatient / Walk-in Clinic", "th": "คลินิกผู้ป่วยนอก"},
    5: {"en": "Outpatient / Pharmacy Desk", "th": "ผู้ป่วยนอก / เภสัชกรรม"},
}

_URGENCY_STATEMENT: Final[dict[int, dict[LanguageCode, str]]] = {
    1: {
        "en": "Life-threatening. You need immediate care right now.",
        "th": "เป็นอันตรายถึงชีวิต ต้องได้รับการดูแลทันที",
    },
    2: {
        "en": "High risk. You should be seen within the next 10 to 15 minutes.",
        "th": "ความเสี่ยงสูง ควรพบเจ้าหน้าที่ภายใน 10 ถึง 15 นาที",
    },
    3: {
        "en": "Urgent. You should be seen within the next hour.",
        "th": "เร่งด่วน ควรพบเจ้าหน้าที่ภายในหนึ่งชั่วโมง",
    },
    4: {
        "en": "Standard care. You should be seen within about two hours.",
        "th": "ดูแลตามปกติ จะได้รับการพบเจ้าหน้าที่ภายในประมาณสองชั่วโมง",
    },
    5: {
        "en": "Non-urgent. You can be seen within about four hours.",
        "th": "ไม่เร่งด่วน สามารถเข้ารับบริการได้ภายในประมาณสี่ชั่วโมง",
    },
}

_INTERIM_ACTION: Final[dict[int, dict[LanguageCode, str]]] = {
    1: {
        "en": (
            "Stay where you are. Do not eat or drink anything. If you can, "
            "stay on the line; emergency staff are being notified now."
        ),
        "th": (
            "อยู่กับที่ ห้ามรับประทานอาหารหรือเครื่องดื่ม หากเป็นไปได้กรุณาอยู่ในสาย "
            "เจ้าหน้าที่ฉุกเฉินกำลังได้รับแจ้งแล้ว"
        ),
    },
    2: {
        "en": (
            "Come to the Emergency Department right away. Have someone "
            "accompany you if possible, and do not drive yourself."
        ),
        "th": (
            "กรุณามาที่แผนกฉุกเฉินทันที หากเป็นไปได้ให้มีคนพามาด้วย "
            "ห้ามขับรถมาเอง"
        ),
    },
    3: {
        "en": (
            "Come to the hospital as soon as you can. Check in at the "
            "triage desk and bring a list of any medications you take."
        ),
        "th": (
            "กรุณามาโรงพยาบาลโดยเร็วที่สุด ลงทะเบียนที่จุดคัดกรอง "
            "และนำรายชื่อยาที่รับประทานเป็นประจำมาด้วย"
        ),
    },
    4: {
        "en": (
            "Visit the outpatient clinic during opening hours. Bring any "
            "current medications and your hospital ID if you have one."
        ),
        "th": (
            "กรุณาเข้ารับบริการที่คลินิกผู้ป่วยนอกในเวลาทำการ "
            "นำยาที่ใช้อยู่และบัตรประจำตัวโรงพยาบาลมาด้วยหากมี"
        ),
    },
    5: {
        "en": (
            "Visit the outpatient or pharmacy desk during opening hours. "
            "No on-site emergency action is needed."
        ),
        "th": (
            "กรุณาเข้ารับบริการที่จุดผู้ป่วยนอกหรือเภสัชกรรมในเวลาทำการ "
            "ไม่จำเป็นต้องดำเนินการฉุกเฉิน"
        ),
    },
}


def _coerce_level(level: Any) -> int:
    """Validate and coerce ``level`` to an int in 1..5.

    Raises :class:`ValueError` when the input is not a 1..5 integer, so
    the agent gets a structured error rather than an unhandled crash.
    """

    if isinstance(level, bool) or not isinstance(level, int) or level not in (1, 2, 3, 4, 5):
        raise ValueError(f"level must be one of 1..5, got {level!r}")
    return level


def _coerce_language(language: Any) -> LanguageCode:
    """Validate ``language`` against the supported set (``en`` / ``th``)."""

    if language not in ("en", "th"):
        raise ValueError(f"language must be 'en' or 'th', got {language!r}")
    return language  # type: ignore[return-value]


def _level1_examples_lower() -> list[str]:
    """Return Level 1 example strings from the JSON, lower-cased."""

    level1 = get_triage_level(1)
    return [str(example).lower() for example in (level1.get("examples") or [])]


# ---------------------------------------------------------------------------
# Public tool functions -- the LLM sees the function name, type hints, and
# docstring as the tool schema. Keep docstrings tight and accurate.
# ---------------------------------------------------------------------------


def ask_followup(question: str) -> dict[str, Any]:
    """Ask the patient ONE follow-up question to refine the triage level.

    Call this tool when you do not yet have enough information to commit
    to a triage level. The orchestrator will deliver ``question`` back
    to the patient verbatim, then resume the conversation on the next
    turn.

    :param question: A single, plainly worded follow-up question in the
        patient's language. Must not request PII (name, phone, address,
        ID number, email). Must be one question, not a list.
    :returns: ``{"status": ..., "action": "ask_followup", "question": ...}``
    """

    cleaned = (question or "").strip()
    if not cleaned:
        return {
            "status": "error",
            "action": "ask_followup",
            "question": "",
            "error": "question must be a non-empty string",
        }

    return {
        "status": "ok",
        "action": "ask_followup",
        "question": cleaned,
    }


def classify_triage(
    symptoms_summary: str,
    level: int,
    reasoning: str,
    key_question_answered: str,
) -> dict[str, Any]:
    """Commit to a triage level for the case currently being assessed.

    The returned dict pulls canonical metadata for ``level`` from the
    five-level JSON, plus a lightweight safety check that scans
    ``symptoms_summary`` against the JSON's own Level 1 examples. If the
    safety check fires but the chosen level is not 1, the orchestrator
    can choose to override.

    :param symptoms_summary: One- or two-sentence neutral summary of the
        patient's symptoms (no PII).
    :param level: Integer 1..5 -- the triage level you are committing
        to.
    :param reasoning: Short rationale for picking this level (internal
        only; not shown to the patient).
    :param key_question_answered: How the level's key question was
        answered (``"yes"``, ``"no"``, ``"partial"``, or a short
        free-form phrase). Used to audit triage decisions.
    :returns: Stable structured dict with the keys ``action``,
        ``level``, ``color``, ``response_time``, ``placement``,
        ``key_question``, ``key_question_answered``,
        ``symptoms_summary``, ``reasoning``, ``is_emergency``, and
        ``safety_check``.
    """

    try:
        level_int = _coerce_level(level)
    except ValueError as exc:
        return {
            "status": "error",
            "action": "classify_triage",
            "error": str(exc),
        }

    level_data = get_triage_level(level_int)
    summary_clean = (symptoms_summary or "").strip()
    reasoning_clean = (reasoning or "").strip()
    answered_clean = (key_question_answered or "").strip()

    # Lightweight safety net: scan the agent's own summary against the
    # canonical Level 1 examples. If we see one of those phrases but
    # the chosen level is 2-5, surface the conflict so the orchestrator
    # can override. This is intentionally simple substring matching --
    # the medical team's JSON is the source of truth, not a regex DSL.
    matched_l1: list[str] = []
    if summary_clean:
        lowered = summary_clean.lower()
        for example in _level1_examples_lower():
            if example and example in lowered:
                matched_l1.append(example)

    safety_check: dict[str, Any] = {
        "triggered": bool(matched_l1) and level_int != 1,
        "matched_level1_examples": matched_l1,
        "suggested_level": 1 if matched_l1 else None,
    }

    return {
        "status": "ok",
        "action": "classify_triage",
        "level": level_int,
        "color": level_data["color"],
        "response_time": level_data["response_time"],
        "placement": level_data["placement"],
        "key_question": level_data["key_question"],
        "key_question_answered": answered_clean,
        "symptoms_summary": summary_clean,
        "reasoning": reasoning_clean,
        "is_emergency": level_int in (1, 2),
        "safety_check": safety_check,
    }


def get_department_advice(level: int, language: str = "en") -> dict[str, Any]:
    """Return department, interim action, and urgency wording for a level.

    Call this tool after :func:`classify_triage` so the reply you give
    the patient is grounded in the team-approved routing instructions
    rather than improvised.

    :param level: Integer 1..5.
    :param language: ``"en"`` (default) or ``"th"``.
    :returns: Stable structured dict with the keys ``action``,
        ``level``, ``color``, ``language``, ``department``,
        ``interim_action``, ``urgency_statement``, ``response_time``,
        ``placement``.
    """

    try:
        level_int = _coerce_level(level)
        lang = _coerce_language(language)
    except ValueError as exc:
        return {
            "status": "error",
            "action": "department_advice",
            "error": str(exc),
        }

    level_data = get_triage_level(level_int)

    return {
        "status": "ok",
        "action": "department_advice",
        "level": level_int,
        "color": level_data["color"],
        "language": lang,
        "department": _DEPARTMENT_BY_LEVEL[level_int][lang],
        "interim_action": _INTERIM_ACTION[level_int][lang],
        "urgency_statement": _URGENCY_STATEMENT[level_int][lang],
        "response_time": level_data["response_time"],
        "placement": level_data["placement"],
    }


# Re-export the configured level list as a small convenience for callers
# that want to enumerate available levels without importing triage_config
# directly (for example in tests). Read-only -- do not mutate.
TRIAGE_LEVELS_SNAPSHOT: Final[list[dict[str, Any]]] = get_triage_levels()


__all__ = [
    "TRIAGE_LEVELS_SNAPSHOT",
    "ask_followup",
    "classify_triage",
    "get_department_advice",
]
