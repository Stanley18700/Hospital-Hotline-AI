"""Runtime adapter between the FastAPI / WebSocket layer and the ADK agent.

This module owns the small bit of glue code that:

1. Builds and caches the ADK :class:`Runner` (one per input mode) plus a
   shared :class:`InMemorySessionService`, so a hotline session retains
   conversation history across patient turns regardless of whether the
   turn arrived over HTTP or a WebSocket frame.
2. Sends each patient message through the runner.
3. Walks the resulting event stream to pull out tool calls + their
   responses, and the agent's final user-facing text.
4. Normalises everything into a stable :class:`TriageRunResult` the
   FastAPI / WebSocket layer can serialise without further reshaping.

What this module deliberately does NOT do:

* No FastAPI imports, no request / response models. The caller wraps a
  :class:`TriageRunResult` into whatever HTTP / WS shape it needs.
* No Postgres writes. The migration plan's ``AdkTriageService`` will
  layer DB persistence on top of this adapter in a later step.
* No raw PII storage. We hold the user's message only for the duration
  of a single ``run`` call (as parameters); it is never written to a
  module-level cache, never logged, and never returned in the result.

Concurrency model: ``TriageRunner.run`` is safe to call concurrently
for *different* ``session_id`` values. Concurrent calls for the *same*
``session_id`` should be serialised by the caller (FastAPI per-session
lock or single-flight WebSocket handler), because ADK's session
service does not arbitrate intra-session contention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from app.agent.triage_agent import build_triage_agent
from app.agent.triage_state import TriageState

logger = logging.getLogger(__name__)

DEFAULT_APP_NAME: str = "hospital_hotline"
DEFAULT_USER_ID_PREFIX: str = "patient"

# Stable vocabulary the FastAPI / WebSocket layer can switch on without
# pattern-matching on free-form strings.
NextAction = Literal[
    "await_followup",   # Agent asked a follow-up; wait for the patient's next message.
    "complete",         # Triage finalised, Level 3-5, no escalation.
    "escalate",         # Level 1 or 2; staff have been alerted via dispatch_emergency.
    "collect_pii",      # Level 1; secure PII form must be triggered next.
    "error",            # Runtime failure inside the ADK runner.
]


@dataclass
class TriageRunResult:
    """Normalised output of one :meth:`TriageRunner.run` call.

    Stable JSON-serialisable shape -- the caller can ``asdict(result)``
    and return it directly over HTTP / WebSocket.
    """

    reply: str
    triage_result: dict[str, Any] | None
    advice: dict[str, Any] | None
    next_action: NextAction
    follow_up_question: str | None
    alert_requested: bool
    pii_collection_requested: bool
    state: str  # TriageState value
    session_id: str
    language: str
    input_mode: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


# Tool names the runner consumes downstream. Keep in sync with
# :mod:`app.agent.tools.triage_tools`, :mod:`app.agent.tools.pii_tools`,
# :mod:`app.agent.tools.alert_tools`.
_TOOL_CLASSIFY: str = "classify_triage"
_TOOL_ADVICE: str = "get_department_advice"
_TOOL_FOLLOWUP: str = "ask_followup"
_TOOL_DISPATCH: str = "dispatch_emergency"
_TOOL_PII: str = "trigger_pii_collection"


class TriageRunner:
    """Process-wide adapter that drives the ADK triage agent.

    Construct one instance per process (use :func:`get_default_runner`)
    and reuse it for every HTTP / WebSocket call. Internally caches
    one ADK :class:`Runner` per input mode and shares a single
    :class:`InMemorySessionService` across them so conversations keep
    their memory across modality switches.
    """

    def __init__(self, *, app_name: str = DEFAULT_APP_NAME) -> None:
        self.app_name = app_name
        self._session_service: Any = None
        self._text_runner: Any = None
        self._voice_runner: Any = None

    # -- public API ---------------------------------------------------

    async def run(
        self,
        *,
        session_id: str,
        user_message: str,
        language: str = "en",
        input_mode: str = "text",
        user_id: str | None = None,
    ) -> TriageRunResult:
        """Run one triage turn and return a normalised result.

        :param session_id: Hotline session UUID (matches the
            ``sessions.id`` column the FastAPI layer already uses).
        :param user_message: Patient's latest message, already
            PII-redacted by the caller. Treated as opaque text.
        :param language: ``"en"`` or ``"th"``. Passed to the agent's
            tools for localised advice copy.
        :param input_mode: ``"text"`` or ``"voice"``. Picks which
            cached ADK Runner / agent variant handles the turn.
        :param user_id: Optional ADK user identifier. Defaults to
            ``"patient-{session_id}"`` so each session is isolated.
        :returns: :class:`TriageRunResult` -- never raises; runtime
            failures surface as ``next_action="error"``.
        """

        mode = "voice" if input_mode == "voice" else "text"
        effective_user_id = user_id or f"{DEFAULT_USER_ID_PREFIX}-{session_id}"
        effective_message = user_message or ""

        try:
            runner = await self._ensure_runner(input_mode=mode)
            await self._ensure_session(
                user_id=effective_user_id, session_id=session_id
            )
            tool_calls, final_reply = await self._invoke(
                runner=runner,
                user_id=effective_user_id,
                session_id=session_id,
                user_message=effective_message,
            )
        except RuntimeError as exc:
            # google-adk not installed, or runner init failed.
            logger.warning("Triage runner unavailable: %s", exc)
            return self._error_result(
                session_id=session_id,
                language=language,
                input_mode=mode,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 -- never let agent errors escape
            logger.exception("Triage runner failed for session %s: %s", session_id, exc)
            return self._error_result(
                session_id=session_id,
                language=language,
                input_mode=mode,
                error=f"{type(exc).__name__}: {exc}",
            )

        return self.normalize(
            tool_calls=tool_calls,
            reply=final_reply,
            session_id=session_id,
            language=language,
            input_mode=mode,
        )

    # -- normalisation (pure, testable) -------------------------------

    @staticmethod
    def normalize(
        *,
        tool_calls: list[dict[str, Any]],
        reply: str,
        session_id: str,
        language: str,
        input_mode: str = "text",
    ) -> TriageRunResult:
        """Turn a tool-call stream + final text into a :class:`TriageRunResult`.

        Pure function so unit tests and the demo can exercise the
        adapter logic without needing ADK installed or Vertex
        credentials.
        """

        classify_tc = _find_last(tool_calls, _TOOL_CLASSIFY)
        advice_tc = _find_last(tool_calls, _TOOL_ADVICE)
        followup_tc = _find_last(tool_calls, _TOOL_FOLLOWUP)
        dispatch_tc = _find_last(tool_calls, _TOOL_DISPATCH)
        pii_tc = _find_last(tool_calls, _TOOL_PII)

        triage_result = _tool_response(classify_tc)
        advice = _tool_response(advice_tc)
        follow_up_question = (_tool_response(followup_tc) or {}).get("question")
        alert_requested = dispatch_tc is not None
        pii_collection_requested = pii_tc is not None

        # Branch resolution. Order matters: PII collection is the most
        # specific terminal state, then escalation, then plain
        # completion, then "still gathering info".
        if pii_collection_requested:
            next_action: NextAction = "collect_pii"
            state = TriageState.PII_COLLECT.value
        elif triage_result and (
            alert_requested or _is_emergency(triage_result)
        ):
            next_action = "escalate"
            state = TriageState.DONE.value
        elif triage_result:
            next_action = "complete"
            state = TriageState.DONE.value
        else:
            # No classification yet -- the agent is still gathering
            # information. The reply is either the follow-up question
            # itself (text mode) or an open-ended prompt.
            next_action = "await_followup"
            state = TriageState.TRIAGE.value

        return TriageRunResult(
            reply=reply.strip() if reply else "",
            triage_result=triage_result,
            advice=advice,
            next_action=next_action,
            follow_up_question=follow_up_question,
            alert_requested=alert_requested,
            pii_collection_requested=pii_collection_requested,
            state=state,
            session_id=session_id,
            language=language,
            input_mode=input_mode,
            tool_calls=tool_calls,
        )

    # -- internals ----------------------------------------------------

    async def _ensure_runner(self, *, input_mode: str) -> Any:
        if self._session_service is None:
            try:
                from google.adk.sessions import InMemorySessionService
            except ImportError as exc:  # pragma: no cover - install-less envs
                raise RuntimeError(
                    "google-adk is not installed. Run `pip install -e .` or "
                    "`uv sync` inside hospital-hotline-assistant-api/."
                ) from exc
            self._session_service = InMemorySessionService()

        if input_mode == "voice":
            if self._voice_runner is None:
                self._voice_runner = self._build_runner(input_mode="voice")
            return self._voice_runner
        if self._text_runner is None:
            self._text_runner = self._build_runner(input_mode="text")
        return self._text_runner

    def _build_runner(self, *, input_mode: str) -> Any:
        try:
            from google.adk.runners import Runner
        except ImportError as exc:  # pragma: no cover - install-less envs
            raise RuntimeError(
                "google-adk is not installed. Run `pip install -e .` or "
                "`uv sync` inside hospital-hotline-assistant-api/."
            ) from exc

        agent = build_triage_agent(input_mode=input_mode)
        return Runner(
            app_name=self.app_name,
            agent=agent,
            session_service=self._session_service,
        )

    async def _ensure_session(self, *, user_id: str, session_id: str) -> None:
        """Look up the ADK session or create it on first use."""

        try:
            existing = await self._session_service.get_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id,
            )
            if existing is not None:
                return
        except Exception:  # noqa: BLE001 -- treat any lookup failure as "create it"
            pass

        await self._session_service.create_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )

    async def _invoke(
        self,
        *,
        runner: Any,
        user_id: str,
        session_id: str,
        user_message: str,
    ) -> tuple[list[dict[str, Any]], str]:
        """Drive the ADK runner and collect tool calls + final reply."""

        try:
            from google.genai import types as genai_types
        except ImportError as exc:  # pragma: no cover - install-less envs
            raise RuntimeError(
                "google-genai is not installed. Run `pip install -e .` or "
                "`uv sync` inside hospital-hotline-assistant-api/."
            ) from exc

        new_message = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)],
        )

        tool_calls: list[dict[str, Any]] = []
        final_text_parts: list[str] = []

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message,
        ):
            self._absorb_event(event, tool_calls, final_text_parts)

        return tool_calls, "".join(final_text_parts)

    @staticmethod
    def _absorb_event(
        event: Any,
        tool_calls: list[dict[str, Any]],
        final_text_parts: list[str],
    ) -> None:
        """Pull tool calls + final-response text out of one ADK event."""

        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None) or []

        for part in parts:
            function_call = getattr(part, "function_call", None)
            function_response = getattr(part, "function_response", None)

            if function_call is not None and getattr(function_call, "name", None):
                tool_calls.append(
                    {
                        "name": function_call.name,
                        "args": dict(getattr(function_call, "args", None) or {}),
                        "response": None,
                    }
                )
                continue

            if function_response is not None and getattr(function_response, "name", None):
                response_value = getattr(function_response, "response", None)
                payload: dict[str, Any] | None
                if isinstance(response_value, dict):
                    payload = response_value
                elif response_value is None:
                    payload = {}
                else:
                    payload = {"value": response_value}
                # Attach this response to the most recent unfilled call
                # of the same name. If we can't find one, append a
                # synthetic entry so downstream code still sees it.
                attached = False
                for call_entry in reversed(tool_calls):
                    if (
                        call_entry["name"] == function_response.name
                        and call_entry["response"] is None
                    ):
                        call_entry["response"] = payload
                        attached = True
                        break
                if not attached:
                    tool_calls.append(
                        {
                            "name": function_response.name,
                            "args": {},
                            "response": payload,
                        }
                    )
                continue

        is_final = False
        check = getattr(event, "is_final_response", None)
        if callable(check):
            try:
                is_final = bool(check())
            except Exception:  # noqa: BLE001
                is_final = False
        if is_final:
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    final_text_parts.append(text)

    @staticmethod
    def _error_result(
        *,
        session_id: str,
        language: str,
        input_mode: str,
        error: str,
    ) -> TriageRunResult:
        return TriageRunResult(
            reply="",
            triage_result=None,
            advice=None,
            next_action="error",
            follow_up_question=None,
            alert_requested=False,
            pii_collection_requested=False,
            state=TriageState.TRIAGE.value,
            session_id=session_id,
            language=language,
            input_mode=input_mode,
            tool_calls=[],
            error=error,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_last(tool_calls: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for call_entry in reversed(tool_calls):
        if call_entry.get("name") == name and call_entry.get("response") is not None:
            return call_entry
    return None


def _tool_response(call_entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not call_entry:
        return None
    response = call_entry.get("response")
    return response if isinstance(response, dict) else None


def _is_emergency(triage_result: dict[str, Any]) -> bool:
    if triage_result.get("is_emergency") is True:
        return True
    level = triage_result.get("level")
    return isinstance(level, int) and level in (1, 2)


# ---------------------------------------------------------------------------
# Process-wide singleton (lazy)
# ---------------------------------------------------------------------------


_DEFAULT_RUNNER: TriageRunner | None = None


def get_default_runner() -> TriageRunner:
    """Return the lazily-initialised process-wide :class:`TriageRunner`.

    The FastAPI app and any WebSocket handlers should obtain the
    runner through this helper so they share one
    :class:`InMemorySessionService` and one set of cached ADK
    :class:`Runner` instances.
    """

    global _DEFAULT_RUNNER
    if _DEFAULT_RUNNER is None:
        _DEFAULT_RUNNER = TriageRunner()
    return _DEFAULT_RUNNER


def reset_default_runner() -> None:
    """Reset the singleton. Intended for tests."""

    global _DEFAULT_RUNNER
    _DEFAULT_RUNNER = None


__all__ = [
    "DEFAULT_APP_NAME",
    "DEFAULT_USER_ID_PREFIX",
    "NextAction",
    "TriageRunResult",
    "TriageRunner",
    "get_default_runner",
    "reset_default_runner",
]
