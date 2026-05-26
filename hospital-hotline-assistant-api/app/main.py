from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import UUID
import asyncpg
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
import logging

from app.agent.triage_runner import TriageRunner, get_default_runner
from app.agent.triage_state import (
    PHASE_UPDATED_AT_METADATA_KEY,
    TriageState,
    get_session_phase,
    with_session_phase,
)
from app.config import settings
from app.database import create_pool, get_connection, record_to_dict, records_to_dicts
from app.services import TriageService
from app.services.google_stt import GoogleSttClient
from app.services.google_tts import GoogleTtsClient
from app.services.pii_handoff import (
    CompositePiiHandoffSink,
    EmergencyPiiEvent,
    EmergencyPiiPayload,
    EmergencyPiiReceipt,
    build_redacted_receipt_metadata,
    generate_case_id,
    get_default_pii_handoff_sink,
    next_instruction_for_patient,
)
from app.services.voice_pii import evaluate_voice_guard
from app.schemas import (
    ChatAdkExtras,
    ChatAdkResponse,
    ChatRequest,
    ChatResponse,
    ConversationSummaryOut,
    DepartmentOut,
    DepartmentRecommendationCreate,
    EmergencyEventCreate,
    EmergencyEventOut,
    EmergencyPiiRequest,
    EmergencyPiiResponse,
    EmergencyTriggerOut,
    FollowUpQuestionAnswerUpdate,
    FollowUpQuestionCreate,
    FollowUpQuestionOut,
    MessageCreate,
    MessageOut,
    RoutingRuleOut,
    SessionCreate,
    SessionOut,
    SessionPhaseOut,
    SessionUpdate,
    SeverityAssessmentCreate,
    SttResponse,
    SymptomEntryCreate,
    TtsRequest,
)


logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_pool = await create_pool()
    app.state.triage_service = TriageService()
    app.state.tts_client = GoogleTtsClient()
    app.state.stt_client = GoogleSttClient()
    # ADK runner is constructed eagerly so app.state always has it, but
    # the heavy google-adk imports and the Vertex client are deferred
    # until the first /chat-adk turn. That keeps boot fast and lets the
    # legacy endpoints stay healthy even if google-adk isn't installed.
    app.state.adk_runner = get_default_runner()
    # Secure emergency-PII sink. The default is a TEMPORARY in-memory
    # placeholder (see app.services.pii_handoff) so the demo dashboard
    # has something to render; production deployments should swap this
    # for an encrypted Postgres / external dispatcher implementation.
    app.state.pii_sink = get_default_pii_handoff_sink()
    try:
        yield
    finally:
        await app.state.db_pool.close()


# ---------------------------------------------------------------------------
# Severity mapping helpers (ADK 5-level <-> legacy 4-bucket schema)
# ---------------------------------------------------------------------------

def _level_to_severity(level: object) -> str:
    """Map the ADK five-level system down to the existing SeverityLevel.

    * Level 1 (Red), 2 (Orange) -> ``"emergency"``
    * Level 3 (Yellow)          -> ``"urgent"``
    * Level 4 (Green), 5 (Blue) -> ``"general"``
    * Anything else / missing   -> ``"unknown"``

    This is the bridge that lets the new endpoint reuse the existing
    :class:`ChatResponse` shape while we keep ADK as the source of
    truth for the richer five-level metadata under ``response.adk``.
    """

    if not isinstance(level, int):
        return "unknown"
    if level in (1, 2):
        return "emergency"
    if level == 3:
        return "urgent"
    if level in (4, 5):
        return "general"
    return "unknown"


def _fallback_reply(language: str) -> str:
    if language == "th":
        return "กรุณาให้รายละเอียดเพิ่มเติมเกี่ยวกับอาการเพื่อประเมินระดับความเร่งด่วน"
    return "Please provide more details about your symptoms for accurate triage."


# ---------------------------------------------------------------------------
# PII collection guard helpers
# ---------------------------------------------------------------------------

def _pii_redirect_reply(language: str) -> str:
    """Canned, language-aware redirect shown while in PII_COLLECT phase.

    This text never depends on the LLM and is never assembled from
    patient input, so it cannot leak PII or trigger another model call.
    """

    if language == "th":
        return (
            "เจ้าหน้าที่ฉุกเฉินกำลังดูแลกรณีของคุณ "
            "โปรดใช้แบบฟอร์มที่ปลอดภัยบนหน้าจอเพื่อกรอกชื่อ เบอร์โทรศัพท์ และที่อยู่ของคุณ"
        )
    return (
        "An emergency responder is being notified for your case. "
        "Please use the secure form on the call screen to share your name, phone number, and address."
    )


def _pii_redacted_placeholder(language: str) -> str:
    """Placeholder content stored in ``messages`` while in PII_COLLECT phase.

    We log the *fact* of a patient turn so the chat history keeps its
    turn count and the admin dashboard can audit the redirect, but we
    discard the raw content because we cannot rule out that the
    patient typed PII into the chat instead of the secure form.
    """

    if language == "th":
        return "[ข้อความถูกระงับ: กำลังเก็บข้อมูลผ่านแบบฟอร์มที่ปลอดภัย]"
    return "[message withheld: secure PII form active]"


async def _set_session_phase(
    connection: asyncpg.Connection,
    *,
    session_id: UUID,
    phase: TriageState,
    current_metadata: dict[str, Any] | None,
    extra: dict[str, Any] | None = None,
    completed: bool = False,
) -> dict[str, Any]:
    """Update ``sessions.metadata.triage_phase`` (and optionally ``status``).

    Returns the new metadata dict that was written, so callers can
    keep their in-memory view in sync without re-reading the row.
    """

    new_metadata = with_session_phase(current_metadata, phase)
    if extra:
        new_metadata.update(extra)

    if completed:
        await connection.execute(
            """
            UPDATE sessions
            SET status = 'completed',
                ended_at = COALESCE(ended_at, NOW()),
                metadata = $2::jsonb
            WHERE id = $1
            """,
            session_id,
            new_metadata,
        )
    else:
        await connection.execute(
            "UPDATE sessions SET metadata = $2::jsonb WHERE id = $1",
            session_id,
            new_metadata,
        )
    return new_metadata


async def _build_pii_redirect_response(
    *,
    connection: asyncpg.Connection,
    session_id: UUID,
    payload: ChatRequest,
) -> ChatAdkResponse:
    """Short-circuit reply while the session is in :attr:`TriageState.PII_COLLECT`.

    Critical property: the patient's raw ``payload.content`` is NEVER
    persisted and NEVER passed to the ADK runner / Vertex. We insert
    a placeholder user message so the chat log keeps its structure,
    then return a canned redirect reply with ``next_action=collect_pii``.
    The frontend uses that signal to render the secure form instead of
    a normal chat bubble.
    """

    await connection.fetchrow(
        """
        INSERT INTO messages (session_id, role, input_mode, content, metadata)
        VALUES ($1, 'user', $2, $3, $4::jsonb)
        RETURNING id
        """,
        session_id,
        payload.input_mode,
        _pii_redacted_placeholder(payload.language),
        {"redacted": True, "reason": "pii_collect_phase"},
    )

    redirect_reply = _pii_redirect_reply(payload.language)
    msg_assistant = await connection.fetchrow(
        """
        INSERT INTO messages (
            session_id, role, input_mode, content, model_name, metadata
        )
        VALUES ($1, 'assistant', NULL, $2, $3, $4::jsonb)
        RETURNING id
        """,
        session_id,
        redirect_reply,
        "adk:guard",
        {"source": "adk_guard", "phase": TriageState.PII_COLLECT.value},
    )

    extras = ChatAdkExtras(
        next_action="collect_pii",
        state=TriageState.PII_COLLECT.value,
        triage_result=None,
        advice=None,
        follow_up_question=None,
        alert_requested=False,
        pii_collection_requested=True,
        error=None,
    )

    return ChatAdkResponse(
        reply=redirect_reply,
        severity={
            "level": "emergency",
            "explanation": "PII collection in progress",
            "confidence": None,
        },
        department=None,
        emergency={
            "trigger_id": None,
            "alert_message": redirect_reply,
            "detected_symptoms": [],
        },
        symptoms=None,
        follow_up_question=None,
        follow_up_reason=None,
        model_name="adk:guard",
        latency_ms=0,
        alert_sent=False,
        assistant_message_id=msg_assistant["id"] if msg_assistant else None,
        adk=extras,
    )

app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(asyncpg.ForeignKeyViolationError)
async def foreign_key_violation_handler(request: Request, exc: asyncpg.ForeignKeyViolationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "Referenced record does not exist. Check session_id, message_id, assessment_id, department_id, or trigger_id."},
    )

@app.exception_handler(asyncpg.UniqueViolationError)
async def unique_violation_handler(request: Request, exc: asyncpg.UniqueViolationError):
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": "Record already exists."},
    )

@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "status": "running",
        "docs": "/docs",
    }

@app.get("/health")
async def health(connection: asyncpg.Connection = Depends(get_connection)) -> dict[str, str]:
    await connection.fetchval("SELECT 1")
    return {"status": "ok", "environment": settings.environment}

@app.post("/sessions", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(payload: SessionCreate, connection: asyncpg.Connection = Depends(get_connection)):
    record = await connection.fetchrow(
        """
        INSERT INTO sessions (language, user_agent, ip_hash, metadata)
        VALUES ($1, $2, $3, $4::jsonb)
        RETURNING *
        """,
        payload.language,
        payload.user_agent,
        payload.ip_hash,
        payload.metadata,
    )
    return record_to_dict(record)

@app.get("/sessions/{session_id}", response_model=SessionOut)
async def get_session(session_id: UUID, connection: asyncpg.Connection = Depends(get_connection)):
    record = await connection.fetchrow("SELECT * FROM sessions WHERE id = $1", session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return record_to_dict(record)

@app.patch("/sessions/{session_id}", response_model=SessionOut)
async def update_session(session_id: UUID, payload: SessionUpdate, connection: asyncpg.Connection = Depends(get_connection)):
    ended_sql = "NOW()" if payload.status in {"completed", "reset", "escalated"} else "ended_at"
    record = await connection.fetchrow(
        f"""
        UPDATE sessions
        SET status = $2, ended_at = {ended_sql}
        WHERE id = $1
        RETURNING *
        """,
        session_id,
        payload.status,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return record_to_dict(record)

@app.post("/sessions/{session_id}/messages", response_model=MessageOut, status_code=status.HTTP_201_CREATED)
async def create_message(session_id: UUID, payload: MessageCreate, connection: asyncpg.Connection = Depends(get_connection)):
    record = await connection.fetchrow(
        """
        INSERT INTO messages (
            session_id, role, input_mode, content, audio_url, transcript_confidence,
            model_name, response_latency_ms, metadata
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
        RETURNING *
        """,
        session_id,
        payload.role,
        payload.input_mode,
        payload.content,
        payload.audio_url,
        payload.transcript_confidence,
        payload.model_name,
        payload.response_latency_ms,
        payload.metadata,
    )
    return record_to_dict(record)

@app.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
async def list_messages(session_id: UUID, connection: asyncpg.Connection = Depends(get_connection)):
    records = await connection.fetch(
        "SELECT * FROM messages WHERE session_id = $1 ORDER BY created_at ASC",
        session_id,
    )
    return records_to_dicts(records)

@app.post("/sessions/{session_id}/chat", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
async def chat(
    session_id: UUID,
    payload: ChatRequest,
    request: Request,
    connection: asyncpg.Connection = Depends(get_connection),
):
    triage_service: TriageService = request.app.state.triage_service
    try:
        result, assistant_message = await triage_service.process_chat(
            connection=connection,
            session_id=str(session_id),
            language=payload.language,
            input_mode=payload.input_mode,
            content=payload.content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ChatResponse(
        reply=result.reply,
        severity={
            "level": result.severity_level,
            "explanation": result.severity_explanation,
            "confidence": result.severity_confidence,
        },
        department={
            "department_id": result.department_id,
            "reason": result.department_reason,
            "confidence": result.department_confidence,
        }
        if result.department_id
        else None,
        emergency={
            "trigger_id": result.emergency_trigger_id,
            "alert_message": result.emergency_alert_message,
            "detected_symptoms": result.detected_symptoms,
        }
        if result.severity_level == "emergency"
        else None,
        symptoms={
            "raw_text": payload.content,
            "body_location": None,
            "duration_text": None,
        },
        follow_up_question=result.follow_up_question,
        follow_up_reason=result.follow_up_reason,
        model_name=result.model_name,
        latency_ms=result.latency_ms,
        alert_sent=result.alert_sent,
        assistant_message_id=assistant_message.get("id"),
    )


@app.post(
    "/sessions/{session_id}/chat-adk",
    response_model=ChatAdkResponse,
    status_code=status.HTTP_201_CREATED,
)
async def chat_adk(
    session_id: UUID,
    payload: ChatRequest,
    request: Request,
    connection: asyncpg.Connection = Depends(get_connection),
):
    """ADK-backed triage turn.

    This endpoint runs in parallel with :func:`chat`. It uses the Google
    ADK runner (see :mod:`app.agent.triage_runner`) instead of
    :class:`TriageService` and adds an ``adk`` payload to the response
    alongside the legacy :class:`ChatResponse` fields.

    The endpoint persists user / assistant messages in ``messages`` so
    the existing ``GET /sessions/{id}/messages`` chat-history flow keeps
    working. Severity / department / emergency-event DB writes are left
    to a future ``AdkTriageService`` so this step stays additive.

    Failure behaviour: if google-adk is not installed or the runner
    raises, the runner returns ``next_action="error"`` and this endpoint
    still produces a valid ``ChatAdkResponse`` with a friendly fallback
    reply, so the chat UI never hangs.
    """

    session_row = await connection.fetchrow(
        "SELECT id, metadata FROM sessions WHERE id = $1", session_id
    )
    if session_row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    current_metadata = dict(session_row["metadata"] or {})
    current_phase = get_session_phase(current_metadata)

    # PII collection guard: if the session is already in the secure
    # hand-off window, do not pass the patient's text into the LLM and
    # do not store it as a free-form message. Return a canned redirect.
    if current_phase is TriageState.PII_COLLECT:
        return await _build_pii_redirect_response(
            connection=connection, session_id=session_id, payload=payload
        )

    msg_user = await connection.fetchrow(
        """
        INSERT INTO messages (session_id, role, input_mode, content, metadata)
        VALUES ($1, 'user', $2, $3, '{}'::jsonb)
        RETURNING *
        """,
        session_id,
        payload.input_mode,
        payload.content,
    )

    runner: TriageRunner = request.app.state.adk_runner
    start = perf_counter()
    run_result = await runner.run(
        session_id=str(session_id),
        user_message=payload.content,
        language=payload.language,
        input_mode=payload.input_mode,
    )
    latency_ms = int((perf_counter() - start) * 1000)

    triage_result = run_result.triage_result or {}
    advice = run_result.advice or {}
    severity_level = _level_to_severity(triage_result.get("level"))
    severity_explanation = triage_result.get("reasoning") or run_result.error
    model_name = f"adk:{settings.google_model_name}"

    reply = (
        run_result.reply
        or run_result.follow_up_question
        or _fallback_reply(payload.language)
    )

    msg_assistant = await connection.fetchrow(
        """
        INSERT INTO messages (
            session_id, role, input_mode, content, model_name, response_latency_ms, metadata
        )
        VALUES ($1, 'assistant', NULL, $2, $3, $4, $5::jsonb)
        RETURNING *
        """,
        session_id,
        reply,
        model_name,
        latency_ms,
        {
            "source": "adk",
            "next_action": run_result.next_action,
            "state": run_result.state,
            "alert_requested": run_result.alert_requested,
            "pii_collection_requested": run_result.pii_collection_requested,
        },
    )

    # Phase transitions driven by the runner's normalised next_action.
    # We only transition forward; the legacy /chat endpoint never
    # touches the phase key, so the two endpoints remain isolated.
    if run_result.pii_collection_requested:
        # Capture clinical triage context (NOT PII) so the dedicated
        # /emergency-pii endpoint can hand richer detail to the
        # notification sink without re-asking the LLM. Level, color,
        # and symptoms_summary all come from the agent's structured
        # classify_triage output -- patient name / phone / address
        # are never present in this scope.
        triage_context_extra: dict[str, Any] = {
            "pii_requested_at": datetime.now(timezone.utc).isoformat(),
        }
        if isinstance(triage_result.get("level"), int):
            triage_context_extra["triage_level"] = triage_result["level"]
        if isinstance(triage_result.get("color"), str):
            triage_context_extra["triage_color"] = triage_result["color"]
        if isinstance(triage_result.get("symptoms_summary"), str):
            triage_context_extra["symptoms_summary"] = triage_result["symptoms_summary"]
        current_metadata = await _set_session_phase(
            connection,
            session_id=session_id,
            phase=TriageState.PII_COLLECT,
            current_metadata=current_metadata,
            extra=triage_context_extra,
        )
    elif run_result.next_action == "complete":
        current_metadata = await _set_session_phase(
            connection,
            session_id=session_id,
            phase=TriageState.DONE,
            current_metadata=current_metadata,
        )

    emergency_block = None
    if run_result.alert_requested or severity_level == "emergency":
        emergency_block = {
            "trigger_id": None,
            "alert_message": (
                advice.get("urgency_statement")
                or triage_result.get("response_time")
                or ("กรุณาติดต่อเจ้าหน้าที่ทันที" if payload.language == "th" else "Please contact medical staff immediately")
            ),
            "detected_symptoms": [payload.content] if payload.content else [],
        }

    extras = ChatAdkExtras(
        next_action=run_result.next_action,
        state=run_result.state,
        triage_result=run_result.triage_result,
        advice=run_result.advice,
        follow_up_question=run_result.follow_up_question,
        alert_requested=run_result.alert_requested,
        pii_collection_requested=run_result.pii_collection_requested,
        error=run_result.error,
    )

    return ChatAdkResponse(
        reply=reply,
        severity={
            "level": severity_level,
            "explanation": severity_explanation,
            "confidence": None,
        },
        department=None,
        emergency=emergency_block,
        symptoms={
            "raw_text": payload.content,
            "body_location": None,
            "duration_text": None,
        },
        follow_up_question=run_result.follow_up_question,
        follow_up_reason=None,
        model_name=model_name,
        latency_ms=latency_ms,
        alert_sent=run_result.alert_requested,
        assistant_message_id=msg_assistant.get("id") if msg_assistant else None,
        adk=extras,
    )


@app.get("/sessions/{session_id}/phase", response_model=SessionPhaseOut)
async def get_session_phase_endpoint(
    session_id: UUID,
    connection: asyncpg.Connection = Depends(get_connection),
):
    """Return the session's current triage phase.

    The frontend can poll this (or read the ``adk.state`` field on the
    last chat-adk response) to decide whether to render the chat input
    or the secure PII form.
    """

    session_row = await connection.fetchrow(
        "SELECT id, metadata FROM sessions WHERE id = $1", session_id
    )
    if session_row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    metadata = dict(session_row["metadata"] or {})
    phase = get_session_phase(metadata)

    def _parse_dt(value: Any) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    return SessionPhaseOut(
        session_id=session_id,
        phase=phase.value,
        pii_collection_requested=phase is TriageState.PII_COLLECT,
        pii_received_at=_parse_dt(metadata.get("pii_received_at")),
        updated_at=_parse_dt(metadata.get(PHASE_UPDATED_AT_METADATA_KEY)),
    )


@app.post(
    "/sessions/{session_id}/emergency-pii",
    response_model=EmergencyPiiResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_emergency_pii(
    session_id: UUID,
    payload: EmergencyPiiRequest,
    request: Request,
    connection: asyncpg.Connection = Depends(get_connection),
):
    """Secure emergency-PII submission for a Level 1 (Red) case.

    SECURITY CONTRACT
    -----------------

    The patient's ``name`` / ``phone`` / ``address`` arrive on this
    endpoint and follow a single, deliberate path:

        HTTP body
            -> EmergencyPiiRequest (pydantic validation)
            -> EmergencyPiiPayload (frozen dataclass, request-local)
            -> EmergencyPiiEvent (sink contract)
            -> CompositePiiHandoffSink.dispatch(event)
                   |
                   |--> InMemoryPiiHandoffSink (demo storage)
                   |--> SlackPiiHandoffSink (-> SlackNotifier
                   |       .send_emergency_dispatch -> Slack webhook)
            -> the dataclass references go out of scope at return

    Things that DO NOT happen anywhere in this handler:

    * The ADK runner is never invoked. ``app.state.adk_runner`` is not
      touched here. ``Runner.run_async(...)`` is not called.
    * The PII values are never used to format a log line. We log only
      populated field *names* (see :class:`InMemoryPiiHandoffSink`).
    * The PII values are never written to Postgres. Only a redacted
      :class:`EmergencyPiiReceipt` (case id + field names + timestamp
      + per-sink success map) lands in ``sessions.metadata`` via
      ``_set_session_phase``.

    BEHAVIOURAL CONTRACT
    --------------------

    * Validates the session exists (404 otherwise).
    * Validates the payload via Pydantic (422 on missing / too-long
      / malformed phone). Empty/whitespace-only inputs are rejected
      by the ``min_length`` constraints after the strip validator.
    * Generates a unique ``case_id`` (``EM-YYYYMMDD-XXXXXXXX``).
    * Fans out to every registered sink in parallel via the
      composite. If Slack is misconfigured the case_id is still
      returned (with ``alert_sent=false``) so the operator sees the
      submission and can re-dispatch manually.
    * Transitions the session phase to :attr:`TriageState.DONE` and
      marks ``sessions.status = 'completed'`` atomically with the
      metadata write.

    RESPONSE SHAPE
    --------------

    Deliberately minimal: ``case_id``, ``alert_sent``,
    ``next_instruction``. No PII echoes, no internal sink-result
    map. Anything richer lives in ``sessions.metadata`` for the
    admin dashboard to read.
    """

    session_row = await connection.fetchrow(
        "SELECT id, language, metadata FROM sessions WHERE id = $1", session_id
    )
    if session_row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    language = session_row["language"]
    current_metadata = dict(session_row["metadata"] or {})
    received_at = datetime.now(timezone.utc)
    case_id = generate_case_id(now=received_at)

    # Clinical context pulled from sessions.metadata if a prior
    # /chat-adk turn populated it during the PII_COLLECT transition.
    # These fields are clinical metadata, NEVER patient identifiers.
    triage_level_raw = current_metadata.get("triage_level")
    triage_level = triage_level_raw if isinstance(triage_level_raw, int) else None
    triage_color_raw = current_metadata.get("triage_color")
    triage_color = triage_color_raw if isinstance(triage_color_raw, str) else None
    symptoms_raw = current_metadata.get("symptoms_summary")
    symptoms_summary = symptoms_raw if isinstance(symptoms_raw, str) else None

    # Build the in-process, request-local payload. This is the ONLY
    # place the raw values live after Pydantic finishes validating.
    secure_payload = EmergencyPiiPayload(
        name=payload.name,
        phone=payload.phone,
        address=payload.address,
        notes=payload.notes or None,
    )
    event = EmergencyPiiEvent(
        case_id=case_id,
        session_id=str(session_id),
        language=language,
        triage_level=triage_level,
        triage_color=triage_color,
        symptoms_summary=symptoms_summary,
        received_at=received_at,
        payload=secure_payload,
    )

    # Fan-out to every configured sink (in-memory + Slack today; add
    # Postgres / paging in pii_handoff.get_default_pii_handoff_sink
    # without touching this handler).
    sink: CompositePiiHandoffSink = request.app.state.pii_sink
    outcome = await sink.dispatch(event)

    received_fields = [
        name
        for name, value in (
            ("name", secure_payload.name),
            ("phone", secure_payload.phone),
            ("address", secure_payload.address),
            ("notes", secure_payload.notes),
        )
        if value
    ]
    receipt = EmergencyPiiReceipt(
        case_id=case_id,
        session_id=str(session_id),
        received_at=received_at,
        received_fields=received_fields,
        notification_dispatched=outcome.notification_sent,
    )

    # Drop the raw event references from this scope immediately after
    # the redacted receipt is built. The sinks already hold their own
    # references; nothing below this line should still hold PII.
    del event
    del secure_payload

    redacted_extra = build_redacted_receipt_metadata(receipt, outcome=outcome)
    await _set_session_phase(
        connection,
        session_id=session_id,
        phase=TriageState.DONE,
        current_metadata=current_metadata,
        extra=redacted_extra,
        completed=True,
    )

    return EmergencyPiiResponse(
        case_id=case_id,
        alert_sent=outcome.notification_sent,
        next_instruction=next_instruction_for_patient(
            language=language,
            case_id=case_id,
            alert_sent=outcome.notification_sent,
        ),
    )


@app.post("/sessions/{session_id}/symptoms", status_code=status.HTTP_201_CREATED)
async def create_symptom_entry(session_id: UUID, payload: SymptomEntryCreate, connection: asyncpg.Connection = Depends(get_connection)):
    record = await connection.fetchrow(
        """
        INSERT INTO symptom_entries (
            session_id, message_id, raw_text, normalized_symptoms,
            body_location, duration_text, pain_score
        )
        VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
        RETURNING *
        """,
        session_id,
        payload.message_id,
        payload.raw_text,
        payload.normalized_symptoms,
        payload.body_location,
        payload.duration_text,
        payload.pain_score,
    )
    return record_to_dict(record)

@app.post("/sessions/{session_id}/severity-assessments", status_code=status.HTTP_201_CREATED)
async def create_severity_assessment(
    session_id: UUID,
    payload: SeverityAssessmentCreate,
    connection: asyncpg.Connection = Depends(get_connection),
    ):
    record = await connection.fetchrow(
        """
        INSERT INTO severity_assessments (
            session_id, source_message_id, severity, confidence, explanation, detected_triggers
        )
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        RETURNING *
        """,
        session_id,
        payload.source_message_id,
        payload.severity,
        payload.confidence,
        payload.explanation,
        payload.detected_triggers,
    )
    return record_to_dict(record)

@app.post("/sessions/{session_id}/follow-up-questions", response_model=FollowUpQuestionOut, status_code=status.HTTP_201_CREATED)
async def create_follow_up_question(
    session_id: UUID,
    payload: FollowUpQuestionCreate,
    connection: asyncpg.Connection = Depends(get_connection),
):
    record = await connection.fetchrow(
        """
        INSERT INTO follow_up_questions (session_id, question_text, reason)
        VALUES ($1, $2, $3)
        RETURNING *
        """,
        session_id,
        payload.question_text,
        payload.reason,
    )
    return record_to_dict(record)

@app.get("/sessions/{session_id}/follow-up-questions", response_model=list[FollowUpQuestionOut])
async def list_follow_up_questions(
    session_id: UUID,
    connection: asyncpg.Connection = Depends(get_connection),
):
    records = await connection.fetch(
        """
        SELECT *
        FROM follow_up_questions
        WHERE session_id = $1
        ORDER BY asked_at ASC
        """,
        session_id,
    )
    return records_to_dicts(records)

@app.patch("/sessions/{session_id}/follow-up-questions/{question_id}/answer", response_model=FollowUpQuestionOut)
async def answer_follow_up_question(
    session_id: UUID,
    question_id: UUID,
    payload: FollowUpQuestionAnswerUpdate,
    connection: asyncpg.Connection = Depends(get_connection),
):
    record = await connection.fetchrow(
        """
        UPDATE follow_up_questions
        SET answer_message_id = $3, answered_at = NOW()
        WHERE id = $1 AND session_id = $2
        RETURNING *
        """,
        question_id,
        session_id,
        payload.answer_message_id,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Follow-up question not found")
    return record_to_dict(record)

@app.get("/departments", response_model=list[DepartmentOut])
async def list_departments(connection: asyncpg.Connection = Depends(get_connection)):
    records = await connection.fetch(
        "SELECT * FROM departments WHERE is_active = TRUE ORDER BY name_en ASC"
    )
    return records_to_dicts(records)

@app.get("/routing-rules", response_model=list[RoutingRuleOut])
async def list_routing_rules(connection: asyncpg.Connection = Depends(get_connection)):
    records = await connection.fetch(
        "SELECT * FROM routing_rules WHERE is_active = TRUE ORDER BY priority ASC, rule_name ASC"
    )
    return records_to_dicts(records)

@app.get("/emergency-triggers", response_model=list[EmergencyTriggerOut])
async def list_emergency_triggers(connection: asyncpg.Connection = Depends(get_connection)):
    records = await connection.fetch(
        "SELECT * FROM emergency_triggers WHERE is_active = TRUE ORDER BY priority ASC, trigger_name ASC"
    )
    return records_to_dicts(records)

@app.post("/sessions/{session_id}/department-recommendations", status_code=status.HTTP_201_CREATED)
async def create_department_recommendation(
    session_id: UUID,
    payload: DepartmentRecommendationCreate,
    connection: asyncpg.Connection = Depends(get_connection),
):
    record = await connection.fetchrow(
        """
        INSERT INTO department_recommendations (
            session_id, assessment_id, department_id, confidence, reason
        )
        VALUES ($1, $2, $3, $4, $5)
        RETURNING *
        """,
        session_id,
        payload.assessment_id,
        payload.department_id,
        payload.confidence,
        payload.reason,
    )
    return record_to_dict(record)

@app.post("/sessions/{session_id}/emergency-events", status_code=status.HTTP_201_CREATED)
async def create_emergency_event(
    session_id: UUID,
    payload: EmergencyEventCreate,
    connection: asyncpg.Connection = Depends(get_connection),
):
    record = await connection.fetchrow(
        """
        INSERT INTO emergency_events (
            session_id, trigger_id, source_message_id, detected_symptoms, alert_message
        )
        VALUES ($1, $2, $3, $4::jsonb, $5)
        RETURNING *
        """,
        session_id,
        payload.trigger_id,
        payload.source_message_id,
        payload.detected_symptoms,
        payload.alert_message,
    )
    return record_to_dict(record)

@app.get("/sessions/{session_id}/emergency-events", response_model=list[EmergencyEventOut])
async def list_emergency_events(
    session_id: UUID,
    connection: asyncpg.Connection = Depends(get_connection),
):
    records = await connection.fetch(
        """
        SELECT *
        FROM emergency_events
        WHERE session_id = $1
        ORDER BY created_at DESC
        """,
        session_id,
    )
    return records_to_dicts(records)

@app.post("/tts")
async def text_to_speech(payload: TtsRequest, request: Request):
    """Synthesize speech for the given text.

    Streams Gemini TTS audio chunk-by-chunk so the browser can begin
    playback before synthesis finishes (typically saves 2-4 s of
    pre-roll latency). If streaming setup fails for any reason
    (SDK missing, network error, etc.), we transparently fall back
    to the blocking :meth:`GoogleTtsClient.synthesize` path so the
    caller always gets a valid WAV body.
    """

    tts_client: GoogleTtsClient = request.app.state.tts_client

    # Validate up-front so empty input still returns the documented
    # 400. We do this before opening the stream because a
    # ``StreamingResponse`` can no longer change its status code once
    # the first byte has been flushed to the client.
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    try:
        audio_stream = tts_client.synthesize_stream(
            text=payload.text,
            language=payload.language,
        )
        return StreamingResponse(
            audio_stream,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="speech.wav"',
                "Cache-Control": "no-cache",
            },
        )
    except Exception:
        logger.exception("TTS streaming failed, falling back to non-streaming synthesize()")

    try:
        audio_bytes = await tts_client.synthesize(
            text=payload.text,
            language=payload.language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="speech.wav"'},
    )


@app.post("/stt", response_model=SttResponse)
async def speech_to_text(
    request: Request,
    audio: UploadFile = File(..., description="Short audio clip from MediaRecorder"),
    language: str = Form("en"),
    session_id: str | None = Form(
        None,
        description=(
            "Optional session UUID. When supplied, the endpoint honours the "
            "triage-phase guard: if the session is in pii_collect, the audio "
            "is NOT transcribed and the endpoint returns 409 with "
            "next_action='collect_pii'. When omitted, behaviour is unchanged."
        ),
    ),
    connection: asyncpg.Connection = Depends(get_connection),
):
    """Transcribe a short audio clip. Returns the recognized text.

    Voice-flow contract (architecturally enforced):

    * ``TRIAGE``      -> audio is transcribed normally; the frontend
                          then posts the transcript to ``/chat-adk``
                          which routes through the ADK runner.
    * ``PII_COLLECT`` -> audio is REFUSED here. The endpoint returns
                          HTTP 409 with a structured ``next_action``
                          signal so the frontend can switch to the
                          secure PII form. **Cloud STT is not
                          called**, so no PII-bearing transcript is
                          ever generated, logged, or returned -- the
                          audio buffer is discarded server-side after
                          the read.
    * ``DONE``        -> audio is transcribed normally (re-engagement
                          is treated as a fresh triage).

    The ``session_id`` parameter is OPTIONAL so existing standalone
    callers (admin tools, the legacy chat path) keep working. New
    voice-call flows should always pass it; the docstring on
    :func:`app.services.voice_pii.voice_input_is_allowed` is the
    single source of truth for the guard rule.

    Note: ``/tts`` does not need a phase guard. TTS is output-only
    and is generated from backend-controlled strings; it cannot
    smuggle patient-typed PII into the model surface.
    """

    if language not in {"en", "th"}:
        raise HTTPException(status_code=400, detail="language must be 'en' or 'th'")

    # Resolve and apply the phase guard BEFORE consuming the upload or
    # invoking Cloud STT. We deliberately short-circuit early: the
    # raw audio buffer goes out of scope on function return without
    # ever being transcribed.
    if session_id is not None:
        try:
            session_uuid = UUID(session_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="session_id must be a valid UUID"
            ) from exc
        session_row = await connection.fetchrow(
            "SELECT id, metadata FROM sessions WHERE id = $1", session_uuid
        )
        if session_row is None:
            raise HTTPException(status_code=404, detail="Session not found")

        phase = get_session_phase(dict(session_row["metadata"] or {}))
        decision = evaluate_voice_guard(phase)
        if not decision.allowed:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": decision.reason,
                    "next_action": decision.next_action,
                    "state": phase.value,
                    "message": (
                        "Voice transcription is paused during secure PII "
                        "collection. Submit the patient's name, phone, and "
                        "address through POST /sessions/{id}/emergency-pii "
                        "instead. The audio clip was not transcribed."
                    ),
                },
            )

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="audio file is empty")

    stt_client: GoogleSttClient = request.app.state.stt_client
    try:
        result = await stt_client.transcribe(
            audio_bytes=audio_bytes,
            language=language,
            mime_type=audio.content_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return SttResponse(
        transcript=result.transcript,
        confidence=result.confidence,
        language_code=result.language_code,
    )


@app.get("/conversation-summary", response_model=list[ConversationSummaryOut])
async def conversation_summary(connection: asyncpg.Connection = Depends(get_connection)):
    records = await connection.fetch(
        """
        SELECT
            cs.*,
            COALESCE((s.metadata->>'alert_sent')::boolean, FALSE) AS has_alert,
            s.metadata->>'escalation_reason' AS escalation_reason
        FROM conversation_summary cs
        JOIN sessions s ON s.id = cs.session_id
        ORDER BY cs.started_at DESC
        LIMIT 100
        """
    )
    return records_to_dicts(records)
