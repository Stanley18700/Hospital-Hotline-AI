from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

LanguageCode = Literal["th", "en"]
SessionStatus = Literal["active", "completed", "reset", "escalated"]
MessageRole = Literal["user", "assistant", "system"]
InputMode = Literal["voice", "text"]
SeverityLevel = Literal["emergency", "urgent", "general", "unknown"]


class TtsRequest(BaseModel):
    text: str
    language: LanguageCode = "en"


class SttResponse(BaseModel):
    transcript: str
    confidence: float | None = None
    language_code: str


class SessionCreate(BaseModel):
    language: LanguageCode = "th"
    user_agent: str | None = None
    ip_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionUpdate(BaseModel):
    status: SessionStatus


class SessionOut(BaseModel):
    id: UUID
    language: LanguageCode
    status: SessionStatus
    started_at: datetime
    ended_at: datetime | None = None
    user_agent: str | None = None
    ip_hash: str | None = None
    metadata: dict[str, Any]


class MessageCreate(BaseModel):
    role: MessageRole
    input_mode: InputMode | None = None
    content: str
    audio_url: str | None = None
    transcript_confidence: float | None = Field(default=None, ge=0, le=1)
    model_name: str | None = None
    response_latency_ms: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageOut(MessageCreate):
    id: UUID
    session_id: UUID
    created_at: datetime


class SymptomEntryCreate(BaseModel):
    message_id: UUID | None = None
    raw_text: str
    normalized_symptoms: list[Any] = Field(default_factory=list)
    body_location: str | None = None
    duration_text: str | None = None
    pain_score: int | None = Field(default=None, ge=0, le=10)


class SeverityAssessmentCreate(BaseModel):
    source_message_id: UUID | None = None
    severity: SeverityLevel = "unknown"
    confidence: float | None = Field(default=None, ge=0, le=1)
    explanation: str | None = None
    detected_triggers: list[Any] = Field(default_factory=list)


class DepartmentOut(BaseModel):
    id: UUID
    code: str
    name_en: str
    name_th: str | None = None
    description_en: str | None = None
    description_th: str | None = None
    is_active: bool


class RoutingRuleOut(BaseModel):
    id: UUID
    department_id: UUID
    rule_name: str
    description: str | None = None
    symptom_keywords: list[str]
    condition_json: dict[str, Any]
    severity_override: SeverityLevel | None = None
    priority: int
    is_active: bool


class EmergencyTriggerOut(BaseModel):
    id: UUID
    trigger_name: str
    description: str | None = None
    trigger_keywords: list[str]
    condition_json: dict[str, Any]
    alert_message_en: str
    alert_message_th: str | None = None
    priority: int
    is_active: bool


class DepartmentRecommendationCreate(BaseModel):
    assessment_id: UUID | None = None
    department_id: UUID
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason: str | None = None


class EmergencyEventCreate(BaseModel):
    trigger_id: UUID | None = None
    source_message_id: UUID | None = None
    detected_symptoms: list[Any] = Field(default_factory=list)
    alert_message: str


class EmergencyEventOut(EmergencyEventCreate):
    id: UUID
    session_id: UUID
    created_at: datetime


class FollowUpQuestionCreate(BaseModel):
    question_text: str
    reason: str | None = None


class FollowUpQuestionOut(BaseModel):
    id: UUID
    session_id: UUID
    question_text: str
    reason: str | None = None
    asked_at: datetime
    answer_message_id: UUID | None = None
    answered_at: datetime | None = None


class FollowUpQuestionAnswerUpdate(BaseModel):
    answer_message_id: UUID


class ChatRequest(BaseModel):
    content: str
    input_mode: InputMode = "text"
    language: LanguageCode = "en"
    history: list[Any] = Field(default_factory=list)


class ChatSeverityOut(BaseModel):
    level: SeverityLevel
    explanation: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class ChatDepartmentOut(BaseModel):
    department_id: UUID | None = None
    reason: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class ChatEmergencyOut(BaseModel):
    trigger_id: UUID | None = None
    alert_message: str
    detected_symptoms: list[str] = Field(default_factory=list)


class ChatSymptomsOut(BaseModel):
    raw_text: str
    body_location: str | None = None
    duration_text: str | None = None


class ChatResponse(BaseModel):
    reply: str
    severity: ChatSeverityOut
    department: ChatDepartmentOut | None = None
    emergency: ChatEmergencyOut | None = None
    symptoms: ChatSymptomsOut | None = None
    follow_up_question: str | None = None
    follow_up_reason: str | None = None
    model_name: str | None = None
    latency_ms: int | None = None
    alert_sent: bool = False
    assistant_message_id: UUID | None = None


# Stable vocabulary the ADK triage runner returns. Keep in sync with
# :data:`app.agent.triage_runner.NextAction`.
AdkNextAction = Literal[
    "await_followup",
    "complete",
    "escalate",
    "collect_pii",
    "error",
]


class ChatAdkExtras(BaseModel):
    """ADK-specific payload returned alongside the legacy ChatResponse fields.

    These fields are *additive* -- they let the frontend switch on a
    stable vocabulary (``next_action``) and render the secure PII form
    or escalation banner without having to re-derive state from the
    free-form ``reply`` text.
    """

    next_action: AdkNextAction
    state: str
    triage_result: dict[str, Any] | None = None
    advice: dict[str, Any] | None = None
    follow_up_question: str | None = None
    alert_requested: bool = False
    pii_collection_requested: bool = False
    error: str | None = None


class ChatAdkResponse(ChatResponse):
    """Response from the ADK chat endpoint.

    Mirrors :class:`ChatResponse` so existing clients keep working, and
    layers an :attr:`adk` block on top for the new triage signals.
    """

    adk: ChatAdkExtras


class EmergencyPiiRequest(BaseModel):
    """Secure-form payload for ``POST /sessions/{id}/emergency-pii``.

    All three core fields are REQUIRED -- the endpoint is the official
    Level 1 hand-off, so we refuse partial submissions. ``notes`` is
    optional (e.g. apartment number, gate code).

    Privacy contract: these values arrive on the API surface and are
    handed directly to the notification sink. They are NEVER passed
    to the ADK runner / Vertex, NEVER logged in the clear, and NEVER
    persisted to Postgres in the current demo build.
    """

    name: str = Field(
        min_length=1,
        max_length=200,
        description="Patient or caller's full name.",
    )
    phone: str = Field(
        min_length=3,
        max_length=64,
        pattern=r"^[\d\s\-\+\(\)\.]{3,64}$",
        description="Callback phone number. Digits, spaces, dashes, plus, parens, and dots only.",
    )
    address: str = Field(
        min_length=3,
        max_length=500,
        description="Dispatch address.",
    )
    notes: str | None = Field(
        default=None,
        max_length=500,
        description="Optional dispatcher hints (apartment number, gate code, landmarks).",
    )

    @field_validator("name", "address", "notes", "phone", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class EmergencyPiiResponse(BaseModel):
    """Acknowledgement returned to the secure form on the client.

    Deliberately minimal: only what the patient-facing UI needs to
    render the post-submission confirmation screen. No PII echoes,
    no received-field list (already in ``sessions.metadata`` for the
    admin dashboard), no internal sink-result map.
    """

    case_id: str
    alert_sent: bool
    next_instruction: str


class SessionPhaseOut(BaseModel):
    """Read-only projection of the session's triage phase.

    Used by ``GET /sessions/{id}/phase`` for the frontend to poll when
    it needs to know whether to render the chat input or the secure
    PII form. The phase mirrors :class:`TriageState` values.
    """

    session_id: UUID
    phase: str
    pii_collection_requested: bool
    pii_received_at: datetime | None = None
    updated_at: datetime | None = None


class ConversationSummaryOut(BaseModel):
    session_id: UUID
    language: LanguageCode
    status: SessionStatus
    started_at: datetime
    ended_at: datetime | None = None
    severity: SeverityLevel | None = None
    department_name_en: str | None = None
    department_name_th: str | None = None
    message_count: int
    has_alert: bool = False
    escalation_reason: str | None = None