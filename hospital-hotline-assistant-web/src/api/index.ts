import { ApiClientError, baseUrl, parseErrorBody, request } from './client';
import type {
  ChatRequestPayload,
  ChatResponsePayload,
  ConversationSummaryOut,
  DepartmentOut,
  DepartmentRecommendationCreate,
  EmergencyEventOut,
  EmergencyEventCreate,
  EmergencyPiiRequest,
  EmergencyPiiResponse,
  EmergencyTriggerOut,
  FollowUpQuestionOut,
  LanguageCode,
  MessageCreate,
  MessageOut,
  RoutingRuleOut,
  SessionCreate,
  SessionOut,
  SessionPhaseOut,
  SessionUpdate,
  SeverityAssessmentCreate,
  SttResponsePayload,
  SymptomEntryCreate,
} from './types';

async function ttsRequest(payload: { text: string; language: LanguageCode }): Promise<Blob> {
  const response = await fetch(`${baseUrl}/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new ApiClientError(response.status, await parseErrorBody(response));
  }
  return response.blob();
}

async function sttRequest(payload: {
  audio: Blob;
  language: LanguageCode;
  filename?: string;
  sessionId?: string | null;
}): Promise<SttResponsePayload> {
  const form = new FormData();
  form.append('audio', payload.audio, payload.filename ?? 'speech.webm');
  form.append('language', payload.language);
  // Pass session_id so the backend can refuse transcription (HTTP 409)
  // when the session is in PII_COLLECT phase. The audio never reaches
  // Cloud STT in that case, protecting PII from showing up in transcripts.
  if (payload.sessionId) {
    form.append('session_id', payload.sessionId);
  }

  const response = await fetch(`${baseUrl}/stt`, {
    method: 'POST',
    body: form,
  });
  if (!response.ok) {
    throw new ApiClientError(response.status, await parseErrorBody(response));
  }
  return response.json() as Promise<SttResponsePayload>;
}

export const api = {
  health: () => request<{ status: string; environment: string }>('/health'),

  createSession: (payload: SessionCreate) =>
    request<SessionOut>('/sessions', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getSession: (sessionId: string) => request<SessionOut>(`/sessions/${sessionId}`),

  updateSession: (sessionId: string, payload: SessionUpdate) =>
    request<SessionOut>(`/sessions/${sessionId}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),

  createMessage: (sessionId: string, payload: MessageCreate) =>
    request<MessageOut>(`/sessions/${sessionId}/messages`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  listMessages: (sessionId: string) =>
    request<MessageOut[]>(`/sessions/${sessionId}/messages`),

  /**
   * Primary chat endpoint. Routes through the Google ADK triage agent
   * (Vertex AI Gemini + tools) on the backend. Responses include the
   * legacy ChatResponse fields PLUS an ``adk`` block carrying the
   * runner output (state, next_action, triage classification, etc).
   */
  chat: (sessionId: string, payload: ChatRequestPayload) =>
    request<ChatResponsePayload>(`/sessions/${sessionId}/chat-adk`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  /**
   * Legacy rule-engine + direct-Vertex chat endpoint. Kept as a
   * fallback in case the ADK runner needs to be bypassed for an
   * incident. Not used in the default UI flow anymore.
   */
  chatLegacy: (sessionId: string, payload: ChatRequestPayload) =>
    request<ChatResponsePayload>(`/sessions/${sessionId}/chat`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getSessionPhase: (sessionId: string) =>
    request<SessionPhaseOut>(`/sessions/${sessionId}/phase`),

  /**
   * Submit the patient's secure PII (name, phone, address) after a
   * Level 1 emergency. This call never touches the LLM. The backend
   * generates a case_id, dispatches Slack/admin notifications, and
   * marks the session as DONE.
   */
  submitEmergencyPii: (sessionId: string, payload: EmergencyPiiRequest) =>
    request<EmergencyPiiResponse>(`/sessions/${sessionId}/emergency-pii`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  createSymptomEntry: (sessionId: string, payload: SymptomEntryCreate) =>
    request<Record<string, unknown>>(`/sessions/${sessionId}/symptoms`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  createSeverityAssessment: (sessionId: string, payload: SeverityAssessmentCreate) =>
    request<Record<string, unknown>>(`/sessions/${sessionId}/severity-assessments`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  createDepartmentRecommendation: (
    sessionId: string,
    payload: DepartmentRecommendationCreate,
  ) =>
    request<Record<string, unknown>>(`/sessions/${sessionId}/department-recommendations`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  createEmergencyEvent: (sessionId: string, payload: EmergencyEventCreate) =>
    request<Record<string, unknown>>(`/sessions/${sessionId}/emergency-events`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  listEmergencyEvents: (sessionId: string) =>
    request<EmergencyEventOut[]>(`/sessions/${sessionId}/emergency-events`),

  createFollowUpQuestion: (
    sessionId: string,
    payload: { question_text: string; reason?: string | null },
  ) =>
    request<FollowUpQuestionOut>(`/sessions/${sessionId}/follow-up-questions`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  listFollowUpQuestions: (sessionId: string) =>
    request<FollowUpQuestionOut[]>(`/sessions/${sessionId}/follow-up-questions`),

  answerFollowUpQuestion: (sessionId: string, questionId: string, answerMessageId: string) =>
    request<FollowUpQuestionOut>(
      `/sessions/${sessionId}/follow-up-questions/${questionId}/answer`,
      {
        method: 'PATCH',
        body: JSON.stringify({ answer_message_id: answerMessageId }),
      },
    ),

  listDepartments: () => request<DepartmentOut[]>('/departments'),

  listRoutingRules: () => request<RoutingRuleOut[]>('/routing-rules'),

  listEmergencyTriggers: () => request<EmergencyTriggerOut[]>('/emergency-triggers'),

  getConversationSummary: () =>
    request<ConversationSummaryOut[]>('/conversation-summary'),

  tts: (text: string, language: LanguageCode) => ttsRequest({ text, language }),

  stt: (
    audio: Blob,
    language: LanguageCode,
    options?: { filename?: string; sessionId?: string | null },
  ) =>
    sttRequest({
      audio,
      language,
      filename: options?.filename,
      sessionId: options?.sessionId,
    }),
};

export { ApiClientError, isPiiCollectionGate } from './client';

export type { MessageOut, SessionOut, ConversationSummaryOut, DepartmentOut };
