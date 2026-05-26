import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type {
  AdkNextAction,
  ChatAdkExtras,
  ChatResponsePayload,
  EmergencyPiiRequest,
  EmergencyPiiResponse,
  InputMode,
  MessageOut,
  SeverityLevel,
  TriagePhase,
} from '../api/types';
import type { AppLanguage } from '../i18n/resources';

export interface ChatAssessment {
  severity?: {
    level: SeverityLevel;
    explanation?: string;
    confidence?: number;
  };
  department?: {
    departmentId: string;
    reason?: string;
    confidence?: number;
    name?: string;
  };
  emergency?: {
    triggerId?: string;
    alertMessage: string;
    detectedSymptoms?: string[];
  };
  symptoms?: {
    rawText: string;
    bodyLocation?: string;
    durationText?: string;
  };
  followUpQuestion?: string;
  followUpReason?: string;
  alertSent?: boolean;
  modelName?: string;
  latencyMs?: number;
  assistantMessageId?: string;
  /**
   * Lifted from the ADK ``adk`` block so the UI can branch on the
   * triage phase, the requested next action (e.g. ``"collect_pii"``)
   * and the structured triage classification without reaching into
   * the raw payload at every render site.
   */
  adk?: {
    nextAction: AdkNextAction;
    state: TriagePhase;
    triageLevel?: number;
    triageColor?: string;
    symptomsSummary?: string;
    advice?: ChatAdkExtras['advice'];
    piiCollectionRequested?: boolean;
    alertRequested?: boolean;
  };
}

function toAssessment(
  payload: ChatResponsePayload,
  departmentNames: Map<string, string>,
): ChatAssessment {
  const deptId = payload.department?.department_id;
  const adk = payload.adk ?? undefined;
  return {
    severity: payload.severity
      ? {
          level: payload.severity.level,
          explanation: payload.severity.explanation,
          confidence: payload.severity.confidence,
        }
      : undefined,
    department: deptId
      ? {
          departmentId: deptId,
          reason: payload.department?.reason,
          confidence: payload.department?.confidence,
          name: departmentNames.get(deptId),
        }
      : undefined,
    emergency: payload.emergency
      ? {
          triggerId: payload.emergency.trigger_id,
          alertMessage: payload.emergency.alert_message,
          detectedSymptoms: payload.emergency.detected_symptoms,
        }
      : undefined,
    symptoms: payload.symptoms
      ? {
          rawText: payload.symptoms.raw_text,
          bodyLocation: payload.symptoms.body_location,
          durationText: payload.symptoms.duration_text,
        }
      : undefined,
    followUpQuestion:
      payload.follow_up_question ?? adk?.follow_up_question ?? undefined,
    followUpReason: payload.follow_up_reason ?? undefined,
    alertSent: payload.alert_sent ?? false,
    modelName: payload.model_name ?? undefined,
    latencyMs: payload.latency_ms ?? undefined,
    assistantMessageId: payload.assistant_message_id ?? undefined,
    adk: adk
      ? {
          nextAction: adk.next_action,
          state: adk.state,
          triageLevel: adk.triage_result?.level,
          triageColor: adk.triage_result?.color,
          symptomsSummary: adk.triage_result?.symptoms_summary,
          advice: adk.advice ?? undefined,
          piiCollectionRequested: adk.pii_collection_requested,
          alertRequested: adk.alert_requested,
        }
      : undefined,
  };
}

export function useChat(sessionId: string | null, language: AppLanguage) {
  const [messages, setMessages] = useState<MessageOut[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [assessment, setAssessment] = useState<ChatAssessment | null>(null);
  const [phase, setPhase] = useState<TriagePhase>('triage');
  const [piiReceipt, setPiiReceipt] = useState<EmergencyPiiResponse | null>(null);
  const departmentsRef = useRef<Map<string, string>>(new Map());

  useEffect(() => {
    setMessages([]);
    setAssessment(null);
    setError(null);
    setPhase('triage');
    setPiiReceipt(null);
  }, [sessionId]);

  const loadMessages = useCallback(async () => {
    if (!sessionId) return;
    setIsLoading(true);
    setError(null);
    try {
      const [msgs, departments] = await Promise.all([
        api.listMessages(sessionId),
        api.listDepartments(),
      ]);
      departmentsRef.current = new Map(
        departments.map((d) => [d.id, language === 'th' ? d.name_th ?? d.name_en : d.name_en]),
      );
      setMessages(msgs);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load messages');
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, language]);

  const refreshPhase = useCallback(async () => {
    if (!sessionId) return;
    try {
      const phaseOut = await api.getSessionPhase(sessionId);
      setPhase(phaseOut.phase);
    } catch {
      // best-effort polling; ignore network blips
    }
  }, [sessionId]);

  const sendMessage = useCallback(
    async (content: string, inputMode: InputMode = 'text') => {
      if (!sessionId || !content.trim() || isSending) return null;

      setIsSending(true);
      setError(null);

      try {
        const response = await api.chat(sessionId, {
          content: content.trim(),
          input_mode: inputMode,
          language,
          history: [],
        });

        if (departmentsRef.current.size === 0) {
          try {
            const departments = await api.listDepartments();
            departmentsRef.current = new Map(
              departments.map((d) => [
                d.id,
                language === 'th' ? d.name_th ?? d.name_en : d.name_en,
              ]),
            );
          } catch {
            // non-fatal; name will fall back to the id in the UI
          }
        }

        await loadMessages();

        const nextAssessment = toAssessment(response, departmentsRef.current);
        setAssessment(nextAssessment);
        // Trust the ADK extras when present; they are authoritative.
        if (nextAssessment.adk) {
          setPhase(nextAssessment.adk.state);
        }
        return { response, assessment: nextAssessment };
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to send message');
        return null;
      } finally {
        setIsSending(false);
      }
    },
    [sessionId, isSending, language, loadMessages],
  );

  const submitEmergencyPii = useCallback(
    async (payload: EmergencyPiiRequest) => {
      if (!sessionId) {
        throw new Error('No active session');
      }
      const receipt = await api.submitEmergencyPii(sessionId, payload);
      setPiiReceipt(receipt);
      // The backend marks the session as DONE after a successful submission.
      setPhase('done');
      // Refresh chat history so the redacted "[message withheld]" placeholders
      // and any wrap-up assistant messages are reflected in the UI.
      void loadMessages();
      return receipt;
    },
    [sessionId, loadMessages],
  );

  return {
    messages,
    isLoading,
    isSending,
    error,
    assessment,
    phase,
    piiReceipt,
    loadMessages,
    refreshPhase,
    sendMessage,
    submitEmergencyPii,
    setAssessment,
  };
}
