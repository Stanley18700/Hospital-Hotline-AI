import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { api } from '../api';
import { EmergencyBanner } from '../components/EmergencyBanner';
import { Layout } from '../components/Layout';
import { SecurePiiForm } from '../components/SecurePiiForm';
import { useChat } from '../hooks/useChat';
import { useLanguage, useSessionStorage } from '../hooks/useSession';
import { useVoiceCall } from '../hooks/useVoiceCall';

function PhoneIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M6.6 10.8c1.4 2.8 3.8 5.1 6.6 6.6l2.2-2.2c.3-.3.8-.4 1.2-.2 1 .4 2 .7 3 .9.4.1.7.4.7.9V20c0 .6-.4 1-1 1C10.1 21 3 13.9 3 5c0-.6.4-1 1-1h3.5c.5 0 .9.3 1 .8.2 1 .5 2 1 3 .1.4 0 .9-.3 1.2L6.6 10.8z" />
    </svg>
  );
}

function HangUpIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M12 8c-3 0-5.7 1-7.9 2.5-.4.3-.6.7-.6 1.2v2.1c0 .6.5 1 1 1h3.5c.6 0 1-.5 1-1v-1.5c1.2-.4 2.6-.6 3.9-.6s2.8.2 3.9.6V14c0 .6.5 1 1 1H21c.6 0 1-.5 1-1v-2.1c0-.5-.2-.9-.6-1.2C17.7 9 14.9 8 12 8z"
        transform="rotate(135 12 12)"
      />
    </svg>
  );
}

function MicIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2z" />
    </svg>
  );
}

function MicOffIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M19 11h-1.7c0 .74-.16 1.43-.43 2.05l1.23 1.23A6.94 6.94 0 0 0 19 11zm-4.02.17c0-.06.02-.11.02-.17V6a3 3 0 0 0-5.94-.59l5.92 5.92zM4.27 3 3 4.27l6.01 6.01V11a3 3 0 0 0 4.47 2.61l1.66 1.66A4.9 4.9 0 0 1 12 16a5 5 0 0 1-5-5H5a7 7 0 0 0 6 6.92V21h2v-3.08c.91-.13 1.77-.45 2.55-.9L19.73 21 21 19.73 4.27 3z" />
    </svg>
  );
}

export function CallPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { language, setLanguage } = useLanguage();
  const { sessionId, setSessionId } = useSessionStorage();

  const { assessment, phase, piiReceipt, sendMessage, submitEmergencyPii } = useChat(
    sessionId,
    language,
  );

  const greeting = t('callGreeting');

  const voiceCall = useVoiceCall({
    language,
    sessionId,
    initialGreeting: greeting,
    onGreeting: (text) => {
      if (!sessionId) return;
      // Persist the greeting as an assistant message so it shows up if the
      // user later switches to the chat view or admin dashboard. Best-effort.
      void api
        .createMessage(sessionId, {
          role: 'assistant',
          content: text,
          model_name: 'hotline-greeting-script',
        })
        .catch(() => undefined);
    },
    onTranscript: async (transcript) => {
      const result = await sendMessage(transcript, 'voice');
      return result?.response.reply ?? null;
    },
  });

  const callActive = voiceCall.state !== 'idle' && voiceCall.state !== 'error';
  const piiRequired = phase === 'pii_collect';
  const autoStartedRef = useRef(false);
  const [autoStartBlocked, setAutoStartBlocked] = useState(false);

  // The voice loop has two ways to learn the call must yield to the
  // secure form: (1) the /stt 409 phase guard fires mid-turn (handled
  // inside the hook), or (2) the previous chat-adk response surfaced
  // ``next_action="collect_pii"`` and useChat updated phase. Handle
  // case (2) here so the mic never re-opens after the assistant's
  // confirmation reply finishes playing.
  useEffect(() => {
    if (piiRequired && callActive) {
      voiceCall.requirePii();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [piiRequired, callActive]);

  useEffect(() => {
    if (!sessionId) {
      navigate('/');
    }
  }, [sessionId, navigate]);

  useEffect(() => {
    if (!sessionId || !voiceCall.supported || autoStartedRef.current) return;
    autoStartedRef.current = true;
    void (async () => {
      try {
        await voiceCall.start();
      } catch {
        setAutoStartBlocked(true);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, voiceCall.supported]);

  useEffect(() => {
    return () => {
      voiceCall.end();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const statusLabel = useMemo(() => {
    switch (voiceCall.state) {
      case 'starting':
        return t('callStateStarting');
      case 'greeting':
        return t('callStateGreeting');
      case 'listening':
        return t('callStateListening');
      case 'uploading':
        return t('callStateUploading');
      case 'thinking':
        return t('callStateThinking');
      case 'speaking':
        return t('callStateSpeaking');
      case 'muted':
        return t('callStateMuted');
      case 'pii_required':
        return t('callStatePiiRequired');
      case 'error':
        return voiceCall.error ?? '';
      default:
        return t('callEnded');
    }
  }, [voiceCall.state, voiceCall.error, t]);

  const handleManualStart = async () => {
    setAutoStartBlocked(false);
    await voiceCall.start();
  };

  const handleEndCall = async () => {
    voiceCall.end();
    if (sessionId) {
      try {
        await api.updateSession(sessionId, { status: 'completed' });
      } catch {
        // best-effort — do not block UI
      }
      setSessionId(null);
    }
    navigate('/');
  };

  const handleToggleMute = () => {
    void voiceCall.toggleMute();
  };

  // Keyboard shortcuts:
  //   M    - toggle mute (when the call is active)
  //   Esc  - end call
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || target?.isContentEditable) {
        return;
      }
      if (event.key === 'Escape') {
        event.preventDefault();
        void handleEndCall();
        return;
      }
      if (event.key === 'm' || event.key === 'M') {
        if (!callActive && !voiceCall.muted) return;
        event.preventDefault();
        handleToggleMute();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [callActive, voiceCall.muted]);

  if (!sessionId) {
    return null;
  }

  if (!voiceCall.supported) {
    return (
      <Layout language={language} onLanguageChange={setLanguage}>
        <section className="call-page-fallback">
          <h1>{t('callNotSupported')}</h1>
          <p className="muted">{t('callPermissionHelp')}</p>
          <button type="button" className="primary-btn" onClick={() => navigate('/chat')}>
            {t('modeChatTitle')}
          </button>
        </section>
      </Layout>
    );
  }

  return (
    <Layout
      language={language}
      onLanguageChange={setLanguage}
      navTitle={t('callPageTitle')}
      showAdminLink={false}
    >
      <section className="call-page">
        <div className="call-card">
          <div className="call-header">
            <span className="call-status-pill">{t('callPageSubtitle')}</span>
            <h1>{t('callPageTitle')}</h1>
          </div>

          <div className={`call-orb call-orb-${voiceCall.state}`}>
            <div className="call-orb-ring" aria-hidden="true" />
            <div className="call-orb-ring delay" aria-hidden="true" />
            <div className="call-orb-core" aria-hidden="true">
              <PhoneIcon />
            </div>
          </div>

          <div className="call-status-block">
            <span className={`call-status-text state-${voiceCall.state}`}>
              {statusLabel || t('callTapToStart')}
            </span>
            {voiceCall.muted && voiceCall.state !== 'muted' && (
              <span className="call-mute-badge" aria-live="polite">
                <span className="call-mute-badge-icon" aria-hidden="true">
                  <MicOffIcon />
                </span>
                {t('callMutedBadge')}
              </span>
            )}
          </div>

          {(voiceCall.lastTranscript || voiceCall.lastReply) && (
            <div className="call-captions">
              {voiceCall.lastTranscript && (
                <div className="caption caption-user">
                  <span className="caption-label">{t('lastYouSaid')}</span>
                  <p>"{voiceCall.lastTranscript}"</p>
                </div>
              )}
              {voiceCall.lastReply && (
                <div className="caption caption-assistant">
                  <span className="caption-label">{t('lastAssistantSaid')}</span>
                  <p>{voiceCall.lastReply}</p>
                </div>
              )}
            </div>
          )}

          {assessment?.emergency && !piiRequired && (
            <EmergencyBanner
              message={assessment.emergency.alertMessage}
              ctaLabel={t('callStaffNow')}
              onCtaClick={() => {
                window.alert(t('callStaffInstruction'));
              }}
            />
          )}

          {(piiRequired || piiReceipt) && (
            <SecurePiiForm
              onSubmit={submitEmergencyPii}
              triageLevel={assessment?.adk?.triageLevel}
              triageColor={assessment?.adk?.triageColor}
              receipt={piiReceipt}
            />
          )}

          {voiceCall.state === 'muted' && (
            <p className="call-mute-hint muted">{t('callMuteHint')}</p>
          )}

          <div className="call-actions">
            {!callActive && autoStartBlocked && (
              <button
                type="button"
                className="call-btn start"
                onClick={() => void handleManualStart()}
              >
                <span aria-hidden="true" className="call-btn-icon">{'\u260E'}</span>
                {t('callTapToStart')}
              </button>
            )}
            {callActive && !piiRequired && (
              <button
                type="button"
                className={`call-btn mute${voiceCall.muted ? ' muted' : ''}`}
                onClick={handleToggleMute}
                aria-pressed={voiceCall.muted}
                title={voiceCall.muted ? t('unmuteMic') : t('muteMic')}
              >
                <span aria-hidden="true" className="call-btn-icon">
                  {voiceCall.muted ? <MicOffIcon /> : <MicIcon />}
                </span>
                {voiceCall.muted ? t('unmuteMic') : t('muteMic')}
              </button>
            )}
            <button
              type="button"
              className="call-btn end call-btn-hangup"
              onClick={() => void handleEndCall()}
            >
              <span aria-hidden="true" className="call-btn-icon">
                <HangUpIcon />
              </span>
              {t('endCall')}
            </button>
          </div>

          {voiceCall.error && <p className="error-text call-error">{voiceCall.error}</p>}
        </div>
      </section>
    </Layout>
  );
}
