import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { AppLanguage } from '../i18n/resources';

const voiceFeatureEnabled = import.meta.env.VITE_ENABLE_VOICE === 'true';

function pickMediaRecorderMimeType(): string | undefined {
  if (typeof MediaRecorder === 'undefined') return undefined;
  const preferences = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/ogg',
    'audio/mp4',
    'audio/mpeg',
  ];
  for (const candidate of preferences) {
    if (MediaRecorder.isTypeSupported(candidate)) {
      return candidate;
    }
  }
  return undefined;
}

export function useSpeechRecognition(language: AppLanguage) {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [confidence, setConfidence] = useState<number | null>(null);
  const [supported, setSupported] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    const hasMedia =
      typeof navigator !== 'undefined' &&
      typeof navigator.mediaDevices?.getUserMedia === 'function' &&
      typeof MediaRecorder !== 'undefined';
    setSupported(voiceFeatureEnabled && hasMedia);
  }, []);

  const cleanupStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }, []);

  const startListening = useCallback(async () => {
    if (!voiceFeatureEnabled || !supported) return;
    if (recorderRef.current) return;

    setError(null);
    setTranscript('');
    setConfidence(null);

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Microphone access denied');
      return;
    }

    streamRef.current = stream;
    const mimeType = pickMediaRecorderMimeType();
    const recorder = mimeType
      ? new MediaRecorder(stream, { mimeType })
      : new MediaRecorder(stream);
    recorderRef.current = recorder;
    chunksRef.current = [];

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    recorder.onstop = async () => {
      const blobType = recorder.mimeType || mimeType || 'audio/webm';
      const audioBlob = new Blob(chunksRef.current, { type: blobType });
      chunksRef.current = [];
      cleanupStream();
      recorderRef.current = null;
      setIsListening(false);

      if (audioBlob.size === 0) {
        setError('No audio captured');
        return;
      }

      try {
        const result = await api.stt(audioBlob, language, fileNameForMime(blobType));
        setTranscript(result.transcript);
        setConfidence(result.confidence);
        if (!result.transcript) {
          setError('Could not understand the audio. Please try again.');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Transcription failed');
      }
    };

    recorder.start();
    setIsListening(true);
  }, [supported, language, cleanupStream]);

  const stopListening = useCallback(() => {
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop();
    } else {
      cleanupStream();
      setIsListening(false);
    }
  }, [cleanupStream]);

  const clearTranscript = useCallback(() => {
    setTranscript('');
    setConfidence(null);
    setError(null);
  }, []);

  useEffect(() => {
    return () => {
      if (recorderRef.current && recorderRef.current.state !== 'inactive') {
        recorderRef.current.stop();
      }
      cleanupStream();
    };
  }, [cleanupStream]);

  return {
    isListening,
    transcript,
    confidence,
    supported,
    enabled: voiceFeatureEnabled,
    error,
    startListening,
    stopListening,
    clearTranscript,
  };
}

function fileNameForMime(mime: string): string {
  if (mime.includes('webm')) return 'speech.webm';
  if (mime.includes('ogg')) return 'speech.ogg';
  if (mime.includes('mp4')) return 'speech.mp4';
  if (mime.includes('mpeg')) return 'speech.mp3';
  if (mime.includes('wav')) return 'speech.wav';
  return 'speech.bin';
}

/**
 * Sentence-boundary regex used to chunk streaming text for TTS. Matches
 * "..text.", "..text!", "..text?" and the Thai sentence terminator
 * "ๆ" / "ฯ", with an optional trailing whitespace run. Greedy across
 * a single segment so we send full clauses rather than one token at
 * a time, which keeps TTS audio natural-sounding.
 */
const SENTENCE_BOUNDARY_RE = /([^.!?…\n]+[.!?…\n]+)/g;

interface QueuedAudio {
  url: string;
  audio: HTMLAudioElement;
}

export function useSpeechSynthesis(language: AppLanguage) {
  const [enabled, setEnabled] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Sequential playback queue. Each entry is an MP3 blob already
  // fetched from /tts; ``isSpeaking`` flips true while any entry is
  // playing. The first entry's element is the currently-playing one
  // (so we can stop / cancel it cheaply). Subsequent entries wait.
  const queueRef = useRef<QueuedAudio[]>([]);
  const playingRef = useRef<QueuedAudio | null>(null);
  // Tracks the unflushed tail of streaming text — anything before the
  // last sentence terminator has already been queued for TTS.
  const pendingTextRef = useRef('');
  // Incrementing token so an old in-flight ``api.tts`` fetch can
  // detect that ``stop()`` was called and silently discard its blob
  // rather than enqueuing audio from a cancelled turn.
  const streamTokenRef = useRef(0);
  // Mirror of ``enabled`` state into a ref so callers that captured
  // ``speakStreamChunk`` / ``flushStream`` in a stale closure (e.g.
  // an in-flight ``useChat.sendMessageStream`` invocation that
  // started before the user toggled the speaker off) still respect
  // the latest toggle state. Without this the React-state guard
  // ``if (!enabled) return;`` reads the stale ``true`` value and
  // keeps fetching TTS blobs for several seconds after the user
  // clicked the speaker icon.
  const enabledRef = useRef(enabled);
  useEffect(() => {
    enabledRef.current = enabled;
  }, [enabled]);

  const supported = typeof window !== 'undefined' && typeof Audio !== 'undefined';

  const playNext = useCallback(() => {
    const next = queueRef.current.shift();
    if (!next) {
      playingRef.current = null;
      setIsSpeaking(false);
      return;
    }
    playingRef.current = next;
    setIsSpeaking(true);
    next.audio.onended = () => {
      URL.revokeObjectURL(next.url);
      if (playingRef.current === next) {
        playingRef.current = null;
      }
      playNext();
    };
    next.audio.onerror = () => {
      URL.revokeObjectURL(next.url);
      if (playingRef.current === next) {
        playingRef.current = null;
      }
      setError('Audio playback failed');
      playNext();
    };
    void next.audio.play().catch(() => {
      // Autoplay rejection — surface as a soft error and continue
      // with the queue. The user can re-enable by clicking the
      // speaker toggle (which re-anchors a gesture).
      setError('Audio playback blocked by browser');
      playNext();
    });
  }, []);

  const stop = useCallback(() => {
    streamTokenRef.current += 1;
    pendingTextRef.current = '';
    const playing = playingRef.current;
    if (playing) {
      playing.audio.pause();
      playing.audio.src = '';
      URL.revokeObjectURL(playing.url);
      playingRef.current = null;
    }
    for (const queued of queueRef.current) {
      try {
        queued.audio.pause();
      } catch {
        // ignore
      }
      URL.revokeObjectURL(queued.url);
    }
    queueRef.current = [];
    setIsSpeaking(false);
  }, []);

  const enqueue = useCallback(
    async (text: string, token: number) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      try {
        const blob = await api.tts(trimmed, language);
        // ``stop()`` was called while we were fetching — abandon the
        // blob silently. Without this check a stopped-then-restarted
        // turn would interleave old audio into the new turn.
        if (token !== streamTokenRef.current) return;
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        queueRef.current.push({ url, audio });
        if (!playingRef.current) {
          playNext();
        }
      } catch (err) {
        if (token === streamTokenRef.current) {
          setError(err instanceof Error ? err.message : 'Text-to-speech failed');
        }
      }
    },
    [language, playNext],
  );

  /**
   * Speak a complete text in one shot. Used by the non-streaming
   * fallback path. Cancels any in-flight queue first so the new text
   * doesn't queue behind stale audio.
   *
   * Reads ``enabledRef.current`` (not ``enabled``) so a stale
   * closure can never start a new utterance after the user toggled
   * the speaker off.
   */
  const speak = useCallback(
    async (text: string) => {
      if (!enabledRef.current || !supported || !text.trim()) return;
      stop();
      setError(null);
      const token = streamTokenRef.current;
      await enqueue(text, token);
    },
    [supported, enqueue, stop],
  );

  /**
   * Feed a chunk of streaming text. We accumulate into
   * ``pendingTextRef`` and emit a TTS request for each complete
   * sentence as boundaries (``.``, ``!``, ``?``, ``…``, newline) come
   * in. The leftover tail (an in-progress sentence) waits for the
   * next chunk — or for :meth:`flushStream` to finalise it.
   *
   * Gated on ``enabledRef.current`` rather than the ``enabled``
   * closure value so that an in-flight ``sendMessageStream`` whose
   * onDelta captured the old function reference still drops chunks
   * the moment the user toggles the speaker off mid-utterance.
   */
  const speakStreamChunk = useCallback(
    (chunk: string) => {
      if (!enabledRef.current || !supported || !chunk) return;
      pendingTextRef.current += chunk;
      const token = streamTokenRef.current;

      const matches = pendingTextRef.current.match(SENTENCE_BOUNDARY_RE);
      if (!matches || matches.length === 0) return;
      const joined = matches.join('');
      pendingTextRef.current = pendingTextRef.current.slice(joined.length);

      for (const sentence of matches) {
        void enqueue(sentence, token);
      }
    },
    [supported, enqueue],
  );

  /**
   * Flush any unfinalised tail at end-of-stream. Call this from the
   * ``complete`` handler so the last (possibly punctuation-less)
   * fragment still gets spoken. Same enabledRef gating as
   * :func:`speakStreamChunk` for the same stale-closure reason.
   */
  const flushStream = useCallback(() => {
    if (!enabledRef.current || !supported) return;
    const tail = pendingTextRef.current.trim();
    pendingTextRef.current = '';
    if (tail) {
      void enqueue(tail, streamTokenRef.current);
    }
  }, [supported, enqueue]);

  const toggle = useCallback(() => {
    setEnabled((prev) => {
      const next = !prev;
      // Mirror into the ref synchronously so stale-closure callers
      // (e.g. an in-flight stream's onDelta) see the new value on
      // the very next chunk — not on the next render after the
      // useEffect mirror runs.
      enabledRef.current = next;
      if (!next) {
        // Speaker turned off mid-utterance — cut everything immediately
        // so the caller stops hearing audio the moment they click.
        stop();
      }
      return next;
    });
  }, [stop]);

  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  return {
    enabled,
    supported,
    isSpeaking,
    error,
    speak,
    speakStreamChunk,
    flushStream,
    stop,
    toggle,
    setEnabled,
  };
}
