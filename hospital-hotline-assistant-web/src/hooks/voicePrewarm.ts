/**
 * Voice call prewarm utilities.
 *
 * The first time a user makes a voice call, the bulk of the perceived
 * latency comes from things that have nothing to do with the agent
 * itself:
 *   - Browser microphone permission prompt (~200ms-2s, sometimes longer
 *     if the user is slow to click "Allow").
 *   - First-ever `AudioContext` construction (~50-100ms of audio thread
 *     setup, plus the autoplay-policy "user gesture" anchoring).
 *   - AudioWorklet module compile and load (~50-100ms).
 *
 * Once those are done they stay done for the rest of the tab's lifetime:
 *   - The browser caches the mic permission decision (so subsequent
 *     getUserMedia calls resolve instantly without prompting).
 *   - We keep the AudioContext alive in a module-scoped ref so the call
 *     hook can reuse it instead of building a fresh one.
 *
 * This module lets pages that *know* a voice call is imminent (most
 * notably the landing page's "Call" tile) fire all that work in parallel
 * with their own async work (session creation, navigation), so by the
 * time `useVoiceCall.start()` runs, every gate is already open.
 *
 * The functions are intentionally idempotent and best-effort: failures
 * never throw to the caller, they just leave the resource unprepared
 * and the regular `start()` flow will request it on demand.
 */

const OUTPUT_SAMPLE_RATE = 24000;

let warmedMicPermissionPromise: Promise<void> | null = null;
let warmedPlaybackContext: AudioContext | null = null;

interface GlobalWithVendorAudioContext {
  AudioContext?: typeof AudioContext;
  webkitAudioContext?: typeof AudioContext;
}

function resolveAudioContextCtor(): typeof AudioContext | null {
  if (typeof window === 'undefined') return null;
  const g = window as unknown as GlobalWithVendorAudioContext;
  return g.AudioContext ?? g.webkitAudioContext ?? null;
}

/**
 * Trigger the microphone permission prompt and immediately release the
 * MediaStream. The browser remembers the permission decision (per origin
 * + per permissions-policy lifetime) so the real ``getUserMedia`` call
 * inside ``useVoiceCall.start()`` resolves without re-prompting.
 *
 * Returns a memoised promise so concurrent callers share the same
 * single prompt. Safe to invoke repeatedly.
 */
export function prewarmMicrophonePermission(): Promise<void> {
  if (warmedMicPermissionPromise) return warmedMicPermissionPromise;

  if (
    typeof navigator === 'undefined' ||
    typeof navigator.mediaDevices?.getUserMedia !== 'function'
  ) {
    return Promise.resolve();
  }

  warmedMicPermissionPromise = (async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });
      stream.getTracks().forEach((track) => track.stop());
    } catch {
      // User denied or browser failed to expose mic. Reset the cache so
      // a later attempt (e.g. after the user updates site permissions)
      // re-tries the prompt rather than serving the rejection forever.
      warmedMicPermissionPromise = null;
    }
  })();

  return warmedMicPermissionPromise;
}

/**
 * Build (or reuse) a 24 kHz playback AudioContext. Browsers require the
 * AudioContext to be created inside a user-gesture callback to bypass
 * autoplay restrictions; calling this from a click handler on the
 * landing page anchors the gesture, so the same context can later be
 * resumed on the call page without needing another click.
 *
 * Returns ``null`` if the platform has no AudioContext (e.g. very old
 * Safari, SSR). Callers should fall back to lazy creation in that case.
 */
export function prewarmPlaybackContext(): AudioContext | null {
  if (warmedPlaybackContext && warmedPlaybackContext.state !== 'closed') {
    return warmedPlaybackContext;
  }

  const Ctor = resolveAudioContextCtor();
  if (!Ctor) return null;

  try {
    warmedPlaybackContext = new Ctor({ sampleRate: OUTPUT_SAMPLE_RATE });
  } catch {
    warmedPlaybackContext = null;
    return null;
  }

  return warmedPlaybackContext;
}

/**
 * Hand the prewarmed playback context off to the call hook. The hook
 * takes ownership: once consumed, the module no longer references it,
 * so a future ``prewarmPlaybackContext()`` will build a fresh one.
 *
 * This handoff pattern (rather than letting the hook read directly from
 * the module-scoped ref) avoids accidental double-ownership where both
 * the hook and the prewarm cache try to close the same context.
 */
export function takePrewarmedPlaybackContext(): AudioContext | null {
  const ctx = warmedPlaybackContext;
  warmedPlaybackContext = null;
  if (!ctx || ctx.state === 'closed') return null;
  return ctx;
}

/**
 * Fire-and-forget convenience used by pages that know a call is about
 * to start (e.g. landing page Call tile). Runs both prewarm steps in
 * parallel; resolves once both complete (or fail). Never throws.
 */
export async function prewarmVoiceCall(): Promise<void> {
  prewarmPlaybackContext();
  await prewarmMicrophonePermission();
}
