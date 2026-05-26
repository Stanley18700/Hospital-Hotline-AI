import type { ApiError, PiiCollectionGateDetail } from './types';

const baseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

/**
 * Error thrown for non-2xx HTTP responses. Preserves the original
 * status code and the raw FastAPI ``detail`` payload (string OR
 * structured object) so callers can branch on it without re-parsing.
 *
 * The most important structured case we care about is the /stt 409
 * gate fired when the session is in PII_COLLECT phase — callers
 * can use :func:`isPiiCollectionGate` to detect it.
 */
export class ApiClientError extends Error {
  readonly status: number;
  readonly detail: string | Record<string, unknown>;

  constructor(status: number, detail: string | Record<string, unknown>) {
    super(typeof detail === 'string' ? detail : (detail.message as string) ?? `HTTP ${status}`);
    this.name = 'ApiClientError';
    this.status = status;
    this.detail = detail;
  }
}

export function isPiiCollectionGate(
  err: unknown,
): err is ApiClientError & { detail: PiiCollectionGateDetail } {
  if (!(err instanceof ApiClientError)) return false;
  if (err.status !== 409) return false;
  const detail = err.detail;
  if (typeof detail !== 'object' || detail === null) return false;
  return (detail as { code?: string }).code === 'pii_collection_active';
}

async function parseErrorBody(response: Response): Promise<string | Record<string, unknown>> {
  try {
    const body = (await response.json()) as ApiError;
    if (body.detail !== undefined && body.detail !== null) return body.detail;
  } catch {
    // body wasn't JSON — fall through
  }
  return response.statusText || `HTTP ${response.status}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  });

  if (!response.ok) {
    throw new ApiClientError(response.status, await parseErrorBody(response));
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

export { baseUrl, parseErrorBody, request };
