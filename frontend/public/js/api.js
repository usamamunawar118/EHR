// Thin, testable client for the backend. `fetch` is injected so tests need no network.

export class ApiError extends Error {
  constructor(message, { status = 0, cause } = {}) {
    super(message, { cause });
    this.name = 'ApiError';
    this.status = status;
  }
}

/** Build the POST /search request. Filters are query parameters; the query is the JSON body. */
export function buildSearchRequest({ query, resourceType, dateFrom, dateTo }) {
  const params = new URLSearchParams();
  if (resourceType) params.set('resource_type', resourceType);
  if (dateFrom) params.set('date_from', dateFrom);
  if (dateTo) params.set('date_to', dateTo);
  const qs = params.toString();
  return {
    url: `/search${qs ? `?${qs}` : ''}`,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ query: query.trim() }),
    },
  };
}

/** FastAPI returns `detail` as a string or as a list of {loc, msg}. Make it readable. */
export function describeError(status, body) {
  const detail = body?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((d) => {
        const where = Array.isArray(d.loc) ? d.loc.filter((p) => p !== 'body' && p !== 'query').join('.') : '';
        return where ? `${where}: ${d.msg}` : String(d.msg);
      })
      .join('; ');
  }
  if (status >= 500) return 'The server had a problem handling this request. Please try again.';
  return `Request failed (HTTP ${status}).`;
}

export function createApi({ fetchImpl = globalThis.fetch?.bind(globalThis) } = {}) {
  async function request(url, init = {}, signal) {
    let response;
    try {
      response = await fetchImpl(url, { ...init, signal });
    } catch (error) {
      if (error?.name === 'AbortError') throw error; // a superseded request is not an error
      throw new ApiError('Cannot reach the server. Check your connection and try again.', {
        cause: error,
      });
    }
    if (!response.ok) {
      let body = null;
      try {
        body = await response.json();
      } catch {
        /* non-JSON error body */
      }
      throw new ApiError(describeError(response.status, body), { status: response.status });
    }
    return response.json();
  }

  return {
    search: (criteria, signal) => {
      const { url, init } = buildSearchRequest(criteria);
      return request(url, init, signal);
    },
    getPatient: (patientId, signal) =>
      request(`/patients/${encodeURIComponent(patientId)}`, {}, signal),
    getResource: (patientId, resourceId, signal) =>
      request(
        `/patients/${encodeURIComponent(patientId)}/resources/${encodeURIComponent(resourceId)}`,
        {},
        signal,
      ),
    getResourceTypes: (signal) => request('/meta/resource-types', {}, signal),
  };
}

export const bundleUrl = (patientId) =>
  `/patients/${encodeURIComponent(patientId)}/bundle?download=true`;
