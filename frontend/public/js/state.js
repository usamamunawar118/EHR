// Minimal observable store: enough for one screen, no framework.

export function createStore(initial) {
  let state = initial;
  const subscribers = new Set();
  return {
    get: () => state,
    set(patch) {
      state = { ...state, ...patch };
      for (const fn of subscribers) fn(state);
    },
    subscribe(fn) {
      subscribers.add(fn);
      return () => subscribers.delete(fn);
    },
  };
}

export const MIN_QUERY_CHARS = 2;

export const initialState = Object.freeze({
  query: '',
  resourceType: '',
  dateFrom: '',
  dateTo: '',
  status: 'idle', // idle | loading | success | error
  results: [],
  total: 0,
  tookMs: null,
  error: null,
});

/** Validate user input before spending a request. Returns an error message or null. */
export function validateCriteria({ query, dateFrom, dateTo }) {
  if (dateFrom && dateTo && dateFrom > dateTo) return 'The start date must be on or before the end date.';
  if (query.trim().length > 0 && query.trim().length < MIN_QUERY_CHARS) {
    return `Type at least ${MIN_QUERY_CHARS} characters to search.`;
  }
  return null;
}

export const hasFilters = ({ resourceType, dateFrom, dateTo }) =>
  Boolean(resourceType || dateFrom || dateTo);
