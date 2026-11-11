// Entry point: wires the search form, filters, results and drawer together.

import { createApi } from './api.js';
import { announce } from './dom.js';
import { createDrawer } from './drawer.js';
import { focusFirstResult, renderResults, resultsSummary } from './results.js';
import { createStore, hasFilters, initialState, MIN_QUERY_CHARS, validateCriteria } from './state.js';
import { fromQueryString, toQueryString } from './urlstate.js';

const DEBOUNCE_MS = 300;

const $ = (id) => document.getElementById(id);
const els = {
  form: $('search-form'),
  query: $('q'),
  type: $('resource-type'),
  from: $('date-from'),
  to: $('date-to'),
  clear: $('clear-filters'),
  filterError: $('filter-error'),
  results: $('results'),
  dialog: $('patient-dialog'),
};

const api = createApi();
const store = createStore({ ...initialState });
const drawer = createDrawer({
  dialog: els.dialog,
  body: $('drawer-body'),
  titleEl: $('drawer-title'),
  subtitleEl: $('drawer-subtitle'),
  closeButton: $('drawer-close'),
  api,
});

let inflight = null; // AbortController of the newest request
let debounceTimer = null;

// --- searching ---------------------------------------------------------------------------

async function runSearch() {
  clearTimeout(debounceTimer);
  const criteria = store.get();
  const problem = validateCriteria(criteria);
  els.filterError.textContent = problem ?? '';
  els.filterError.hidden = !problem;
  history.replaceState(null, '', `${location.pathname}${toQueryString(criteria)}`);

  if (problem || criteria.query.trim().length < MIN_QUERY_CHARS) {
    inflight?.abort();
    store.set({ status: 'idle', results: [], total: 0, tookMs: null, error: null });
    announce('');
    return;
  }

  inflight?.abort(); // a newer request supersedes the old one: no stale results can render
  const mine = (inflight = new AbortController());
  store.set({ status: 'loading', error: null });
  announce('Searching…');
  try {
    const response = await api.search(criteria, mine.signal);
    if (inflight !== mine) return;
    store.set({
      status: 'success',
      results: response.results,
      total: response.total,
      tookMs: response.took_ms,
    });
    announce(resultsSummary(store.get()) || 'No results');
  } catch (error) {
    if (error?.name === 'AbortError' || inflight !== mine) return;
    store.set({ status: 'error', error: error.message, results: [] });
    announce(`Search failed: ${error.message}`);
  }
}

function scheduleSearch() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(runSearch, DEBOUNCE_MS);
}

// --- rendering ---------------------------------------------------------------------------

const handlers = {
  onOpen: (result, button) => drawer.open(result.patient_id, button, result.patient_name),
  onExample: (query) => {
    els.query.value = query;
    store.set({ query });
    runSearch();
    els.query.focus();
  },
  onClearFilters: () => clearFilters(true),
  onRetry: runSearch,
  onLeaveTop: () => els.query.focus(),
};

// Re-render the results only when something they display changed. Typing in the search box
// updates `query` on every keystroke; redrawing then would flicker and reset keyboard focus.
let lastRendered = null;
store.subscribe((state) => {
  els.clear.hidden = !hasFilters(state);
  const view = {
    status: state.status,
    results: state.results,
    error: state.error,
    filtered: hasFilters(state),
  };
  if (lastRendered && Object.keys(view).every((k) => view[k] === lastRendered[k])) return;
  lastRendered = view;
  renderResults(els.results, state, handlers);
});

function clearFilters(search) {
  els.type.value = els.from.value = els.to.value = '';
  store.set({ resourceType: '', dateFrom: '', dateTo: '' });
  if (search) runSearch();
}

// --- events ------------------------------------------------------------------------------

els.query.addEventListener('input', () => {
  store.set({ query: els.query.value });
  scheduleSearch();
});
els.form.addEventListener('submit', (event) => {
  event.preventDefault();
  store.set({ query: els.query.value });
  runSearch();
});
for (const [el, key] of [
  [els.type, 'resourceType'],
  [els.from, 'dateFrom'],
  [els.to, 'dateTo'],
]) {
  el.addEventListener('change', () => {
    store.set({ [key]: el.value });
    runSearch();
  });
}
els.clear.addEventListener('click', () => {
  clearFilters(true);
  els.query.focus();
});

// "/" jumps to the search box (like most dashboards); ArrowDown moves into the results.
document.addEventListener('keydown', (event) => {
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName) || event.target.isContentEditable;
  if (event.key === '/' && !typing && !els.dialog.open && !event.ctrlKey && !event.metaKey) {
    event.preventDefault();
    els.query.focus();
    els.query.select();
  }
});
els.query.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowDown' && focusFirstResult(els.results)) event.preventDefault();
});

// --- start-up ----------------------------------------------------------------------------

async function loadResourceTypes() {
  try {
    const options = await api.getResourceTypes();
    for (const { value, label } of options) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      els.type.append(option);
    }
    return options.map((o) => o.value);
  } catch {
    return null; // the filter simply stays at "All record types"
  }
}

async function start() {
  renderResults(els.results, store.get(), handlers); // never show a blank page while we fetch
  const allowedTypes = await loadResourceTypes();
  const initial = fromQueryString(location.search, allowedTypes);
  els.query.value = initial.query;
  els.type.value = initial.resourceType;
  els.from.value = initial.dateFrom;
  els.to.value = initial.dateTo;
  store.set(initial);
  if (initial.query) runSearch();
}

start();
