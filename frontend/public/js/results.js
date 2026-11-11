// Renders the results region: initial / loading / empty / error / results, with keyboard support.

import { clear, h } from './dom.js';
import {
  badgeFor,
  formatDate,
  meterColour,
  plural,
  relevanceWord,
  scorePercent,
} from './format.js';
import { hasFilters } from './state.js';

export const EXAMPLE_QUERIES = [
  'chest x-ray pneumonia',
  'high potassium kidney disease',
  'wrist fracture after a fall',
  'poorly controlled diabetes',
  'heart failure ejection fraction',
];

const CARD =
  'relative flex h-full flex-col rounded-xl border border-slate-200 bg-white p-4 shadow-sm ' +
  'transition-shadow hover:shadow-md focus-within:ring-2 focus-within:ring-indigo-700';
const PRIMARY_BUTTON =
  'inline-flex items-center rounded-lg bg-indigo-700 px-4 py-2 text-sm font-medium text-white ' +
  'hover:bg-indigo-800';
const CHIP =
  'rounded-full border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-800 ' +
  'hover:border-indigo-700 hover:bg-indigo-50';

export function badge({ label, classes }) {
  return h(
    'span',
    {
      class: `inline-flex shrink-0 items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${classes}`,
    },
    label,
  );
}

function relevanceMeter(score) {
  const percent = scorePercent(score);
  const fill = h('div', { class: `h-full rounded-full ${meterColour(percent)}` });
  fill.style.width = `${percent}%`; // CSSOM, not an inline style attribute (CSP-friendly)
  return h(
    'div',
    {
      role: 'meter',
      'aria-valuemin': 0,
      'aria-valuemax': 100,
      'aria-valuenow': percent,
      'aria-label': `Relevance: ${relevanceWord(percent)}, ${percent} percent`,
      class: 'flex items-center gap-2',
    },
    h('div', { class: 'h-2 w-24 overflow-hidden rounded-full bg-slate-200' }, fill),
    h(
      'span',
      { class: 'text-xs font-medium tabular-nums text-slate-700', 'aria-hidden': 'true' },
      `${relevanceWord(percent)} · ${percent}%`,
    ),
  );
}

function resultCard(result, index, onOpen) {
  const titleId = `result-${index}-title`;
  const isSummary = result.kind === 'summary';
  const button = h(
    'button',
    {
      type: 'button',
      class:
        `result-open mt-4 self-start ${PRIMARY_BUTTON} ` +
        'after:absolute after:inset-0 after:content-[""]', // stretched link: whole card is clickable
      // WCAG 2.5.3 "Label in Name": the accessible name must contain the visible text.
      'aria-label': `View patient ${result.patient_name}, MRN ${result.mrn}`,
      onclick: () => onOpen(result, button),
    },
    'View patient',
  );
  return h(
    'li',
    {},
    h(
      'article',
      { class: CARD, 'aria-labelledby': titleId },
      h(
        'header',
        { class: 'flex items-start justify-between gap-3' },
        h(
          'div',
          { class: 'min-w-0' },
          h('h3', { id: titleId, class: 'truncate text-base font-semibold text-slate-900' }, result.patient_name),
          h(
            'p',
            { class: 'text-sm text-slate-700' },
            'MRN ',
            h('span', { class: 'font-mono' }, result.mrn),
            ' · ',
            h('time', { datetime: result.record_date ?? undefined }, formatDate(result.record_date)),
          ),
        ),
        badge(badgeFor(result)),
      ),
      isSummary
        ? null
        : h(
            'p',
            { class: 'mt-3 line-clamp-2 text-sm text-slate-700' },
            h('span', { class: 'font-medium text-slate-900' }, 'Matched: '),
            result.matched_snippet,
          ),
      h(
        'div',
        { class: 'mt-3 rounded-lg bg-slate-50 p-3' },
        h('p', { class: 'text-xs font-semibold uppercase tracking-wide text-violet-900' }, 'AI summary'),
        h(
          'p',
          { class: 'mt-1 line-clamp-3 text-sm text-slate-800' },
          result.summary_snippet ?? result.matched_snippet,
        ),
      ),
      h('div', { class: 'mt-3' }, relevanceMeter(result.score)),
      button,
    ),
  );
}

function skeleton() {
  const bar = (w) => h('div', { class: `h-3 rounded bg-slate-200 ${w}` });
  const card = () =>
    h(
      'li',
      { 'aria-hidden': 'true' },
      h(
        'div',
        { class: 'space-y-3 rounded-xl border border-slate-200 bg-white p-4 motion-safe:animate-pulse' },
        bar('w-2/3'),
        bar('w-1/3'),
        bar('w-full'),
        bar('w-5/6'),
      ),
    );
  return [
    h('p', { class: 'sr-only' }, 'Searching…'),
    h('ul', { role: 'list', class: 'grid gap-4 lg:grid-cols-2' }, [1, 2, 3, 4].map(card)),
  ];
}

function panel(...children) {
  return h('div', { class: 'rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center' }, ...children);
}

export function renderResults(container, state, handlers) {
  container.setAttribute('aria-busy', String(state.status === 'loading'));
  clear(container);

  if (state.status === 'loading') {
    container.append(...skeleton());
    return;
  }

  if (state.status === 'error') {
    container.append(
      h(
        'div',
        { role: 'alert', class: 'rounded-xl border border-red-300 bg-red-50 p-6 text-red-900' },
        h('h2', { class: 'text-base font-semibold' }, 'The search could not be completed'),
        h('p', { class: 'mt-1 text-sm' }, state.error),
        h(
          'button',
          { type: 'button', class: `mt-4 ${PRIMARY_BUTTON}`, onclick: handlers.onRetry },
          'Try again',
        ),
      ),
    );
    return;
  }

  if (state.status === 'success' && state.results.length === 0) {
    container.append(
      panel(
        h('h2', { class: 'text-lg font-semibold text-slate-900' }, 'No matching records'),
        h(
          'p',
          { class: 'mx-auto mt-2 max-w-prose text-sm text-slate-700' },
          'Nothing matched that description. Try different words or a broader description',
          hasFilters(state) ? ', or remove the filters.' : '.',
        ),
        hasFilters(state)
          ? h(
              'button',
              { type: 'button', class: `mt-4 ${PRIMARY_BUTTON}`, onclick: handlers.onClearFilters },
              'Clear filters',
            )
          : null,
      ),
    );
    return;
  }

  if (state.status === 'success') {
    container.append(
      // Keeps the heading outline h1 > h2 > h3 for screen-reader users who navigate by heading.
      h('h2', { class: 'sr-only' }, 'Search results'),
      h(
        'ul',
        { role: 'list', class: 'result-list grid gap-4 lg:grid-cols-2' },
        state.results.map((r, i) => resultCard(r, i, handlers.onOpen)),
      ),
    );
    initRovingFocus(container.querySelector('.result-list'), handlers.onLeaveTop);
    return;
  }

  // idle
  container.append(
    panel(
      h('h2', { class: 'text-lg font-semibold text-slate-900' }, 'Search patient records'),
      h(
        'p',
        { class: 'mx-auto mt-2 max-w-prose text-sm text-slate-700' },
        'Describe what you are looking for in plain language. Results combine source documents, ' +
          'lab and imaging reports, and AI-generated patient summaries.',
      ),
      h('p', { class: 'mt-5 text-xs font-medium uppercase tracking-wide text-slate-600' }, 'Try one of these'),
      h(
        'ul',
        { role: 'list', class: 'mt-2 flex flex-wrap justify-center gap-2' },
        EXAMPLE_QUERIES.map((q) =>
          h('li', {}, h('button', { type: 'button', class: CHIP, onclick: () => handlers.onExample(q) }, q)),
        ),
      ),
    ),
  );
}

// --- keyboard: arrow keys move between result buttons; only one is in the tab order ----------

function initRovingFocus(list, onLeaveTop) {
  if (!list) return;
  const buttons = () => [...list.querySelectorAll('.result-open')];
  const setActive = (active) => buttons().forEach((b) => (b.tabIndex = b === active ? 0 : -1));
  setActive(buttons()[0]);

  list.addEventListener('focusin', (event) => {
    if (event.target.classList?.contains('result-open')) setActive(event.target);
  });
  list.addEventListener('keydown', (event) => {
    const items = buttons();
    const index = items.indexOf(document.activeElement);
    if (index === -1) return;
    let target = null;
    if (event.key === 'ArrowDown' || event.key === 'ArrowRight') target = items[Math.min(index + 1, items.length - 1)];
    else if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') {
      if (index === 0) {
        event.preventDefault();
        onLeaveTop();
        return;
      }
      target = items[index - 1];
    } else if (event.key === 'Home') target = items[0];
    else if (event.key === 'End') target = items[items.length - 1];
    if (target) {
      event.preventDefault();
      target.focus();
    }
  });
}

export function focusFirstResult(container) {
  const first = container.querySelector('.result-open');
  first?.focus();
  return Boolean(first);
}

export function resultsSummary(state) {
  if (state.status !== 'success') return '';
  const took = state.tookMs == null ? '' : ` in ${Math.round(state.tookMs)} ms`;
  return `${plural(state.results.length, 'result')} for “${state.query.trim()}”${took}`;
}
