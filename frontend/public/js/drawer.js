// Patient detail drawer: a native modal <dialog> (focus trap, Esc, inert background and
// aria-modal come from the platform rather than hand-rolled code).

import { bundleUrl } from './api.js';
import { clear, h } from './dom.js';
import {
  badgeFor,
  confidenceClasses,
  describeGenerator,
  formatDate,
  groupResources,
  plural,
  redactAttachments,
} from './format.js';
import { badge } from './results.js';

const SECTION_TITLE = 'text-sm font-semibold uppercase tracking-wide text-slate-700';

function section(id, title, ...children) {
  return h(
    'section',
    { 'aria-labelledby': id, class: 'space-y-2' },
    h('h3', { id, class: SECTION_TITLE }, title),
    ...children,
  );
}

function bulletList(items, itemClass = 'text-sm text-slate-900') {
  return h(
    'ul',
    { role: 'list', class: 'list-disc space-y-1 pl-5' },
    items.map((text) => h('li', { class: itemClass }, text)),
  );
}

function notDocumented() {
  return h('p', { class: 'text-sm italic text-slate-600' }, 'None documented');
}

function summaryView(summary) {
  const media = summary.recent_media.length
    ? h(
        'ul',
        { role: 'list', class: 'space-y-2' },
        summary.recent_media.map((m) =>
          h(
            'li',
            { class: 'rounded-lg border border-slate-200 p-3 text-sm' },
            h(
              'div',
              { class: 'flex flex-wrap items-center gap-2' },
              badge(
                m.type === 'imaging'
                  ? { label: 'Imaging', classes: 'bg-sky-100 text-sky-900 ring-sky-300' }
                  : { label: 'Lab', classes: 'bg-emerald-100 text-emerald-900 ring-emerald-300' },
              ),
              h('time', { datetime: m.date, class: 'font-medium text-slate-900' }, formatDate(m.date)),
            ),
            h('p', { class: 'mt-1 text-slate-800' }, m.finding),
          ),
        ),
      )
    : notDocumented();

  const anomalies = summary.flagged_anomalies.length
    ? h(
        'ul',
        { role: 'list', class: 'space-y-1' },
        summary.flagged_anomalies.map((a) =>
          h(
            'li',
            { class: 'flex gap-2 rounded-lg border border-amber-300 bg-amber-50 p-2 text-sm text-amber-950' },
            h('span', { 'aria-hidden': 'true' }, '⚠'),
            h('span', {}, a),
          ),
        ),
      )
    : notDocumented();

  const ungrounded = summary.guardrails?.ungrounded_values ?? [];
  return [
    h(
      'div',
      { role: 'note', class: 'rounded-lg border border-violet-300 bg-violet-50 p-3 text-sm text-violet-950' },
      h('p', { class: 'font-semibold' }, 'AI-generated — not a clinical decision'),
      h('p', { class: 'mt-1' }, summary.disclaimer),
      h(
        'p',
        { class: 'mt-2 flex flex-wrap items-center gap-2 text-xs' },
        h('span', {}, 'Confidence:'),
        badge({
          label: summary.confidence.charAt(0).toUpperCase() + summary.confidence.slice(1),
          classes: confidenceClasses(summary.confidence),
        }),
        h('span', {}, `· ${plural(summary.word_count, 'word')} · ${describeGenerator(summary.generated_by)}`),
      ),
      ungrounded.length
        ? h(
            'p',
            { class: 'mt-2 text-xs font-medium text-red-900' },
            `Check before relying on this: ${ungrounded.join(', ')} could not be traced to the source records.`,
          )
        : null,
    ),
    section('sum-concern', 'Chief concern', h('p', { class: 'text-sm text-slate-900' }, summary.chief_concern)),
    section(
      'sum-dx',
      'Key diagnoses',
      summary.key_diagnoses.length ? bulletList(summary.key_diagnoses) : notDocumented(),
    ),
    section('sum-media', 'Recent media records', media),
    section('sum-flags', 'Flagged anomalies', anomalies),
  ];
}

function resourceItem(api, patientId, resource, signalFor) {
  let loaded = false;
  const pre = h(
    'pre',
    {
      tabindex: '0',
      'aria-label': `FHIR JSON for ${resource.display}`,
      class: 'max-h-64 overflow-auto rounded bg-slate-900 p-3 text-xs leading-relaxed text-slate-100',
    },
    'Loading…',
  );
  const details = h(
    'details',
    { class: 'rounded-lg border border-slate-200 bg-white' },
    h(
      'summary',
      { class: 'flex cursor-pointer flex-wrap items-center gap-x-3 gap-y-1 p-3 text-sm' },
      badge(badgeFor(resource)),
      h('span', { class: 'font-medium text-slate-900' }, resource.display),
      h('time', { class: 'text-slate-700', datetime: resource.effective_date ?? undefined }, resource.effective_date ? formatDate(resource.effective_date) : ''),
    ),
    h('div', { class: 'border-t border-slate-200 p-3' }, h('p', { class: 'mb-2 font-mono text-xs text-slate-600' }, resource.resource_id), pre),
  );
  details.addEventListener('toggle', async () => {
    if (!details.open || loaded) return;
    try {
      const json = await api.getResource(patientId, resource.resource_id, signalFor());
      pre.textContent = JSON.stringify(redactAttachments(json), null, 2);
      loaded = true;
    } catch (error) {
      if (error?.name !== 'AbortError') pre.textContent = `Could not load this resource: ${error.message}`;
    }
  });
  return h('li', {}, details);
}

function resourcesView(api, detail, signalFor) {
  const groups = groupResources(detail.resources);
  return section(
    'linked-resources',
    `Linked FHIR resources (${detail.resources.length})`,
    h(
      'div',
      { class: 'space-y-4' },
      groups.map(([type, items]) =>
        h(
          'div',
          {},
          h('h4', { class: 'mb-2 text-sm font-medium text-slate-900' }, `${type} · ${items.length}`),
          h(
            'ul',
            { role: 'list', class: 'space-y-2' },
            items.map((r) => resourceItem(api, detail.patient.patient_id, r, signalFor)),
          ),
        ),
      ),
    ),
    h(
      'p',
      { class: 'pt-2' },
      h(
        'a',
        {
          href: bundleUrl(detail.patient.patient_id),
          download: '',
          class: 'text-sm font-medium text-indigo-700 underline underline-offset-2 hover:text-indigo-900',
        },
        'Download the full FHIR R4 Bundle (JSON)',
      ),
    ),
  );
}

export function createDrawer({ dialog, body, titleEl, subtitleEl, closeButton, api }) {
  let controller = null;
  let opener = null;
  const signalFor = () => controller?.signal;

  closeButton.addEventListener('click', () => dialog.close());
  // A click whose target is the <dialog> itself landed on the backdrop.
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) dialog.close();
  });
  dialog.addEventListener('close', () => {
    controller?.abort();
    body.setAttribute('aria-busy', 'false');
    opener?.focus(); // return focus to the control that opened the drawer
  });

  function showLoading() {
    body.setAttribute('aria-busy', 'true');
    clear(body).append(
      h('p', { class: 'sr-only' }, 'Loading patient details…'),
      h(
        'div',
        { 'aria-hidden': 'true', class: 'space-y-3 motion-safe:animate-pulse' },
        ...['w-1/2', 'w-full', 'w-5/6', 'w-2/3', 'w-full'].map((w) => h('div', { class: `h-3 rounded bg-slate-200 ${w}` })),
      ),
    );
  }

  async function open(patientId, openerEl, fallbackName = 'Patient') {
    opener = openerEl ?? opener;
    controller?.abort();
    const mine = (controller = new AbortController());
    titleEl.textContent = fallbackName;
    subtitleEl.textContent = '';
    showLoading();
    if (!dialog.open) dialog.showModal();
    body.scrollTop = 0;

    try {
      const detail = await api.getPatient(patientId, mine.signal);
      if (controller !== mine) return;
      const p = detail.patient;
      titleEl.textContent = p.display_name;
      subtitleEl.textContent = [
        `MRN ${p.mrn}`,
        p.gender,
        p.birth_date ? `Born ${formatDate(p.birth_date)}` : 'Date of birth not recorded',
      ].join(' · ');
      clear(body).append(
        h(
          'div',
          { class: 'space-y-6' },
          p.data_quality_flags.length
            ? h(
                'div',
                { role: 'note', class: 'rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950' },
                h('p', { class: 'font-semibold' }, 'Data-quality flags from ingestion'),
                bulletList(p.data_quality_flags.map((f) => f.replaceAll('_', ' ').toLowerCase()), 'text-sm'),
              )
            : null,
          detail.summary
            ? summaryView(detail.summary)
            : h('p', { class: 'text-sm text-slate-700' }, 'No AI summary has been generated for this patient yet.'),
          resourcesView(api, detail, signalFor),
        ),
      );
    } catch (error) {
      if (error?.name === 'AbortError' || controller !== mine) return;
      clear(body).append(
        h(
          'div',
          { role: 'alert', class: 'rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-900' },
          h('p', { class: 'font-semibold' }, 'Could not load this patient'),
          h('p', { class: 'mt-1' }, error.message),
          h(
            'button',
            {
              type: 'button',
              class: 'mt-3 rounded-lg bg-indigo-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-800',
              onclick: () => open(patientId, opener, fallbackName),
            },
            'Try again',
          ),
        ),
      );
    } finally {
      if (controller === mine) body.setAttribute('aria-busy', 'false');
    }
  }

  return { open, close: () => dialog.close() };
}
