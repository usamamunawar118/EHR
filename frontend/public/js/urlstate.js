// Keep the search in the address bar so a result set can be bookmarked or shared.

const KEYS = { q: 'query', type: 'resourceType', from: 'dateFrom', to: 'dateTo' };

/** criteria -> "?q=...&type=..." (empty string when nothing is set). */
export function toQueryString({ query = '', resourceType = '', dateFrom = '', dateTo = '' }) {
  const params = new URLSearchParams();
  if (query.trim()) params.set('q', query.trim());
  if (resourceType) params.set('type', resourceType);
  if (dateFrom) params.set('from', dateFrom);
  if (dateTo) params.set('to', dateTo);
  const qs = params.toString();
  return qs ? `?${qs}` : '';
}

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

/** "?q=..." -> criteria. Unknown or malformed values are ignored, never trusted. */
export function fromQueryString(search, allowedTypes = null) {
  const params = new URLSearchParams(search);
  const out = { query: '', resourceType: '', dateFrom: '', dateTo: '' };
  for (const [param, field] of Object.entries(KEYS)) {
    const value = (params.get(param) ?? '').trim();
    if (!value) continue;
    if ((field === 'dateFrom' || field === 'dateTo') && !ISO_DATE.test(value)) continue;
    if (field === 'resourceType' && allowedTypes && !allowedTypes.includes(value)) continue;
    out[field] = value.slice(0, 500);
  }
  return out;
}
