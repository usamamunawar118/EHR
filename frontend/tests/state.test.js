import assert from 'node:assert/strict';
import { test } from 'node:test';

import { createStore, hasFilters, initialState, validateCriteria } from '../public/js/state.js';
import { fromQueryString, toQueryString } from '../public/js/urlstate.js';

test('store notifies subscribers with merged state and supports unsubscribe', () => {
  const store = createStore({ a: 1, b: 2 });
  const seen = [];
  const off = store.subscribe((s) => seen.push(s));
  store.set({ b: 3 });
  off();
  store.set({ a: 9 });
  assert.deepEqual(seen, [{ a: 1, b: 3 }]);
  assert.deepEqual(store.get(), { a: 9, b: 3 });
});

test('the initial state is frozen so it cannot be mutated by accident', () => {
  assert.throws(() => {
    'use strict';
    initialState.query = 'x';
  }, TypeError);
});

test('validateCriteria', () => {
  assert.equal(validateCriteria({ query: '', dateFrom: '', dateTo: '' }), null);
  assert.equal(validateCriteria({ query: 'pneumonia', dateFrom: '2024-01-01', dateTo: '2024-01-01' }), null);
  assert.match(validateCriteria({ query: 'p', dateFrom: '', dateTo: '' }), /at least 2 characters/);
  assert.match(
    validateCriteria({ query: 'pneumonia', dateFrom: '2024-02-01', dateTo: '2024-01-01' }),
    /start date must be on or before/,
  );
});

test('hasFilters', () => {
  assert.equal(hasFilters({ resourceType: '', dateFrom: '', dateTo: '' }), false);
  assert.equal(hasFilters({ resourceType: 'Patient', dateFrom: '', dateTo: '' }), true);
  assert.equal(hasFilters({ resourceType: '', dateFrom: '2024-01-01', dateTo: '' }), true);
});

test('URL state round-trips', () => {
  const criteria = { query: 'high potassium', resourceType: 'DiagnosticReport', dateFrom: '2024-01-01', dateTo: '2024-12-31' };
  const qs = toQueryString(criteria);
  assert.deepEqual(fromQueryString(qs, ['DiagnosticReport']), criteria);
});

test('empty criteria give an empty query string', () => {
  assert.equal(toQueryString({}), '');
  assert.equal(toQueryString({ query: '   ' }), '');
});

test('URL state ignores malformed or unknown values instead of trusting them', () => {
  const parsed = fromQueryString('?q=ok&type=Banana&from=yesterday&to=2024-02-30x', ['Patient']);
  assert.deepEqual(parsed, { query: 'ok', resourceType: '', dateFrom: '', dateTo: '' });
});

test('URL state does not restrict types when none are supplied (offline start-up)', () => {
  assert.equal(fromQueryString('?type=Patient').resourceType, 'Patient');
});

test('very long queries from the URL are capped', () => {
  assert.equal(fromQueryString(`?q=${'a'.repeat(900)}`).query.length, 500);
});
