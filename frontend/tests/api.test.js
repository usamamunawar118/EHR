import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ApiError, buildSearchRequest, bundleUrl, createApi, describeError } from '../public/js/api.js';

const jsonResponse = (body, { status = 200 } = {}) =>
  ({ ok: status >= 200 && status < 300, status, json: async () => body });

test('filters are query parameters and the query is the JSON body', () => {
  const { url, init } = buildSearchRequest({
    query: '  chest x-ray  ',
    resourceType: 'DiagnosticReport',
    dateFrom: '2024-01-01',
    dateTo: '2024-12-31',
  });
  assert.equal(init.method, 'POST');
  assert.deepEqual(JSON.parse(init.body), { query: 'chest x-ray' });
  const params = new URL(url, 'http://x').searchParams;
  assert.equal(params.get('resource_type'), 'DiagnosticReport');
  assert.equal(params.get('date_from'), '2024-01-01');
  assert.equal(params.get('date_to'), '2024-12-31');
});

test('no filters means a clean URL', () => {
  assert.equal(buildSearchRequest({ query: 'x', resourceType: '', dateFrom: '', dateTo: '' }).url, '/search');
});

test('describeError reads FastAPI string and list details', () => {
  assert.equal(describeError(404, { detail: 'patient not found' }), 'patient not found');
  assert.equal(
    describeError(422, { detail: [{ loc: ['body', 'query'], msg: 'String should have at least 1 character' }] }),
    'String should have at least 1 character',
  );
  assert.equal(
    describeError(422, { detail: [{ loc: ['query', 'date_from'], msg: 'Input should be a valid date' }] }),
    'date_from: Input should be a valid date',
  );
  assert.match(describeError(500, null), /server had a problem/);
  assert.equal(describeError(418, {}), 'Request failed (HTTP 418).');
});

test('search returns the parsed response and forwards the abort signal', async () => {
  let seen;
  const api = createApi({
    fetchImpl: async (url, init) => {
      seen = { url, init };
      return jsonResponse({ results: [], total: 0, took_ms: 3 });
    },
  });
  const controller = new AbortController();
  const result = await api.search({ query: 'abc' }, controller.signal);
  assert.equal(result.total, 0);
  assert.equal(seen.init.signal, controller.signal);
});

test('HTTP errors become ApiError with a readable message and status', async () => {
  const api = createApi({ fetchImpl: async () => jsonResponse({ detail: 'nope' }, { status: 422 }) });
  await assert.rejects(api.search({ query: 'abc' }), (error) => {
    assert.ok(error instanceof ApiError);
    assert.equal(error.status, 422);
    assert.equal(error.message, 'nope');
    return true;
  });
});

test('non-JSON error bodies still produce a message', async () => {
  const api = createApi({
    fetchImpl: async () => ({ ok: false, status: 502, json: async () => { throw new SyntaxError('bad json'); } }),
  });
  await assert.rejects(api.getPatient('p1'), /server had a problem/);
});

test('network failures are reported as a friendly ApiError', async () => {
  const api = createApi({ fetchImpl: async () => { throw new TypeError('Failed to fetch'); } });
  await assert.rejects(api.search({ query: 'abc' }), /Cannot reach the server/);
});

test('aborted requests propagate AbortError so callers can ignore them', async () => {
  const api = createApi({
    fetchImpl: async () => { throw new DOMException('aborted', 'AbortError'); },
  });
  await assert.rejects(api.search({ query: 'abc' }), (error) => error.name === 'AbortError');
});

test('ids are URL-encoded so they cannot alter the path', async () => {
  const urls = [];
  const api = createApi({ fetchImpl: async (url) => (urls.push(url), jsonResponse({})) });
  await api.getPatient('a/b?c');
  await api.getResource('p 1', '../x');
  assert.deepEqual(urls, ['/patients/a%2Fb%3Fc', '/patients/p%201/resources/..%2Fx']);
  assert.equal(bundleUrl('a/b'), '/patients/a%2Fb/bundle?download=true');
});
