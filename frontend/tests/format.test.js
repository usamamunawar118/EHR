import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  badgeFor,
  describeGenerator,
  formatDate,
  groupResources,
  plural,
  redactAttachments,
  relevanceWord,
  scorePercent,
} from '../public/js/format.js';

test('formatDate renders calendar dates without time-zone drift', () => {
  assert.equal(formatDate('2025-03-04'), 'Mar 4, 2025');
  assert.equal(formatDate('2024-12-31'), 'Dec 31, 2024');
  assert.equal(formatDate('2025-01-01'), 'Jan 1, 2025');
});

test('formatDate handles missing and malformed input', () => {
  assert.equal(formatDate(null), 'Date unknown');
  assert.equal(formatDate(''), 'Date unknown');
  assert.equal(formatDate('not-a-date'), 'not-a-date');
});

test('scorePercent clamps and rounds', () => {
  assert.equal(scorePercent(0.731), 73);
  assert.equal(scorePercent(1.4), 100);
  assert.equal(scorePercent(-0.2), 0);
  assert.equal(scorePercent(undefined), 0);
  assert.equal(scorePercent('0.5'), 50);
});

test('relevanceWord thresholds', () => {
  assert.equal(relevanceWord(73), 'High');
  assert.equal(relevanceWord(45), 'Moderate');
  assert.equal(relevanceWord(12), 'Low');
});

test('summary hits are badged as AI summary regardless of resource type', () => {
  assert.equal(badgeFor({ kind: 'summary', resource_type: 'Patient' }).label, 'AI summary');
  assert.equal(badgeFor({ kind: 'record', resource_type: 'DiagnosticReport' }).label, 'Diagnostic report');
  assert.equal(badgeFor({ kind: 'record', resource_type: 'Mystery' }).label, 'Record');
  assert.equal(badgeFor().label, 'Record');
});

test('every badge states its type in text, not colour alone', () => {
  for (const type of ['DocumentReference', 'DiagnosticReport', 'Encounter', 'Observation', 'Patient']) {
    assert.ok(badgeFor({ kind: 'record', resource_type: type }).label.length > 0);
  }
});

test('groupResources orders by clinical importance and keeps unknown types last', () => {
  const groups = groupResources([
    { resource_type: 'Observation' },
    { resource_type: 'Zeta' },
    { resource_type: 'DocumentReference' },
    { resource_type: 'Encounter' },
    { resource_type: 'DocumentReference' },
  ]);
  assert.deepEqual(
    groups.map(([type, items]) => [type, items.length]),
    [
      ['DocumentReference', 2],
      ['Encounter', 1],
      ['Observation', 1],
      ['Zeta', 1],
    ],
  );
  assert.deepEqual(groupResources(undefined), []);
});

test('redactAttachments shortens base64 payloads without mutating the input', () => {
  const big = 'A'.repeat(500);
  const input = { content: [{ attachment: { data: big, size: 500, title: 'x' } }], data: 'short' };
  const out = redactAttachments(input);
  assert.match(out.content[0].attachment.data, /omitted, 500 characters/);
  assert.equal(out.content[0].attachment.title, 'x');
  assert.equal(out.data, 'short'); // short strings named "data" are left alone
  assert.equal(input.content[0].attachment.data, big);
});

test('plural', () => {
  assert.equal(plural(1, 'result'), '1 result');
  assert.equal(plural(5, 'result'), '5 results');
  assert.equal(plural(0, 'word'), '0 words');
});

test('describeGenerator is honest about the offline baseline', () => {
  assert.match(
    describeGenerator({ provider: 'fake', model: 'extractive-baseline-v2', generated_at: '2025-03-04T10:00:00Z' }),
    /no language model was used/,
  );
  assert.match(
    describeGenerator({ provider: 'anthropic', model: 'claude-haiku-5-5', generated_at: '2025-03-04T10:00:00Z' }),
    /claude-haiku-5-5 on Mar 4, 2025/,
  );
  assert.equal(describeGenerator(null), 'Generator unknown');
});
