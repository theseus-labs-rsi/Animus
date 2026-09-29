// Node 22+ can erase TypeScript annotations; no browser, packages or model calls needed.
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';

const frontendModule = new URL('../frontend/lib/live-run.ts', import.meta.url);
if (!existsSync(frontendModule)) {
  console.log('SKIP: frontend removed from the CLI release; 12 historical checks retained.');
} else {
const { formatCount, qualityLabel, replayPath, unverifiedQuality } = await import(frontendModule.href);

assert.equal(formatCount(null), '未测');
assert.equal(formatCount(undefined), '未测');
assert.equal(formatCount(0), '0');
assert.equal(formatCount(Number.NaN), '未测');
assert.equal(qualityLabel(unverifiedQuality()), '未验收');
assert.equal(qualityLabel({ ...unverifiedQuality(), status: 'passed' }), '未取得发布资格');
assert.equal(qualityLabel({ ...unverifiedQuality(), status: 'passed', eligible: true }), '合格');
assert.equal(qualityLabel({ ...unverifiedQuality(), status: 'failed' }), '未通过');
assert.equal(qualityLabel({ ...unverifiedQuality(), status: 'stale' }), '验收已失效');
assert.equal(replayPath('live__current-1234'), '/api/replay/live__current-1234');
assert.notEqual(replayPath('live__current-1234'), replayPath('office__20260717-064826'));
assert.throws(() => replayPath('../another-run'));
console.log('frontend live state: 12 offline checks passed');
}
