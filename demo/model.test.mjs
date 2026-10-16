import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import {
  beliefIndex,
  cumulativeCost,
  pairedRecord,
  metricText
} from './model.js';

test('the displayed belief changes only at observation boundaries', () => {
  assert.equal(beliefIndex(15, 16, 6), 0);
  assert.equal(beliefIndex(16, 16, 6), 1);
  assert.equal(beliefIndex(80, 16, 6), 5);
});
test('the displayed task metric is trapezoidal executed clarity deficit', () => {
  assert.deepEqual(cumulativeCost([
    [0],
    [.5],
    [1]
  ], [.8], 1), [0, .55, .7000000000000001]);
  assert.equal(metricText(null), 'Infeasible');
});
test('comparisons require the same experiment group and physical-world seed', () => {
  const rows = [{
    group: 'g',
    seed: 1,
    method: 'proposed'
  }, {
    group: 'g',
    seed: 2,
    method: 'current_belief'
  }, {
    group: 'other',
    seed: 1,
    method: 'current_belief'
  }];
  assert.equal(pairedRecord(rows, rows[0]), null);
  rows.push({
    group: 'g',
    seed: 1,
    method: 'current_belief'
  });
  assert.deepEqual(pairedRecord(rows, rows[0]), rows[3]);
});
test('all bundled replays have coherent telemetry and displayed actual costs', () => {
  const catalog = JSON.parse(fs.readFileSync(new URL('./data/catalog.json', import.meta.url), 'utf8'));
  assert.ok(catalog.length);
  for (const item of catalog) {
    const data = JSON.parse(fs.readFileSync(new URL('../' + item.file, import.meta.url), 'utf8'));
    const c = cumulativeCost(data.mission.q, data.environment.targets, data.config.dt);
    assert.ok(Math.abs(c.at(-1) - data.metrics.integrated_deficit) < 2e-5);
    assert.equal(data.planning.length, data.config.mission_horizons);
    for (const p of data.planning) {
      assert.equal(p.costs.length, data.config.diffusion_steps);
      for (let i = 0; i < p.costs.length; i++) {
        const total = p.weights[i].reduce((a, b) => a + b, 0);
        assert.ok(Math.abs(total - 1) < 1e-5 || total === 0);
        assert.equal(p.candidate_paths[i].length, data.config.proposals);
        assert.equal(p.candidate_paths[i][0].length, data.config.stage_steps + data.config.future_steps +
          1);
        p.costs[i].forEach((cost, j) => {
          if (cost === null) assert.equal(p.weights[i][j], 0);
        });
      }
    }
  }
});
