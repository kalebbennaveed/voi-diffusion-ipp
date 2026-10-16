import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import {
  beliefIndex,
  cumulativeCost,
  pairedRecord,
  metricText,
  missionSummary,
  comparisonPair
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
  const catalog = JSON.parse(fs.readFileSync(new URL('./data/catalog.json', import.meta.url),
    'utf8'));
  assert.ok(catalog.length);
  for (const item of catalog) {
    const data = JSON.parse(fs.readFileSync(new URL('../' + item.file, import.meta.url),
      'utf8'));
    const c = cumulativeCost(data.mission.q, data.environment.targets, data.config.dt);
    const summary = missionSummary(data);
    assert.ok(Math.abs(summary.cost - summary.first - summary.later) < 1e-12);
    assert.equal(summary.relaxedSteps, data.mission.feasible.filter(v => !v).length);
    assert.equal(summary.peakSlack, data.metrics.max_qp_slack);
    assert.ok(Math.abs(c.at(-1) - data.metrics.integrated_deficit) < 2e-5);
    assert.equal(data.planning.length, data.config.mission_horizons);
    for (const p of data.planning) {
      assert.equal(p.costs.length, data.config.diffusion_steps);
      for (let i = 0; i < p.costs.length; i++) {
        const total = p.weights[i].reduce((a, b) => a + b, 0);
        assert.ok(Math.abs(total - 1) < 1e-5 || total === 0);
        assert.equal(p.candidate_paths[i].length, data.config.proposals);
        assert.equal(p.candidate_paths[i][0].length, data.config.stage_steps + data.config
          .future_steps +
          1);
        p.costs[i].forEach((cost, j) => {
          if (cost === null) assert.equal(p.weights[i][j], 0);
        });
      }
    }
  }
});

test('soft slack is reported independently of collision and task cost', () => {
  const summary = missionSummary({
    config: {
      dt: 0.5,
      stage_steps: 1
    },
    environment: {
      targets: [1]
    },
    metrics: {
      collision: false,
      roi_a_access: false,
      roi_a_time: null
    },
    mission: {
      q: [
        [0],
        [0.5],
        [1]
      ],
      x: [
        [0],
        [1],
        [2]
      ],
      slack: [0.4, 0.6],
      clearance: [0.2, 0.1]
    }
  });
  assert.equal(summary.cost, 0.5);
  assert.equal(summary.first, 0.375);
  assert.equal(summary.later, 0.125);
  assert.equal(summary.peakSlack, 0.6);
  assert.equal(summary.integratedSlack, 0.5);
  assert.equal(summary.relaxedSeconds, 1);
  assert.equal(summary.collision, false);
  assert.equal(summary.accessTime, null);
});

test('comparison orientation never reverses the proposed and baseline identities', () => {
  const proposed = {
    metrics: {
      method: 'proposed'
    }
  };
  const baseline = {
    metrics: {
      method: 'current_belief'
    }
  };
  assert.deepEqual(comparisonPair(proposed, baseline), [proposed, baseline]);
  assert.deepEqual(comparisonPair(baseline, proposed), [proposed, baseline]);
  assert.deepEqual(comparisonPair(baseline, null), [baseline, null]);
});

test('the final comparison uses matched physical worlds and preserves cost decomposition', () => {
  const catalog = JSON.parse(fs.readFileSync(new URL('./data/catalog.json', import.meta.url),
    'utf8'));
  for (const row of catalog.filter(r => r.method === 'proposed')) {
    const other = pairedRecord(catalog, row);
    if (!other) continue;
    const [a, b] = [row, other].map(r => JSON.parse(fs.readFileSync(new URL('../' + r.file,
      import.meta.url), 'utf8')));
    assert.deepEqual(a.physical_wind, b.physical_wind);
    assert.deepEqual(a.mission.x[0], b.mission.x[0]);
    const sa = missionSummary(a),
      sb = missionSummary(b);
    const initialSacrifice = sa.first - sb.first;
    const laterRecovery = sb.later - sa.later;
    assert.ok(Math.abs((laterRecovery - initialSacrifice) - (sb.cost - sa.cost)) < 1e-12);
  }
});
