import {
  clamp,
  deficit,
  cumulativeCost,
  beliefIndex,
  pairedRecord,
  metricText
} from './model.js';

const $ = id => document.getElementById(id);
const state = {
  data: null,
  comparison: null,
  records: [],
  record: null,
  mode: 'diffusion',
  period: 0,
  level: 0,
  step: 0,
  playing: false,
  accumulator: 0,
  last: 0,
  hover: -1,
  loading: false,
  cost: []
};
const colors = {
  ink: '#223f50',
  teal: '#087f83',
  violet: '#8171b0',
  gray: '#a4b2bb',
  amber: '#b27738',
  red: '#bb514c'
};

function status(text, kind = 'ready') {
  $('status').textContent = text;
  $('status-dot').className = 'dot' + (kind === 'busy' ? ' busy' : kind === 'error' ? ' error' : '');
}

function surface(id) {
  const c = $(id),
    r = c.getBoundingClientRect(),
    dpr = window.devicePixelRatio || 1;
  const w = r.width,
    h = r.height;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
  }
  const g = c.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return {
    c,
    g,
    w,
    h
  };
}

function path(g, points, xy, color, width = 1, dash = []) {
  if (!points?.length) return;
  g.strokeStyle = color;
  g.lineWidth = width;
  g.setLineDash(dash);
  g.beginPath();
  points.forEach((p, i) => {
    const [x, y] = xy(p);
    if (i) g.lineTo(x, y);
    else g.moveTo(x, y);
  });
  g.stroke();
  g.setLineDash([]);
}

function arrow(g, a, b, color, width = 1) {
  g.strokeStyle = color;
  g.lineWidth = width;
  g.beginPath();
  g.moveTo(...a);
  g.lineTo(...b);
  const t = Math.atan2(b[1] - a[1], b[0] - a[0]);
  g.moveTo(b[0] - 4 * Math.cos(t - .5), b[1] - 4 * Math.sin(t - .5));
  g.lineTo(...b);
  g.lineTo(b[0] - 4 * Math.cos(t + .5), b[1] - 4 * Math.sin(t + .5));
  g.stroke();
}

function label(g, text, x, y, color = '#6e7e88', size = 10, align = 'left') {
  g.fillStyle = color;
  g.font = `${size}px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif`;
  g.textAlign = align;
  g.fillText(text, x, y);
}

function active() {
  const d = state.data,
    H = d.config.stage_steps;
  return {
    d,
    H,
    p: d.planning[state.period],
    t: state.mode === 'mission' ? state.step : state.period * H
  };
}

function drawMap() {
  const {
    g,
    w,
    h
  } = surface('map');
  if (!state.data) return;
  const {
    d,
    H,
    p,
    t
  } = active(), e = d.environment, b = e.bounds;
  const scale = Math.min((w - 95) / (b[1][0] - b[0][0]), (h - 112) / (b[1][1] - b[0][1]));
  const left = (w - scale * (b[1][0] - b[0][0])) / 2,
    top = (h - scale * (b[1][1] - b[0][1])) / 2 + 13;
  const xy = a => [left + (a[0] - b[0][0]) * scale, top + (b[1][1] - a[1]) * scale];
  const belief = d.beliefs[beliefIndex(t, H, d.beliefs.length)];
  g.save();
  g.beginPath();
  g.rect(left, top, (b[1][0] - b[0][0]) * scale, (b[1][1] - b[0][1]) * scale);
  g.clip();
  g.fillStyle = '#f9fbfb';
  g.fillRect(0, 0, w, h);
  if ($('uncertainty').checked) {
    d.wind_grid.forEach((v, i) => {
      const a = xy(v),
        alpha = clamp(belief.std[i] / Math.max(.8, d.config.wind_std), 0, 1) * .25;
      g.fillStyle = `rgba(39,123,151,${alpha})`;
      g.fillRect(a[0] - scale * .105, a[1] - scale * .105, scale * .21, scale * .21);
    });
  }
  for (let x = -2; x <= 2; x += .5) {
    const a = xy([x, b[0][1]]),
      z = xy([x, b[1][1]]);
    path(g, [a, z], v => v, '#dce5e877', .7);
  }
  for (let y = -1.5; y <= 1.5; y += .5) {
    path(g, [xy([b[0][0], y]), xy([b[1][0], y])], v => v, '#dce5e877', .7);
  }
  if ($('wind').checked || $('truth').checked) {
    const vectors = $('truth').checked ? d.physical_wind : belief.mean;
    d.wind_grid.forEach((v, i) => {
      if (i % 2 || Math.floor(i / 25) % 2) return;
      const m = vectors[i];
      if (Math.hypot(...m) < .012) return;
      arrow(g, xy(v), xy([v[0] + m[0] * .27, v[1] + m[1] * .27]), $('truth').checked ? '#b27738aa' :
        '#457d88aa', .9);
    });
  }
  e.obstacles.forEach(o => {
    const a = xy(o);
    g.beginPath();
    g.arc(...a, o[2] * scale, 0, Math.PI * 2);
    g.fillStyle = '#d9dfe3';
    g.fill();
    g.strokeStyle = '#a6b3bd';
    g.lineWidth = 1;
    g.stroke();
    g.setLineDash([3, 4]);
    g.beginPath();
    g.arc(...a, (o[2] + d.config.safety_margin) * scale, 0, Math.PI * 2);
    g.strokeStyle = '#a6b3bd88';
    g.stroke();
    g.setLineDash([]);
  });
  const q = d.mission.q[t];
  e.cells.forEach((c, i) => {
    const a = xy(c),
      v = clamp(q[i] / e.targets[i], 0, 1);
    g.fillStyle = `rgb(${Math.round(228-202*v)},${Math.round(235-96*v)},${Math.round(233-97*v)})`;
    g.fillRect(a[0] - 6, a[1] - 6, 12, 12);
    g.strokeStyle = '#ffffffee';
    g.strokeRect(a[0] - 6, a[1] - 6, 12, 12);
  });
  if ($('compare').checked && state.comparison) {
    const c = state.comparison;
    path(g, c.mission.x.slice(0, t + 1), xy, colors.amber, 1.8, [5, 4]);
    if (state.mode === 'mission') {
      const a = xy(c.mission.x[t]);
      g.strokeStyle = colors.amber;
      g.lineWidth = 1.5;
      g.beginPath();
      g.arc(...a, 5, 0, 7);
      g.stroke();
    }
  }
  if (state.mode === 'diffusion') {
    const l = state.level,
      paths = p.candidate_paths[l],
      weights = p.weights[l],
      max = Math.max(...weights, .001);
    paths.forEach((v, i) => {
      if (state.hover >= 0 && i !== state.hover) return;
      const alpha = .25 + .45 * weights[i] / max;
      path(g, v, xy, p.costs[l][i] === null ? `rgba(187,81,76,${alpha})` : `rgba(71,115,135,${alpha})`,
        state.hover === i ? 2.5 : .8 + 1.3 * weights[i] / max);
    });
    path(g, p.mean_paths[l], xy, colors.violet, 1.8, [5, 4]);
    path(g, l === d.config.diffusion_steps - 1 ? p.selected_path : p.incumbent_paths[l], xy, colors.teal,
      2.7);
    const selected = l === d.config.diffusion_steps - 1 ? p.selected_path : p.incumbent_paths[l];
    const boundary = xy(selected[H]);
    g.fillStyle = '#fff';
    g.strokeStyle = colors.teal;
    g.lineWidth = 1.5;
    g.beginPath();
    g.arc(...boundary, 3.5, 0, Math.PI * 2);
    g.fill();
    g.stroke();
    const start = d.mission.x[state.period * H];
    path(g, d.mission.x.slice(0, state.period * H + 1), xy, colors.ink, 1.7);
    const a = xy(start);
    g.fillStyle = colors.ink;
    g.beginPath();
    g.arc(...a, 5, 0, 7);
    g.fill();
    if (state.hover >= 0) {
      path(g, paths[state.hover], xy, p.costs[l][state.hover] === null ? colors.red : colors.ink, 2.6);
    }
  } else {
    const period = Math.min(Math.floor(t / H), d.planning.length - 1);
    path(g, d.mission.nominal_paths[period], xy, '#8599a7', 1.1, [3, 4]);
    path(g, d.mission.x.slice(0, t + 1), xy, colors.teal, 2.8);
    for (let i = H; i <= t; i += H) {
      const a = xy(d.mission.x[i]);
      g.fillStyle = '#fff';
      g.strokeStyle = colors.teal;
      g.lineWidth = 1.4;
      g.beginPath();
      g.arc(...a, 3, 0, 7);
      g.fill();
      g.stroke();
    }
    const x = d.mission.x[t],
      a = xy(x);
    g.fillStyle = colors.ink;
    g.beginPath();
    g.arc(...a, 5, 0, 7);
    g.fill();
    arrow(g, a, xy([x[0] + x[2] * .23, x[1] + x[3] * .23]), colors.ink, 1.4);
    if (t > 0 && !d.mission.feasible[t - 1]) {
      g.strokeStyle = colors.red;
      g.lineWidth = 2;
      g.beginPath();
      g.arc(...a, 11, 0, 7);
      g.stroke();
    }
  }
  if ($('observations').checked) {
    d.mission.x.slice(1, t + 1).forEach(x => {
      const a = xy(x);
      g.fillStyle = '#223f5077';
      g.fillRect(a[0] - 1, a[1] - 1, 2, 2);
    });
  }
  g.restore();
  for (let x = -2; x <= 2; x++) {
    const a = xy([x, b[0][1]]);
    label(g, x.toFixed(0), a[0], a[1] + 18, '#8898a1', 9, 'center');
  }
  for (let y = -1; y <= 1; y++) {
    const a = xy([b[0][0], y]);
    label(g, y.toFixed(0), a[0] - 12, a[1] + 3, '#8898a1', 9, 'right');
  }
  for (const [name, key, dy] of [
      ['ROI A', 'roi_a', -.35],
      ['ROI B', 'roi_b', -.34]
    ]) {
    const a = xy(e[key]);
    label(g, name, a[0], a[1] + dy * scale, colors.ink, 10, 'center');
  }
  const probe = xy(e.probe);
  g.strokeStyle = '#668491';
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(probe[0] - 4, probe[1]);
  g.lineTo(probe[0] + 4, probe[1]);
  g.moveTo(probe[0], probe[1] - 4);
  g.lineTo(probe[0], probe[1] + 4);
  g.stroke();
  if ($('uncertainty').checked) {
    const y = h - 14;
    label(g, 'Wind σ', 18, y, '#788d99', 8);
    for (let i = 0; i < 50; i++) {
      g.fillStyle = `rgba(39,123,151,${i/50*.3})`;
      g.fillRect(56 + i, y - 7, 1, 7);
    }
    label(g, `0 → ${Math.max(.8,d.config.wind_std).toFixed(1)}`, 111, y, '#788d99', 8);
  }
}

function chartAxes(id, xmax, ymax, ylabel) {
  const s = surface(id),
    {
      g,
      w,
      h
    } = s;
  const box = {
    left: 35,
    right: w - 12,
    top: 9,
    bottom: h - 23
  };
  const xy = ([x, y]) => [box.left + x / Math.max(xmax, .001) * (box.right - box.left), box.bottom - y / Math
    .max(ymax, .001) * (box.bottom - box.top)
  ];
  for (let i = 0; i <= 2; i++) {
    const y = ymax * i / 2;
    path(g, [
      [0, y],
      [xmax, y]
    ], xy, '#e5eaed', .8);
    label(g, y.toFixed(ymax < 1 ? 2 : 1), box.left - 7, xy([0, y])[1] + 3, '#8798a2', 8, 'right');
  }
  label(g, '0', box.left, h - 8, '#8798a2', 8);
  label(g, ylabel, box.right, h - 8, '#8798a2', 8, 'right');
  return {
    ...s,
    xy,
    box
  };
}

function drawCharts() {
  if (!state.data) return;
  const {
    d,
    p
  } = active();
  if (state.mode === 'diffusion') {
    const finite = p.trace.flatMap(r => r.slice(0, 2)).filter(Number.isFinite),
      max = Math.max(...finite, 1) * 1.06;
    const a = chartAxes('cost-chart', Math.max(1, p.trace.length - 1), max, 'reverse updates');
    for (const [col, color, dash] of [
        [1, '#b4c1c7', [3, 3]],
        [0, colors.teal, []]
      ]) path(a.g, p.trace.map((r, i) => [i, r[col]]).filter(v => Number.isFinite(v[1])), a.xy, color, 1.7,
      dash);
    const current = p.trace[state.level][0];
    if (Number.isFinite(current)) {
      const xy = a.xy([state.level, current]);
      a.g.fillStyle = colors.teal;
      a.g.beginPath();
      a.g.arc(...xy, 4, 0, 7);
      a.g.fill();
    }
    const {
      g,
      w,
      h
    } = surface('weights-chart'), weights = p.weights[state.level], maxw = Math.max(...weights, .01), bw = (
      w - 22) / weights.length;
    weights.forEach((v, i) => {
      g.fillStyle = state.hover === i ? colors.ink : p.costs[state.level][i] === null ? colors.red :
        colors.teal;
      g.globalAlpha = state.hover >= 0 && state.hover !== i ? .3 : .8;
      const barh = Math.max(v > 0 ? 2 : 1, v / maxw * (h - 31));
      g.fillRect(10 + i * bw, h - 20 - barh, Math.max(1, bw - 4), barh);
      label(g, String(i + 1), 10 + i * bw + (bw - 4) / 2, h - 7, '#8798a2', 8, 'center');
    });
    g.globalAlpha = 1;
    label(g, `max weight ${maxw.toFixed(2)}`, w - 6, 10, '#8798a2', 8, 'right');
  } else {
    const ts = d.mission.q.map((q, i) => [i * d.config.dt, deficit(q, d.environment.targets)]),
      max = Math.max(...ts.map(v => v[1]), .65) * 1.08,
      a = chartAxes('cost-chart', ts.at(-1)[0], max, 'mission seconds');
    if (state.comparison && $('compare').checked) path(a.g, state.comparison.mission.q.map((q, i) => [i * d
      .config.dt, deficit(q, state.comparison.environment.targets)
    ]), a.xy, colors.amber, 1.4, [4, 3]);
    path(a.g, ts, a.xy, colors.teal, 1.8);
    path(a.g, [
      [state.step * d.config.dt, 0],
      [state.step * d.config.dt, max]
    ], a.xy, '#8397a3', 1, [2, 3]);
    const c = d.mission.correction.map((v, i) => [(i + 1) * d.config.dt, v]),
      b = chartAxes('weights-chart', ts.at(-1)[0], Math.max(...c.map(v => v[1]), .1) * 1.1,
      'mission seconds');
    path(b.g, c, b.xy, colors.ink, 1.4);
    d.mission.feasible.forEach((v, i) => {
      if (!v) {
        const p = b.xy(c[i]);
        b.g.fillStyle = colors.red;
        b.g.beginPath();
        b.g.arc(...p, 3.5, 0, 7);
        b.g.fill();
      }
    });
    path(b.g, [
      [state.step * d.config.dt, 0],
      [state.step * d.config.dt, Math.max(...c.map(v => v[1]), .1) * 1.1]
    ], b.xy, '#8397a3', 1, [2, 3]);
  }
}

function setMetric(i, title, value, detail) {
  $(`metric-label-${i}`).textContent = title;
  $(`metric-${i}`).textContent = value;
  $(`metric-detail-${i}`).textContent = detail;
}

function update() {
  if (!state.data) return;
  const {
    d,
    H,
    p,
    t
  } = active(), isDiff = state.mode === 'diffusion';
  $('diffusion-tab').classList.toggle('active', isDiff);
  $('mission-tab').classList.toggle('active', !isDiff);
  $('diffusion-tab').setAttribute('aria-selected', String(isDiff));
  $('mission-tab').setAttribute('aria-selected', String(!isDiff));
  $('play').textContent = state.playing ? 'Ⅱ' : '▶';
  $('play').setAttribute('aria-label', state.playing ? 'Pause playback' : 'Play playback');
  $('timeline').max = isDiff ? d.config.diffusion_steps - 1 : d.mission.x.length - 1;
  $('timeline').value = isDiff ? state.level : state.step;
  $('timeline-label').textContent = isDiff ? 'Reverse diffusion level' : 'Executed mission time';
  $('timeline-value').textContent = isDiff ? `${state.level+1} / ${d.config.diffusion_steps}` :
    `${(state.step*d.config.dt).toFixed(2)} / ${((d.mission.x.length-1)*d.config.dt).toFixed(1)} s`;
  $('period-status').textContent = `Period ${state.period+1} of ${d.planning.length}`;
  $('previous-period').disabled = state.period === 0;
  $('next-period').disabled = state.period === d.planning.length - 1;
  $('variant-tag').textContent = d.config.future_updates ? 'Repeated belief updates' : 'Two-stage';
  $('provenance').textContent =
    `Seed ${d.metrics.seed} · ${d.metrics.method} · JAX ${d.metrics.jax_version} · ${d.metrics.mean_plan_seconds.toFixed(2)} s / plan · ${d.metrics.objective_evaluations_per_plan} objective evaluations`;
  if (isDiff) {
    const l = state.level,
      r = p.trace[l],
      noise = Math.sqrt((1 - p.alphabar[l]) / p.alphabar[l]);
    $('map-title').textContent = d.metrics.method === 'cem' ? 'Cross-entropy optimization' : d.metrics
      .method === 'mppi' ? 'MPPI optimization' : 'Denoising the policy';
    $('map-subtitle').textContent =
      `Plan at t = ${(state.period*H*d.config.dt).toFixed(1)} s · planning world 1 of ${d.config.worlds}`;
    setMetric(1, l === d.config.diffusion_steps - 1 ? 'Final selected cost' : 'Best predicted cost',
      metricText(l === d.config.diffusion_steps - 1 ? p.selected_cost : r[0]), l === d.config
      .diffusion_steps - 1 ? 'Includes evaluation of the final denoised policy' :
      'Task deficit over the planning window');
    setMetric(2, d.metrics.method === 'cem' || d.metrics.method === 'mppi' ? 'Iteration' : 'Proposal noise σ',
      d.metrics.method === 'cem' || d.metrics.method === 'mppi' ? `${l+1}` : noise.toFixed(3),
      'Normalized policy coordinates');
    setMetric(3, 'Effective samples', r[2] > d.config.proposals ? '0.0' : r[2].toFixed(1),
      `Of ${d.config.proposals} sampled policies`);
    setMetric(4, 'Invalid proposals', `${Math.round(r[3]*100)}%`, 'Zero importance weight');
    $('chart-title-1').textContent = 'Optimization progress';
    $('chart-note-1').textContent = 'Best (teal) · population mean (gray)';
    $('chart-title-2').textContent = 'Candidate weights';
    $('chart-note-2').textContent = 'Hover to inspect a policy';
    $('explanation-title').textContent = 'From noisy policies to useful motion';
    $('explanation-text').textContent =
      'Each line is a safety-filtered rollout of a sampled policy. Predicted task cost determines its weight. The weighted policy mean estimates the score used by the next reverse update.';
    $('formula').textContent = d.metrics.method === 'cem' ? 'elite policies → refit distribution' : d.metrics
      .method === 'mppi' ? 'cost weights → weighted policy update' :
      'sample → simulate → weight → reverse update';
    $('forecast-note').textContent =
      'Paths show planning world 1. Costs use all outer worlds. The purple denoised mean may be infeasible; execution uses the best evaluated policy.';
    $('legend').innerHTML =
      '<span><i class="line teal"></i>Best evaluated</span><span><i class="line purple dashed"></i>Denoised mean</span><span><i class="line gray"></i>Sampled policies</span><span><i class="square"></i>Monitoring clarity</span>';
    if (state.hover >= 0) {
      const i = state.hover;
      $('candidate-detail').textContent =
        `Policy ${i+1} · cost ${metricText(p.costs[l][i])} · weight ${p.weights[l][i].toFixed(4)} · branch ${p.candidate_branches[l][i]+1}`;
    } else $('candidate-detail').textContent =
      `Score norm ${Math.hypot(...p.score[l]).toFixed(2)} · costs include the complete stochastic objective.`;
  } else {
    const idx = state.step - 1,
      fail = idx >= 0 && !d.mission.feasible[idx];
    $('map-title').textContent = 'Executed monitoring mission';
    $('map-subtitle').textContent =
      `Belief updated ${Math.floor(t/H)} times · ${fail?'QP relaxation active':'available-belief safety filter'}`;
    setMetric(1, 'Accumulated task cost', state.cost[t].toFixed(3), 'Actual executed clarity deficit');
    setMetric(2, 'Current clarity deficit', deficit(d.mission.q[t], d.environment.targets).toFixed(3),
      'Lower is better');
    setMetric(3, 'Obstacle clearance', idx < 0 ? '—' : `${d.mission.clearance[idx].toFixed(3)} m`,
      'Measured within the current control step');
    setMetric(4, 'Safety filter', fail ? 'Relaxed' : 'Feasible', idx < 0 ? 'Initial state' :
      `Control correction ${d.mission.correction[idx].toFixed(3)}`);
    $('metric-4').style.color = fail ? colors.red : colors.ink;
    $('chart-title-1').textContent = 'Monitoring deficit';
    $('chart-note-1').textContent = 'Actual mission · lower is better';
    $('chart-title-2').textContent = 'Safety interventions';
    $('chart-note-2').textContent = 'Control correction norm';
    $('candidate-detail').textContent =
      `${d.mission.feasible.filter(x=>!x).length} infeasible QP steps in this run · red dots mark relaxation`;
    $('explanation-title').textContent = 'Information has to earn its cost';
    $('explanation-text').textContent =
      'The robot gathers wind measurements along its executed path. The posterior changes only at period boundaries. That belief changes subsequent safety constraints and monitoring decisions.';
    $('formula').textContent = 'execute → observe → update belief → replan';
    $('forecast-note').textContent =
      'Teal is the executed path; gray dashes are the nominal prediction. The amber comparator uses the same physical wind and real observations, but freezes future belief during planning.';
    $('legend').innerHTML =
      '<span><i class="line teal"></i>Executed</span><span><i class="line gray"></i>Nominal prediction</span><span><i class="line amber"></i>Paired comparator</span><span><i class="square"></i>Monitoring clarity</span>';
  }
  if (isDiff) $('metric-4').style.color = colors.ink;
  drawMap();
  drawCharts();
}

function setMode(mode) {
  state.mode = mode;
  state.playing = false;
  state.hover = -1;
  state.accumulator = 0;
  if (mode === 'mission') state.step = state.period * state.data.config.stage_steps;
  update();
}

function setPeriod(period) {
  state.period = clamp(period, 0, state.data.planning.length - 1);
  state.level = 0;
  state.step = state.period * state.data.config.stage_steps;
  state.playing = false;
  state.hover = -1;
  update();
}
async function catalog() {
  const response = await fetch('/api/catalog');
  if (!response.ok) throw Error('Cannot load local experiment catalog');
  state.records = await response.json();
  const selected = $('recording').value;
  $('recording').replaceChildren(...state.records.map(r => {
    const o = document.createElement('option');
    o.value = r.id;
    o.textContent = r.label;
    return o;
  }));
  if (state.records.some(r => r.id === selected)) $('recording').value = selected;
}
async function replay(id) {
  if (state.loading) return;
  state.loading = true;
  state.playing = false;
  status('Loading experiment telemetry…', 'busy');
  try {
    const record = state.records.find(r => r.id === id);
    if (!record) throw Error('Unknown recording');
    const response = await fetch(`/api/replay/${encodeURIComponent(id)}`);
    if (!response.ok) throw Error('Replay data unavailable');
    const data = await response.json();
    state.record = record;
    state.data = data;
    state.comparison = null;
    state.period = 0;
    state.level = 0;
    state.step = 0;
    state.hover = -1;
    state.cost = cumulativeCost(data.mission.q, data.environment.targets, data.config.dt);
    const paired = pairedRecord(state.records, record);
    if (paired) {
      const other = await fetch(`/api/replay/${encodeURIComponent(paired.id)}`);
      if (other.ok) state.comparison = await other.json();
    }
    let summary = 'No matched comparator for this recording.';
    $('comparison-summary').classList.remove('regression');
    if (state.comparison) {
      const own = data.metrics.integrated_deficit,
        other = state.comparison.metrics.integrated_deficit;
      const improvement = (other - own) / other * 100;
      $('comparison-summary').classList.toggle('regression', improvement < 0);
      summary =
        `<strong>${Math.abs(improvement).toFixed(1)}% ${improvement>=0?'lower':'higher'} mission cost</strong> than the paired ${state.comparison.metrics.method==='current_belief'?'frozen-belief':'proposed'} planner.<br>Same physical wind · seed ${data.metrics.seed}`;
    }
    $('comparison-summary').innerHTML = summary;
    $('run-diagnostics').textContent =
      `Full mission: ${data.metrics.collision?'collision observed':'no collisions'} · ${data.mission.feasible.filter(v=>!v).length} relaxed QP steps · ${data.metrics.optimizer_failure_count} optimizer fallbacks`;
    $('run-diagnostics').style.color = data.metrics.collision || data.metrics.optimizer_failure_count ||
      data.mission.feasible.some(v => !v) ? colors.red : '';
    $('compare').disabled = !state.comparison;
    $('record-type').textContent = record.live ? 'New local run' : 'Recorded';
    $('recording').value = id;
    $('method').value = data.metrics.method;
    $('seed').value = data.metrics.seed;
    $('variant').value = data.config.future_updates ? 'multistage' : 'two_stage';
    $('wind-std').value = data.config.wind_std;
    $('observation-std').value = data.config.observation_std;
    $('measurements').checked = data.config.measurements;
    $('levels').value = String(data.config.diffusion_steps);
    $('proposals').value = String(data.config.proposals);
    status(`${record.live?'Local experiment':'Recorded experiment'} ready`);
    update();
  } catch (e) {
    status(e.message, 'error');
  } finally {
    state.loading = false;
  }
}
async function run() {
  const data = {
    method: $('method').value,
    seed: Number($('seed').value),
    variant: $('variant').value,
    wind_std: Number($('wind-std').value),
    observation_std: Number($('observation-std').value),
    measurements: $('measurements').checked,
    diffusion_steps: Number($('levels').value),
    proposals: Number($('proposals').value)
  };
  $('run').disabled = true;
  status('Starting JAX experiment…', 'busy');
  try {
    const response = await fetch('/api/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
      }),
      job = await response.json();
    if (!response.ok) throw Error(job.error || 'Experiment could not start');
    async function poll() {
      try {
        const response = await fetch(`/api/job/${job.id}`),
          result = await response.json();
        if (result.status === 'failed') throw Error(result.message);
        if (result.status === 'complete') {
          await catalog();
          await replay(result.record.id);
          $('run').disabled = false;
          return;
        }
        status(result.message, 'busy');
        setTimeout(poll, 1500);
      } catch (e) {
        status(e.message, 'error');
        $('run').disabled = false;
      }
    }
    poll();
  } catch (e) {
    status(e.message, 'error');
    $('run').disabled = false;
  }
}
$('diffusion-tab').onclick = () => state.data && setMode('diffusion');
$('mission-tab').onclick = () => state.data && setMode('mission');
$('play').onclick = () => {
  if (state.data) {
    state.playing = !state.playing;
    state.accumulator = 0;
    update();
  }
};
$('reset').onclick = () => state.data && setPeriod(0);
$('previous-period').onclick = () => setPeriod(state.period - 1);
$('next-period').onclick = () => setPeriod(state.period + 1);
$('timeline').oninput = () => {
  state.playing = false;
  state.hover = -1;
  if (state.mode === 'diffusion') state.level = Number($('timeline').value);
  else {
    state.step = Number($('timeline').value);
    state.period = Math.min(Math.floor(state.step / state.data.config.stage_steps), state.data.planning
      .length - 1);
  }
  update();
};
$('recording').onchange = () => replay($('recording').value);
$('run').onclick = run;
for (const id of ['compare', 'uncertainty', 'wind', 'observations', 'truth']) $(id).onchange = update;
$('weights-chart').onpointermove = e => {
  if (!state.data || state.mode !== 'diffusion') return;
  const r = e.currentTarget.getBoundingClientRect();
  state.hover = clamp(Math.floor((e.clientX - r.left - 10) / ((r.width - 22) / state.data.config
    .proposals)), 0, state.data.config.proposals - 1);
  update();
};
$('weights-chart').onpointerleave = () => {
  state.hover = -1;
  update();
};
window.addEventListener('resize', update);
window.addEventListener('keydown', e => {
  if (e.code === 'Space' && !['INPUT', 'SELECT', 'BUTTON', 'TEXTAREA'].includes(e.target.tagName)) {
    e.preventDefault();
    $('play').click();
  }
});

function animate(now) {
  const elapsed = Math.min((now - state.last) / 1000, .1);
  state.last = now;
  if (state.playing && state.data) {
    state.accumulator += elapsed * Number($('speed').value);
    const interval = state.mode === 'diffusion' ? .9 : state.data.config.dt;
    let changed = false;
    while (state.accumulator >= interval && state.playing) {
      state.accumulator -= interval;
      changed = true;
      const d = state.data,
        H = d.config.stage_steps;
      if (state.mode === 'diffusion') {
        if (state.level < d.config.diffusion_steps - 1) state.level++;
        else if ($('sequence').checked) {
          state.mode = 'mission';
          state.step = state.period * H;
          state.accumulator = 0;
          break;
        } else state.playing = false;
      } else if (state.step < d.mission.x.length - 1) {
        state.step++;
        if ($('sequence').checked && state.step % H === 0 && state.step < d.mission.x.length - 1) {
          state.period = Math.floor(state.step / H);
          state.mode = 'diffusion';
          state.level = 0;
          state.accumulator = 0;
          break;
        }
        state.period = Math.min(Math.floor(state.step / H), d.planning.length - 1);
      } else state.playing = false;
    }
    if (changed) update();
  }
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
async function initialize() {
  try {
    await catalog();
    if (!state.records.length) {
      status('Preparing the first recorded experiment…', 'busy');
      setTimeout(initialize, 2500);
      return;
    }
    const requested = new URLSearchParams(location.search).get('run');
    await replay(state.records.some(r => r.id === requested) ? requested : state.records[0].id);
  } catch (e) {
    status(e.message, 'error');
  }
}
initialize();
