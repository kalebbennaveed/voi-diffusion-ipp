import {
  clamp,
  deficit,
  cumulativeCost,
  beliefIndex,
  pairedRecord,
  metricText,
  methodLabel,
  comparisonPair,
  missionSummary,
  persistenceMetrics,
  playbackFrame,
  advanceComparison
} from './model.js';

const $ = id => document.getElementById(id);
const state = {
  data: null,
  comparison: null,
  records: [],
  record: null,
  mode: 'comparison',
  comparisonClock: 0,
  period: 0,
  level: 0,
  step: 0,
  playing: false,
  accumulator: 0,
  last: 0,
  hover: -1,
  cell: null,
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
const summaryCache = new WeakMap();
const persistenceCache = new WeakMap();

function persistence(data) {
  if (!persistenceCache.has(data)) persistenceCache.set(data, persistenceMetrics(data));
  return persistenceCache.get(data);
}

function summary(data) {
  if (!summaryCache.has(data)) summaryCache.set(data, missionSummary(data));
  return summaryCache.get(data);
}

function status(text, kind = 'ready') {
  $('status').textContent = text;
  $('status-dot').className = 'dot' + (kind === 'busy' ? ' busy' : kind === 'error' ? ' error' :
    '');
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
  const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
  if (length < .4) return;
  const head = Math.min(4, length * .45);
  g.strokeStyle = color;
  g.lineWidth = width;
  g.beginPath();
  g.moveTo(...a);
  g.lineTo(...b);
  const t = Math.atan2(b[1] - a[1], b[0] - a[0]);
  g.moveTo(b[0] - head * Math.cos(t - .5), b[1] - head * Math.sin(t - .5));
  g.lineTo(...b);
  g.lineTo(b[0] - head * Math.cos(t + .5), b[1] - head * Math.sin(t + .5));
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
    t: state.mode === 'result' ? d.mission.x.length - 1 : state.mode === 'mission' ? state.step :
      state.period * H
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
      arrow(g, xy(v), xy([v[0] + m[0] * .27, v[1] + m[1] * .27]), $('truth').checked ?
        '#b27738aa' :
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
    const cw = d.config.whole_field ? scale * (b[1][0] - b[0][0]) / d.config.grid_nx : 12;
    const ch = d.config.whole_field ? scale * (b[1][1] - b[0][1]) / d.config.grid_ny : 12;
    g.fillStyle =
      `rgb(${Math.round(228-202*v)},${Math.round(235-96*v)},${Math.round(233-97*v)})`;
    g.fillRect(a[0] - cw / 2, a[1] - ch / 2, cw, ch);
    if (!d.config.whole_field) {
      g.strokeStyle = '#ffffffee';
      g.strokeRect(a[0] - cw / 2, a[1] - ch / 2, cw, ch);
    }
  });
  if (d.config.whole_field) e.obstacles.forEach(o => {
    g.beginPath();
    g.arc(...xy(o), o[2] * scale, 0, Math.PI * 2);
    g.fillStyle = '#d9dfe3';
    g.fill();
    g.strokeStyle = '#a6b3bd';
    g.stroke();
  });
  if ($('compare').checked && state.comparison) {
    const c = state.comparison;
    const otherColor = c.metrics.method === 'current_belief' ? colors.amber : colors.teal;
    path(g, c.mission.x.slice(0, t + 1), xy, otherColor, 1.8, [5, 4]);
    if (state.mode === 'mission') {
      const a = xy(c.mission.x[t]);
      g.strokeStyle = otherColor;
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
      path(g, v, xy, p.costs[l][i] === null ? `rgba(187,81,76,${alpha})` :
        `rgba(71,115,135,${alpha})`,
        state.hover === i ? 2.5 : .8 + 1.3 * weights[i] / max);
    });
    path(g, p.mean_paths[l], xy, colors.violet, 1.8, [5, 4]);
    path(g, l === d.config.diffusion_steps - 1 ? p.selected_path : p.incumbent_paths[l], xy, colors
      .teal,
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
      path(g, paths[state.hover], xy, p.costs[l][state.hover] === null ? colors.red : colors.ink,
        2.6);
    }
  } else {
    const ownColor = d.metrics.method === 'current_belief' ? colors.amber : colors.teal;
    const period = Math.min(Math.floor(t / H), d.planning.length - 1);
    path(g, d.mission.nominal_paths[period], xy, '#8599a7', 1.1, [3, 4]);
    path(g, d.mission.x.slice(0, t + 1), xy, ownColor, 2.8);
    for (let i = H; d.config.measurements && i <= t; i += H) {
      const a = xy(d.mission.x[i]);
      g.fillStyle = '#fff';
      g.strokeStyle = ownColor;
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
      g.strokeStyle = colors.amber;
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
  for (const [name, key, dy] of(d.config.whole_field ? [] : [
      ['ROI A', 'roi_a', -.35],
      ['ROI B', 'roi_b', -.34]
    ])) {
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
  const xy = ([x, y]) => [box.left + x / Math.max(xmax, .001) * (box.right - box.left), box.bottom -
    y / Math
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
      ]) path(a.g, p.trace.map((r, i) => [i, r[col]]).filter(v => Number.isFinite(v[1])), a.xy,
      color, 1.7,
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
    } = surface('weights-chart'), weights = p.weights[state.level], maxw = Math.max(...weights,
      .01), bw = (
      w - 22) / weights.length;
    weights.forEach((v, i) => {
      g.fillStyle = state.hover === i ? colors.ink : p.costs[state.level][i] === null ? colors
        .red :
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
    if (state.comparison && $('compare').checked) path(a.g, state.comparison.mission.q.map((q,
        i) => [i * d
        .config.dt, deficit(q, state.comparison.environment.targets)
      ]), a.xy, state.comparison.metrics.method === 'current_belief' ? colors.amber : colors
      .teal, 1.4, [4, 3]);
    path(a.g, ts, a.xy, d.metrics.method === 'current_belief' ? colors.amber : colors.teal, 1.8);
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
        b.g.fillStyle = colors.amber;
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

function drawResultMap(id, d, color, seconds = null) {
  const {
    g,
    w,
    h
  } = surface(id);
  if (!d) {
    label(g, 'No matched recording available', w / 2, h / 2, '#6e7e88', 12, 'center');
    return;
  }
  const e = d.environment,
    b = e.bounds;
  const moving = seconds !== null;
  const frame = playbackFrame(d, moving ? seconds : summary(d).duration);
  const scale = Math.min((w - 44) / (b[1][0] - b[0][0]), (h - 44) / (b[1][1] - b[0][1]));
  const left = (w - scale * (b[1][0] - b[0][0])) / 2;
  const top = (h - scale * (b[1][1] - b[0][1])) / 2;
  const xy = a => [left + (a[0] - b[0][0]) * scale, top + (b[1][1] - a[1]) * scale];
  g.fillStyle = '#fafcfc';
  g.fillRect(0, 0, w, h);
  e.obstacles.forEach(o => {
    g.beginPath();
    g.arc(...xy(o), o[2] * scale, 0, Math.PI * 2);
    g.fillStyle = '#d9dfe3';
    g.fill();
    g.strokeStyle = '#abb7bf';
    g.lineWidth = 1;
    g.stroke();
  });
  const layer = $('field-layer').value;
  const passes = layer === 'visits' ? persistence(d).passHistory[frame.index] : null;
  e.cells.forEach((c, i) => {
    const a = xy(c),
      value = layer === 'clarity' ? frame.q[i] : layer === 'target' ? e.targets[i] : layer === 'process' ? (e.process_noise?.[i] ?? d.config.clarity_decay) : layer === 'visits' ? passes[i] : Math.max(e
        .targets[i] - frame.q[i], 0);
    const v = clamp(value / (layer === 'deficit' ? .8 : layer === 'process' ? .1 : layer === 'visits' ? 4 : 1), 0, 1);
    const cw = d.config.whole_field ? scale * (b[1][0] - b[0][0]) / d.config.grid_nx : 7;
    const ch = d.config.whole_field ? scale * (b[1][1] - b[0][1]) / d.config.grid_ny : 7;
    g.fillStyle = heatColor(v);
    g.fillRect(a[0] - cw / 2, a[1] - ch / 2, cw + .4, ch + .4);
  });
  e.obstacles.forEach(o => {
    g.beginPath();
    g.arc(...xy(o), o[2] * scale, 0, Math.PI * 2);
    g.fillStyle = '#d9dfe3';
    g.fill();
    g.strokeStyle = '#81909a';
    g.stroke();
  });
  const visiblePath = d.mission.x.slice(0, frame.index + 1);
  visiblePath.push(frame.x);
  path(g, visiblePath, xy, color, 2.5);
  // Observation boundaries are actual executed states, not forecast waypoints.
  for (let i = d.config.stage_steps; d.config.measurements && i <= frame.index; i += d
    .config.stage_steps) {
    const a = xy(d.mission.x[i]);
    g.beginPath();
    g.arc(...a, 3, 0, 7);
    g.fillStyle = '#fff';
    g.fill();
    g.strokeStyle = color;
    g.lineWidth = 1.5;
    g.stroke();
  }
  const start = xy(d.mission.x[0]),
    end = xy(frame.x);
  g.beginPath();
  g.arc(...start, 4, 0, 7);
  g.fillStyle = '#fff';
  g.fill();
  g.strokeStyle = color;
  g.stroke();
  if (moving) {
    g.beginPath();
    g.arc(...end, 11, 0, Math.PI * 2);
    g.fillStyle = `${color}22`;
    g.fill();
    g.beginPath();
    g.arc(...end, 5, 0, Math.PI * 2);
    g.fillStyle = color;
    g.fill();
    g.strokeStyle = '#fff';
    g.lineWidth = 1.5;
    g.stroke();
    arrow(g, end, xy([frame.x[0] + frame.x[2] * .28, frame.x[1] + frame.x[3] * .28]), color, 1.7);
  } else {
    g.fillStyle = color;
    g.fillRect(end[0] - 4, end[1] - 4, 8, 8);
    label(g, 'End', end[0], end[1] + 17, color, 10, 'center');
    label(g, 'Start', start[0], start[1] - 10, '#40545f', 10, 'center');
  }
  for (const [name, roi] of(d.config.whole_field ? [] : [
      ['A', e.roi_a],
      ['B', e.roi_b]
    ])) {
    const a = xy(roi);
    label(g, `ROI ${name}`, a[0], a[1] + (name === 'B' ? 20 : -20), '#40545f', 10, 'center');
  }
  label(g, moving ? `t = ${Math.min(seconds, summary(d).duration).toFixed(2)} s` : d.config
    .measurements ? '○ belief update     ■ final position' : '■ final position', w /
    2, h - 5, '#6e7e88', 9, 'center');
  if (state.cell !== null && e.cells[state.cell]) {
    const selected = xy(e.cells[state.cell]);
    g.strokeStyle = '#fff';
    g.lineWidth = 4;
    g.strokeRect(selected[0] - 5, selected[1] - 5, 10, 10);
    g.strokeStyle = '#223f50';
    g.lineWidth = 1.4;
    g.strokeRect(selected[0] - 5, selected[1] - 5, 10, 10);
  }
}

function renderComparison() {
  const [a, b] = comparisonPair(state.data, state.comparison);
  const duration = summary(a).duration;
  const time = Math.min(state.comparisonClock, duration);
  $('comparison-play').textContent = state.playing ? 'Ⅱ Pause' : '▶ Play';
  $('comparison-timeline').max = duration;
  $('comparison-timeline').value = time;
  $('comparison-time').textContent = `${time.toFixed(2)} / ${duration.toFixed(1)} s`;
  $('result-name-a').textContent = methodLabel(a);
  $('result-name-b').textContent = b ? methodLabel(b) : 'No matched comparator';
  $('result-name-a').style.color = a.metrics.method === 'current_belief' ? colors.amber : colors
    .teal;
  $('result-name-b').style.color = colors.amber;
  $('inspect-b').disabled = !b;
  for (const [d, side, color] of [
      [a, 'a', a.metrics.method === 'current_belief' ? colors.amber : colors.teal],
      [b, 'b', colors.amber]
    ]) {
    if (d) {
      const f = playbackFrame(d, time),
        cost = summary(d).costHistory;
      const currentCost = cost[f.index] + f.mix * (cost[f.next] - cost[f.index]);
      $(`result-cost-${side}`).textContent = `Cost so far ${currentCost.toFixed(3)}`;
      $(`result-note-${side}`).textContent =
        `Mean deficit ${deficit(f.q, d.environment.targets).toFixed(3)} · ${f.q.length} monitored cells`;
      $(`result-map-${side}`).setAttribute('aria-label',
        `Animated executed trajectory: ${methodLabel(d)}`);
    } else {
      $(`result-cost-${side}`).textContent = '—';
      $(`result-note-${side}`).textContent = 'Run a matched comparator to animate both.';
    }
    drawResultMap(`result-map-${side}`, d, color, time);
  }
  let message = b ?
    'Both robots execute on the same clock. Watch their routes and accumulated cost.' :
    'Unpaired recording: watch the robot and its accumulated task cost.';
  message = 'Every cell contributes to the task cost. Compare the maps and deficit curves below.';
  if (time >= duration) message = $('comparison-loop').checked && state.playing ?
    'Replay complete. Restarting automatically…' :
    'Replay complete. Open Final result for the full comparison.';
  if ($('comparison-event').textContent !== message) $('comparison-event').textContent = message;
  drawFieldDiagnostics(a, b, time);
}

function renderResults() {
  const [a, b] = comparisonPair(state.data, state.comparison);
  const sa = missionSummary(a),
    sb = b ? missionSummary(b) : null;
  $('result-scope').textContent = b ? 'COMPLETED MISSIONS · SAME WIND AND START' :
    'COMPLETED MISSION · UNPAIRED RECORDING';
  const colorA = a.metrics.method === 'current_belief' ? colors.amber : colors.teal;
  $('result-name-a').textContent = methodLabel(a);
  $('result-name-b').textContent = b ? methodLabel(b) : 'Comparison unavailable';
  $('result-column-a').textContent = a.metrics.method === 'current_belief' ?
    'No anticipated learning' : a.metrics.method === 'proposed' ? 'Proposed' : a.metrics.method
    .toUpperCase();
  $('result-column-b').textContent = b ? 'No anticipated learning' : '—';
  $('result-cost-a').textContent = `Cost ${sa.cost.toFixed(3)}`;
  $('result-cost-b').textContent = sb ? `Cost ${sb.cost.toFixed(3)}` : '—';
  $('result-name-a').style.color = colorA;
  $('result-name-b').style.color = colors.amber;
  $('result-note-a').textContent = a.config.measurements ?
    'Accounts for future measurements while choosing current motion.' :
    'Measurements are disabled in this control run.';
  if (a.metrics.method === 'current_belief') $('result-note-a').textContent =
    'Updates during execution; future planning uses the belief available now.';
  $('result-note-b').textContent = b ?
    'Updates during execution; future planning uses the belief available now.' :
    'Run the other method with the same settings and seed to compare.';
  $('inspect-b').disabled = !b;
  $('replay-pair').textContent = b ? 'Replay both missions from the start →' :
    'Replay this mission from the start →';
  $('result-map-a').setAttribute('aria-label', `Completed trajectory: ${methodLabel(a)}`);
  $('result-map-b').setAttribute('aria-label', b ? `Completed trajectory: ${methodLabel(b)}` :
    'No matched trajectory');
  $('result-verdict').style.color = colors.ink;
  if (sb) {
    const benefit = sb.cost - sa.cost;
    $('result-verdict').textContent =
      `${(100 * Math.abs(benefit) / Math.max(sb.cost, 1e-12)).toFixed(1)}% ${benefit >= 0 ? 'lower' : 'higher'} monitoring cost when anticipating learning`;
    $('result-verdict').style.color = benefit >= 0 ? colors.teal : colors.red;
    const extra = sa.first - sb.first,
      recovered = sb.later - sa.later;
    $('result-context').textContent =
      `First ${(a.config.stage_steps*a.config.dt).toFixed(1)} s: ${Math.abs(extra).toFixed(3)} ${extra >= 0 ? 'extra' : 'less'} task cost. Remaining mission: ${Math.abs(recovered).toFixed(3)} ${recovered >= 0 ? 'less' : 'extra'} cost. These are actual executed outcomes for seed ${a.metrics.seed}, not diffusion forecasts.`;
  } else {
    $('result-verdict').textContent = `Completed mission · cost ${sa.cost.toFixed(3)}`;
    $('result-context').textContent =
      `Seed ${a.metrics.seed} · ${sa.duration.toFixed(1)} seconds · ${a.planning.length} replans. A paired benefit cannot be calculated from one run.`;
  }
  const rows = [
    ['Total monitoring cost ↓', s => s.cost.toFixed(3)],
    [`First ${(a.config.stage_steps*a.config.dt).toFixed(1)} s cost`, s => s.first.toFixed(3)],
    ['Remaining mission cost', s => s.later.toFixed(3)],
    ['Final clarity deficit ↓', s => s.finalDeficit.toFixed(3)],
    ['Minimum obstacle clearance', s => `${s.clearance.toFixed(3)} m`],
    ['Peak allowed barrier slack', s => s.peakSlack.toFixed(4)],
    ['Time using barrier slack', s =>
      `${s.relaxedSeconds.toFixed(2)} s (${s.relaxedSteps} steps)`
    ],
    ['Integrated barrier slack', s => s.integratedSlack.toFixed(4)],
    ['Physical collision', s => s.collision ? 'Observed' : 'None observed']
  ];
  if (a.config.whole_field) {
    sa.persistence = persistence(a);
    if (sb) sb.persistence = persistence(b);
    rows.splice(4, 0,
      ['Cells with repeated sensing passes', s => `${s.persistence.revisitedCells} / ${a.environment.cells.length}`],
      ['Cells recovered after losing target clarity', s => `${s.persistence.recoveredCells} / ${a.environment.cells.length}`],
      ['Median interval between sensing passes', s => s.persistence.medianRevisitSeconds === null ? 'No repeated passes' : `${s.persistence.medianRevisitSeconds.toFixed(1)} s`]
    );
  }
  $('result-rows').replaceChildren(...rows.map(([title, format]) => {
    const row = document.createElement('tr');
    for (const [i, value] of [title, format(sa), sb ? format(sb) : '—'].entries()) {
      const cell = document.createElement(i ? 'td' : 'th');
      if (!i) cell.scope = 'row';
      cell.textContent = value;
      row.append(cell);
    }
    return row;
  }));
  drawResultMap('result-map-a', a, colorA);
  drawResultMap('result-map-b', b, colors.amber);
  drawFieldDiagnostics(a, b, sa.duration);
}

function update() {
  if (!state.data) return;
  $('quick-play').disabled = false;
  $('quick-diffusion').disabled = false;
  $('quick-play').textContent = state.comparison ? '▶ Play comparison' : '▶ Play mission';
  const {
    d,
    H,
    p,
    t
  } = active(), isDiff = state.mode === 'diffusion', isResult = state.mode === 'result', isCompare =
    state.mode === 'comparison';
  const isPairView = isResult || isCompare;
  $('result-view').classList.toggle('hidden', !isPairView);
  $('result-view').classList.toggle('is-comparison', isCompare);
  $('result-view').setAttribute('aria-label', isCompare ? 'Animated planner comparison' :
    'Final executed comparison');
  $('comparison-transport').classList.toggle('hidden', !isCompare);
  for (const id of ['result-heading', 'result-table-wrap', 'result-footnote', 'result-actions']) $(
    id).classList.toggle('hidden', isCompare);
  $('method-explainer').classList.toggle('hidden', isCompare);
  document.querySelector('.quick-actions').classList.toggle('hidden', isCompare);
  $('playback-view').classList.toggle('hidden', isPairView);
  $('mode-note').classList.toggle('hidden', isPairView);
  $('playback-layers').classList.toggle('hidden', isPairView);
  $('metrics-panel').classList.toggle('hidden', isPairView);
  $('charts-panel').classList.toggle('hidden', isPairView);
  $('comparison-tab').classList.toggle('active', isCompare);
  $('comparison-tab').setAttribute('aria-selected', String(isCompare));
  $('result-tab').classList.toggle('active', isResult);
  $('result-tab').setAttribute('aria-selected', String(isResult));
  $('diffusion-tab').classList.toggle('active', isDiff);
  $('mission-tab').classList.toggle('active', state.mode === 'mission');
  $('diffusion-tab').setAttribute('aria-selected', String(isDiff));
  $('mission-tab').setAttribute('aria-selected', String(state.mode === 'mission'));
  $('last-level').classList.toggle('hidden', !isDiff);
  $('learning-explainer-title').textContent = d.config.measurements ?
    'Both planning approaches learn during execution.' :
    'Measurements are disabled in this experiment.';
  $('learning-explainer-text').textContent = d.config.measurements ?
    'The proposed planner also anticipates how future measurements will improve later decisions. The comparator plans its future using the belief available now.' :
    'The wind belief is not updated. This is a separate no-measurement control; it is different from the comparator that learns during execution but does not anticipate learning while planning.';
  $('mode-note').textContent = isDiff ?
    `Denoising: ${methodLabel(d)}. Each candidate predicts ${((H+d.config.future_steps)*d.config.dt).toFixed(1)} s; only the first ${(H*d.config.dt).toFixed(1)} s is executed before updating and replanning. “Final result” shows the completed actual mission.` :
    `Mission playback: ${methodLabel(d)} is ${d.metrics.method === 'current_belief' ? 'amber' : 'teal'}. ${state.comparison ? 'The other planner can be overlaid at the same execution time.' : 'This recording has no paired comparator.'}`;
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
  $('scene-label').textContent = d.config.whole_field ?
    `${d.config.scenario.replaceAll('_', ' ')} · ${d.environment.cells.length} cells` :
    'Legacy ROI experiment · feature GP';
  $('variant-tag').textContent = d.metrics.method === 'current_belief' ? 'No anticipated learning' :
    d.config.future_updates && d.config.future_steps > d.config.stage_steps ? 'Repeated belief updates' : 'Two-stage';
  $('provenance').textContent =
    `Seed ${d.metrics.seed} · ${d.metrics.method} · JAX ${d.metrics.jax_version} · ${d.metrics.mean_plan_seconds.toFixed(2)} s / plan · ${d.metrics.objective_evaluations_per_plan} objective evaluations`;
  if (isCompare) {
    renderComparison();
    return;
  }
  if (isResult) {
    renderResults();
    return;
  }
  if (isDiff) {
    const l = state.level,
      r = p.trace[l],
      noise = Math.sqrt((1 - p.alphabar[l]) / p.alphabar[l]);
    $('map-title').textContent = d.metrics.method === 'cem' ? 'Cross-entropy optimization' : d
      .metrics
      .method === 'mppi' ? 'MPPI optimization' : 'Denoising the policy';
    $('map-subtitle').textContent =
      `Plan at t = ${(state.period*H*d.config.dt).toFixed(1)} s · planning world 1 of ${d.config.worlds}`;
    setMetric(1, l === d.config.diffusion_steps - 1 ? 'Final selected cost' : 'Best predicted cost',
      metricText(l === d.config.diffusion_steps - 1 ? p.selected_cost : r[0]), l === d.config
      .diffusion_steps - 1 ? 'Includes evaluation of the final denoised policy' :
      'Task deficit over the planning window');
    setMetric(2, d.metrics.method === 'cem' || d.metrics.method === 'mppi' ? 'Iteration' :
      'Proposal noise σ',
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
    $('formula').textContent = d.metrics.method === 'cem' ? 'elite policies → refit distribution' :
      d.metrics
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
      `Belief updated ${d.config.measurements ? Math.floor(t/H) : 0} times · ${fail?'allowed barrier slack active':'zero barrier slack'}`;
    setMetric(1, 'Accumulated task cost', state.cost[t].toFixed(3),
      'Actual executed clarity deficit');
    setMetric(2, 'Current clarity deficit', deficit(d.mission.q[t], d.environment.targets).toFixed(
        3),
      'Lower is better');
    setMetric(3, 'Obstacle clearance', idx < 0 ? '—' : `${d.mission.clearance[idx].toFixed(3)} m`,
      'Measured within the current control step');
    setMetric(4, 'Allowed barrier slack', idx < 0 ? '0.0000' : d.mission.slack[idx].toFixed(4),
      idx < 0 ? 'Initial state' :
      `Control correction ${d.mission.correction[idx].toFixed(3)}`);
    $('metric-4').style.color = fail ? colors.amber : colors.ink;
    $('chart-title-1').textContent = 'Monitoring deficit';
    $('chart-note-1').textContent = 'Actual mission · lower is better';
    $('chart-title-2').textContent = 'Safety interventions';
    $('chart-note-2').textContent = 'Control correction norm';
    $('candidate-detail').textContent =
      `${d.mission.feasible.filter(x=>!x).length} steps used allowed barrier relaxation · amber dots mark slack use`;
    $('explanation-title').textContent = 'Information has to earn its cost';
    $('explanation-text').textContent =
      d.config.measurements ?
      'The robot gathers wind measurements along its executed path. The posterior changes only at period boundaries. That belief changes subsequent safety constraints and monitoring decisions.' :
      'Measurements are disabled in this control run. The robot replans as its physical state and monitoring clarity change, while its wind belief stays fixed.';
    $('formula').textContent = d.config.measurements ?
      'execute → observe → update belief → replan' : 'execute → replan with unchanged wind belief';
    $('forecast-note').textContent =
      'Teal anticipates learning; amber plans with its current belief. Both use real observations during execution. Gray dashes show the selected planner’s nominal prediction.';
    $('legend').innerHTML =
      `<span><i class="line ${d.metrics.method === 'current_belief' ? 'amber' : 'teal'}"></i>${methodLabel(d)}</span>` +
      (state.comparison && $('compare').checked ?
        `<span><i class="line ${state.comparison.metrics.method === 'current_belief' ? 'amber' : 'teal'}"></i>${methodLabel(state.comparison)}</span>` :
        '') +
      '<span><i class="line gray"></i>Nominal prediction</span><span><i class="square"></i>Selected planner’s clarity</span>';
  }
  if (isDiff) $('metric-4').style.color = colors.ink;
  $('paired-denoising').classList.toggle('hidden', !isDiff);
  if (isDiff) drawPairedDenoising();
  drawMap();
  drawCharts();
}

function setMode(mode) {
  if (state.mode === 'result' && mode !== 'result') state.period = 0;
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
  $('recording').disabled = true;
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
    diagnosticFrame = "";
    state.cell = null;
    state.mode = 'comparison';
    state.comparisonClock = 0;
    $('comparison-speed').value = (data.mission.x.length-1)*data.config.dt >= 60 ? '4' : '0.5';
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
      const [learning, baseline] = comparisonPair(data, state.comparison);
      const improvement = (baseline.metrics.integrated_deficit - learning.metrics
        .integrated_deficit) / baseline.metrics.integrated_deficit * 100;
      $('comparison-summary').classList.toggle('regression', improvement < 0);
      summary =
        `<strong>${Math.abs(improvement).toFixed(1)}% ${improvement>=0?'lower':'higher'} mission cost</strong> when anticipating learning.<br>Same physical wind · seed ${data.metrics.seed}`;
    }
    $('comparison-summary').innerHTML = summary;
    $('run-diagnostics').textContent =
      `Selected run: ${data.metrics.collision?'collision observed':'no collisions'} · ${data.mission.feasible.filter(v=>!v).length} steps using allowed slack · ${data.metrics.optimizer_failure_count} optimizer fallbacks`;
    $('run-diagnostics').style.color = data.metrics.collision || data.metrics
      .optimizer_failure_count ? colors.red : '';
    $('compare').disabled = !state.comparison;
    $('record-type').textContent = data.config.whole_field ? 'Full-field POAM' :
      'Legacy ROI model';
    document.querySelector('.intro > div > p:last-child').textContent = data.config.whole_field ?
      'Full-field clarity deficit · online POAM wind estimates · actual diffusion rollouts.' :
      'Archived 13-cell experiment · finite-feature GP · select a full-field POAM recording for the new model.';
    $('recording').value = id;
    $('method').value = data.metrics.method;
    if (data.config.whole_field) $('scenario').value = data.config.scenario;
    $('seed').value = data.metrics.seed;
    $('variant').value = data.config.future_updates ? 'multistage' : 'two_stage';
    $('wind-std').value = data.config.wind_std;
    $('observation-std').value = data.config.observation_std;
    $('measurements').checked = data.config.measurements;
    $('levels').value = String(data.config.diffusion_steps);
    $('proposals').value = String(data.config.proposals);
    status(`${record.live?'Local experiment':'Recorded experiment'} ready`);
    state.playing = true;
    update();
    window.scrollTo(0, 0);
  } catch (e) {
    status(e.message, 'error');
  } finally {
    state.loading = false;
    $('recording').disabled = false;
  }
}
async function run() {
  const data = {
    scenario: $('scenario').value,
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
$('result-tab').onclick = () => state.data && setMode('result');
$('comparison-tab').onclick = () => state.data && startComparison();
$('diffusion-tab').onclick = () => {
  if (!state.data) return;
  setMode('diffusion');
  $('sequence').checked = false;
  state.level = 0;
  state.playing = true;
  update();
};
$('mission-tab').onclick = () => state.data && setMode('mission');
$('last-level').onclick = () => {
  state.playing = false;
  state.level = state.data.config.diffusion_steps - 1;
  update();
};
async function inspectPlanner(index) {
  const desired = comparisonPair(state.data, state.comparison)[index];
  if (!desired) return;
  if (desired !== state.data) {
    const record = pairedRecord(state.records, state.record);
    if (record) await replay(record.id);
  }
  setMode('diffusion');
}
$('inspect-a').onclick = () => inspectPlanner(0);
$('inspect-b').onclick = () => inspectPlanner(1);
$('replay-pair').onclick = () => {
  startComparison();
};
$('quick-play').onclick = () => {
  startComparison();
  $('comparison-tab').scrollIntoView({
    behavior: 'smooth',
    block: 'start'
  });
};

function startComparison() {
  if (!state.data) return;
  state.mode = 'comparison';
  state.comparisonClock = 0;
  state.playing = true;
  state.accumulator = 0;
  update();
}
$('comparison-play').onclick = () => {
  if (!state.data) return;
  if (state.comparisonClock >= summary(state.data).duration) state.comparisonClock = 0;
  state.playing = !state.playing;
  renderComparison();
};
$('comparison-restart').onclick = startComparison;
$('comparison-timeline').oninput = () => {
  if (!state.data) return;
  state.comparisonClock = Number($('comparison-timeline').value);
  state.playing = false;
  renderComparison();
};
$('comparison-moment').onclick = () => {
  if (!state.data) return;
  const [a, b] = comparisonPair(state.data, state.comparison);
  let largest = 0,
    index = Math.floor(a.mission.q.length / 2);
  if (b) summary(a).costHistory.forEach((cost, i) => {
    const gap = Math.abs(summary(b).costHistory[i] - cost);
    if (gap > largest) {
      largest = gap;
      index = i;
    }
  });
  state.comparisonClock = index * a.config.dt;
  state.playing = false;
  renderComparison();
};
$('quick-diffusion').onclick = async () => {
  await inspectPlanner(0);
  $('sequence').checked = false;
  state.level = 0;
  state.playing = true;
  update();
  $('diffusion-tab').scrollIntoView({
    behavior: 'smooth',
    block: 'start'
  });
};
$('play').onclick = () => {
  if (state.data) {
    if (state.mode === 'result') setMode('mission');
    if (!state.playing && state.mode === 'diffusion' && state.level === state.data.config
      .diffusion_steps - 1) state.level = 0;
    if (!state.playing && state.mode === 'mission' && state.step === state.data.mission.x.length -
      1) {
      state.step = 0;
      state.period = 0;
    }
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
    state.period = Math.min(Math.floor(state.step / state.data.config.stage_steps), state.data
      .planning
      .length - 1);
  }
  update();
};
$('recording').onchange = () => replay($('recording').value);
$('run').onclick = run;
for (const id of ['compare', 'uncertainty', 'wind', 'observations', 'truth']) $(id).onchange =
  update;
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
  if (e.code === 'Space' && !['INPUT', 'SELECT', 'BUTTON', 'TEXTAREA'].includes(e.target
      .tagName)) {
    e.preventDefault();
    $(state.mode === 'comparison' ? 'comparison-play' : 'play').click();
  }
});

function animate(now) {
  const elapsed = Math.min((now - state.last) / 1000, .1);
  state.last = now;
  if (state.playing && state.data && state.mode === 'comparison') {
    const next = advanceComparison(state.comparisonClock, elapsed, Number($('comparison-speed')
      .value), summary(state.data).duration, $('comparison-loop').checked);
    state.comparisonClock = next.clock;
    state.playing = next.playing;
    renderComparison();
    requestAnimationFrame(animate);
    return;
  }
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
      } else {
        state.playing = false;
        state.mode = 'result';
      }
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

function heatColor(v) {
  v = clamp(v, 0, 1);
  return `rgb(${Math.round(246-202*v)},${Math.round(246-159*v)},${Math.round(240-104*v)})`;
}

function pairedCurve(id, a, b, getPoints, time, fixedMax = null) {
  let pa = getPoints(a),
    pb = b ? getPoints(b) : [];
  if (id.startsWith('wind-')) {
    const steps = points => points.flatMap((p, i) => i ? [
      [p[0], points[i - 1][1]], p
    ] : [p]);
    pa = steps(pa);
    pb = steps(pb);
  }
  const max = fixedMax ?? Math.max(.01, ...pa.map(p => p[1]), ...pb.map(p => p[1])) * 1.1;
  const axes = chartAxes(id, summary(a).duration, max, 'seconds');
  if (pb.length) path(axes.g, pb, axes.xy, colors.amber, 2);
  path(axes.g, pa, axes.xy, colors.teal, 2);
  path(axes.g, [
    [time, 0],
    [time, max]
  ], axes.xy, '#536572', 1, [3, 3]);
}

function drawWindMap(id, d, time, truth = false) {
  const {
    g,
    w,
    h
  } = surface(id);
  if (!d) return;
  const bounds = d.environment.bounds,
    scale = Math.min((w - 10) / 4.8, (h - 24) / 3.7);
  const xy = p => [(w - 4.8 * scale) / 2 + (p[0] - bounds[0][0]) * scale, 8 + (bounds[1][1] - p[
    1]) * scale];
  const bi = beliefIndex(Math.floor(time / d.config.dt + 1e-6), d.config.stage_steps, d.beliefs
    .length);
  const belief = d.beliefs[bi],
    error = $('wind-layer').value === 'error';
  const nx = d.config.whole_field ? d.config.grid_nx : 25,
    ny = d.config.whole_field ? d.config.grid_ny : 19;
  const range = truth ? 1.2 : error ? 1.2 : Math.max(.6, d.config.wind_std);
  const vectors = truth ? d.physical_wind : belief.mean;
  d.wind_grid.forEach((p, i) => {
    const a = xy(p);
    const val = truth ? Math.hypot(...vectors[i]) : error ? Math.hypot(...belief.mean[i].map((v,
      k) => v - d.physical_wind[i][k])) : belief.std[i];
    g.fillStyle = heatColor(val / range);
    g.fillRect(a[0] - scale * 4.8 / nx / 2, a[1] - scale * 3.7 / ny / 2, scale * 4.8 / nx + .3,
      scale * 3.7 / ny + .3);
    if (i % nx % 3 === 1 && Math.floor(i / nx) % 3 === 1) arrow(g, a, xy([p[0] + vectors[i][0] *
      .3, p[1] + vectors[i][1] * .3
    ]), '#1c323e', .8);
  });
  if (!truth && d.config.measurements) d.mission.x.slice(1, Math.floor(time / d.config.dt) + 1).forEach(p => {
    const a = xy(p);
    g.beginPath();
    g.arc(...a, 1.3, 0, 7);
    g.fillStyle = '#f1a441';
    g.fill();
  });
  label(g, `Light 0 → dark ${range.toFixed(1)}  m/s² · arrows × 0.3`, w / 2, h - 3, '#637681', 8,
    'center');
  return {
    belief,
    bi
  };
}

let diagnosticFrame = '';

function drawFieldDiagnostics(a, b, time) {
  const key =
    `${a.metrics.scenario}/${a.metrics.method}/${a.metrics.seed}/${time.toFixed(1)}/${$('wind-layer').value}/${window.innerWidth}`;
  if (diagnosticFrame === key) return;
  diagnosticFrame = key;
  $('field-key').textContent = $('field-layer').value === 'deficit' ?
    'Light: 0 · Dark: 0.8 deficit · lower is better' : $('field-layer').value === 'process' ? 'Light: 0 · Dark: 0.1 process noise Qₚ' : $('field-layer').value === 'visits' ? 'Light: 0 · Dark: ≥4 distinct sensing passes' : 'Light: 0 · Dark: 1 clarity';
  pairedCurve('field-deficit-chart', a, b, d => d.mission.q.map((q, i) => [i * d.config.dt, deficit(
    q, d.environment.targets)]), time);
  pairedCurve('field-cost-chart', a, b, d => summary(d).costHistory.map((v, i) => [i * d.config.dt,
    v
  ]), time);
  drawCellDiagnostic(a, b, time);
  $('wind-map-a').setAttribute('aria-label',
    `Proposed ${a.config.wind_backend === 'poam' ? 'POAM' : 'legacy GP'} wind estimate and uncertainty`
    );
  $('wind-map-b').setAttribute('aria-label',
    `Comparator ${a.config.wind_backend === 'poam' ? 'POAM' : 'legacy GP'} wind estimate and uncertainty`
    );
  $('wind-model-label').textContent = a.config.wind_backend === 'poam' ? 'full POAM prediction' :
    'legacy feature GP';
  for (const [d, side] of [
      [a, 'a'],
      [b, 'b']
    ]) {
    const out = drawWindMap(`wind-map-${side}`, d, time);
    $(`wind-note-${side}`).textContent = out ?
      `Update ${out.bi} · RMSE ${windError(d,out.belief).toFixed(3)} · mean σ ${(out.belief.std.reduce((s,v)=>s+v,0)/out.belief.std.length).toFixed(3)}` :
      'No matched run';
  }
  drawWindMap('wind-map-truth', a, time, true);
  pairedCurve('wind-error-chart', a, b, d => d.beliefs.map((v, i) => [i * d.config.stage_steps * d
    .config.dt, windError(d, v)
  ]), time);
  pairedCurve('wind-std-chart', a, b, d => d.beliefs.map((v, i) => [i * d.config.stage_steps * d
    .config.dt, v.std.reduce((s, x) => s + x, 0) / v.std.length
  ]), time);
}

function windError(d, belief) {
  return Math.sqrt(belief.mean.reduce((s, v, i) => s + v.reduce((a, x, k) => a + (x - d
    .physical_wind[i][k]) ** 2, 0), 0) / (2 * belief.mean.length));
}

function drawPairedDenoising() {
  const pair = comparisonPair(state.data, state.comparison);
  for (const [d, side, color] of [
      [pair[0], 'a', colors.teal],
      [pair[1], 'b', colors.amber]
    ]) {
    const {
      g,
      w,
      h
    } = surface(`denoising-${side}`);
    if (!d) continue;
    const p = d.planning[Math.min(state.period, d.planning.length - 1)],
      l = Math.min(state.level, p.candidate_paths.length - 1);
    const bounds = d.environment.bounds,
      scale = Math.min((w - 24) / 4.8, (h - 24) / 3.7);
    const xy = v => [(w - 4.8 * scale) / 2 + (v[0] - bounds[0][0]) * scale, 12 + (bounds[1][1] - v[
      1]) * scale];
    if (d.config.whole_field) {
      const q = d.mission.q[p.period * d.config.stage_steps];
      const cw = scale * 4.8 / d.config.grid_nx, ch = scale * 3.7 / d.config.grid_ny;
      g.globalAlpha = .4;
      d.environment.cells.forEach((cell, i) => {
        const a = xy(cell);
        g.fillStyle = heatColor(Math.max(d.environment.targets[i] - q[i], 0) / .8);
        g.fillRect(a[0]-cw/2, a[1]-ch/2, cw+.4, ch+.4);
      });
      g.globalAlpha = 1;
    }
    d.environment.obstacles.forEach(o => {
      g.beginPath();
      g.arc(...xy(o), o[2] * scale, 0, 7);
      g.fillStyle = '#d9dfe3';
      g.fill();
    });
    p.candidate_paths[l].forEach((points, i) => path(g, points, xy,
      `rgba(80,105,120,${.12+.55*p.weights[l][i]})`, 1));
    path(g, p.mean_paths[l], xy, colors.violet, 1.5, [3, 3]);
    path(g, l === p.candidate_paths.length - 1 ? p.selected_path : p.incumbent_paths[l], xy, color,
      2.5);
    $(`denoising-note-${side}`).textContent =
      `Level ${l+1}/${p.candidate_paths.length} · best expected deficit cost ${metricText(l === p.candidate_paths.length - 1 ? p.selected_cost : p.trace[l][0])} · bold: best evaluated; purple: denoised mean${d.config.whole_field ? '; background: current clarity deficit' : ''}`;
  }
}
$('field-layer').onchange = () => {
  diagnosticFrame = '';
  update();
};
$('wind-layer').onchange = () => {
  diagnosticFrame = '';
  update();
};


function drawCellDiagnostic(a, b, time) {
  const i = state.cell;
  const axes = chartAxes('cell-clarity-chart', summary(a).duration, 1, 'seconds');
  if (i === null || !a.environment.cells[i]) {
    label(axes.g, 'Click a cell on either field map', axes.w / 2, 65, '#637681', 12, 'center');
    $('cell-description').textContent = 'Click either field map to see clarity rise with sensing and decay between visits.';
    return;
  }
  const p = a.environment.cells[i], target = a.environment.targets[i];
  const process = a.environment.process_noise?.[i] ?? a.config.clarity_decay;
  const q0 = a.mission.q[0][i];
  path(axes.g, a.mission.q.map((_,j)=>[j*a.config.dt,q0/(1+process*q0*j*a.config.dt)]), axes.xy, '#99a4ac', 1.2, [2,3]);
  for (const [d, color] of [[a, colors.teal], [b, colors.amber]]) {
    if (d) path(axes.g, d.mission.q.map((q, j) => [j*d.config.dt, q[i]]), axes.xy, color, 2);
  }
  path(axes.g, [[0, target], [summary(a).duration, target]], axes.xy, '#637681', 1, [4, 4]);
  path(axes.g, [[time, 0], [time, 1]], axes.xy, '#536572', 1, [3, 3]);
  const qa = playbackFrame(a, time).q[i], qb = b ? playbackFrame(b, time).q[i] : null;
  $('cell-description').textContent = `Cell ${i+1} · (${p[0].toFixed(2)}, ${p[1].toFixed(2)}) m · Qₚ ${process.toFixed(3)} · target ${target.toFixed(3)} · clarity now: proposed ${qa.toFixed(3)}${qb === null ? '' : ', comparator '+qb.toFixed(3)}`;
}

for (const id of ['result-map-a', 'result-map-b']) $(id).addEventListener('click', event => {
  if (!state.data) return;
  const [d] = comparisonPair(state.data, state.comparison);
  const rect = $(id).getBoundingClientRect(), bounds = d.environment.bounds;
  const scale = Math.min((rect.width-44)/(bounds[1][0]-bounds[0][0]), (rect.height-44)/(bounds[1][1]-bounds[0][1]));
  const left = (rect.width-scale*(bounds[1][0]-bounds[0][0]))/2;
  const top = (rect.height-scale*(bounds[1][1]-bounds[0][1]))/2;
  const point = [(event.clientX-rect.left-left)/scale+bounds[0][0], bounds[1][1]-(event.clientY-rect.top-top)/scale];
  if (point.some((v,k) => v < bounds[0][k] || v > bounds[1][k])) return;
  let best = Infinity;
  d.environment.cells.forEach((p, i) => {
    const distance = (p[0]-point[0])**2+(p[1]-point[1])**2;
    if (distance < best) { best = distance; state.cell = i; }
  });
  diagnosticFrame = '';
  update();
});
$('show-wind').onclick = () => $('wind-diagnostics').scrollIntoView({behavior:'smooth', block:'start'});

async function persistentProgress() {
  try {
    const response = await fetch('/api/persistent-status');
    if (!response.ok) return;
    const progress = await response.json();
    const jobs = progress.jobs || [];
    $('persistent-progress').classList.toggle('hidden', !jobs.length);
    if (jobs.length) {
      const stale = Date.now()/1000 - progress.updated_at > 120;
      const complete = jobs.filter(j => j.status === 'complete').length;
      $('persistent-progress-summary').textContent = `122 s persistent study · ${complete}/${jobs.length} complete · ${jobs.filter(j => j.status === 'running').length} running`;
      const running = jobs.filter(j => j.status === 'running').map(j =>
        `${j.scenario.replace('persistent_','')} / ${j.method === 'proposed' ? 'proposed' : 'no anticipated learning'}: ${j.completed_periods || 0}/${j.total_periods} periods`);
      const failed = jobs.filter(j => j.status === 'failed').length;
      const paired = jobs.some(j => j.status === 'complete' && jobs.some(k => k.scenario === j.scenario && k.method !== j.method && k.status === 'complete'));
      $('refresh-recordings').disabled = !paired;
      $('refresh-recordings').textContent = paired ? 'Load completed persistent comparison →' : 'Persistent replay available when a pair finishes';
      $('persistent-progress-text').textContent = `${complete}/${jobs.length} missions complete. ${running.join(' · ')}. ${failed ? `${failed} failed; logs retained. ` : ''}${stale && running.length ? 'Progress has not updated recently; execution may be paused. ' : ''}Completed pairs appear in Recorded run. These are offline CPU experiments.`;
    }
    setTimeout(persistentProgress, 10000);
  } catch {
    setTimeout(persistentProgress, 10000);
  }
}
$('refresh-recordings').onclick = async () => {
  await catalog();
  const newest = state.records.find(r => r.id.startsWith('persistent_')) || state.records[0];
  if (newest) await replay(newest.id);
};
persistentProgress();
