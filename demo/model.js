export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
export function deficit(q, targets) {
  return q.reduce((s, v, i) => s + Math.max(0, targets[i] - v), 0) / q.length;
}
export function cumulativeCost(q, targets, dt) {
  const c = [0];
  for (let i = 1; i < q.length; i++) c.push(c.at(-1) + dt * (deficit(q[i - 1], targets) + deficit(q[
      i],
    targets)) / 2);
  return c;
}
export function beliefIndex(step, H, count) {
  return clamp(Math.floor(step / H), 0, count - 1);
}
export function pairedRecord(records, record, comparator = 'current_belief') {
  return records.find(r => r.group === record.group && r.seed === record.seed && r.method === (
    ['current_belief','wind_blind_planner'].includes(record.method) ? 'proposed' : comparator)) || null;
}
export function metricText(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : 'Infeasible';
}

export function methodLabel(data) {
  if (data.metrics.method === 'current_belief') return 'Wind-aware · no anticipated learning';
  if (data.metrics.method === 'wind_blind_planner') return 'Wind-blind plan + safety filter';
  if (!data.config.measurements) return 'Proposed · measurements off';
  if (data.metrics.method === 'cem') return 'CEM · anticipates learning';
  if (data.metrics.method === 'mppi') return 'MPPI · anticipates learning';
  return 'Proposed · anticipates learning';
}

// Keep method identity and colors stable even when the baseline record is selected.
export function comparisonPair(data, other) {
  if (['current_belief','wind_blind_planner'].includes(data.metrics.method) && other) return [other, data];
  return [data, other];
}

export function comparisonClaim(baseline) {
  return baseline?.metrics.method === 'wind_blind_planner' ?
    'with wind-aware planning and anticipated learning' : 'when anticipating learning';
}

export function methodDescription(data) {
  if (data.metrics.method === 'wind_blind_planner') return 'Chooses a still-air nominal trajectory. Only the continuous execution safety filter uses the learned wind.';
  if (data.metrics.method === 'current_belief') return 'Plans through wind-aware safety using today’s belief; learns from real measurements but does not anticipate future updates.';
  return 'Plans through wind-aware safety and simulates how earned measurements change future decisions.';
}

export function missionSummary(data) {
  const {
    mission: m,
    config: c,
    environment: e
  } = data;
  const cost = cumulativeCost(m.q, e.targets, c.dt);
  const first = cost[c.stage_steps];
  return {
    cost: cost.at(-1),
    first,
    later: cost.at(-1) - first,
    finalDeficit: deficit(m.q.at(-1), e.targets),
    peakSlack: Math.max(0, ...m.slack),
    integratedSlack: m.slack.reduce((sum, v) => sum + Math.max(v, 0) * c.dt, 0),
    relaxedSteps: m.slack.filter(v => v > 0).length,
    relaxedSeconds: m.slack.filter(v => v > 0).length * c.dt,
    clearance: Math.min(...m.clearance),
    collision: Boolean(data.metrics.collision),
    accessTime: data.metrics.roi_a_access ? data.metrics.roi_a_time : null,
    duration: (m.x.length - 1) * c.dt,
    costHistory: cost,
    deficitHistory: m.q.map((q,i)=>[i*c.dt,deficit(q,e.targets)])
  };
}

// Interpolate recorded execution states for display only; never feeds the planner.
export function playbackFrame(data, seconds) {
  const last = data.mission.x.length - 1;
  const sample = clamp(seconds / data.config.dt, 0, last);
  const index = Math.floor(sample),
    next = Math.min(index + 1, last),
    mix = sample - index;
  const interpolate = values => values[index].map((v, i) => v + mix * (values[next][i] - v));
  return {
    index,
    next,
    mix,
    x: interpolate(data.mission.x),
    q: interpolate(data.mission.q)
  };
}

export function advanceComparison(clock, elapsed, speed, duration, loop) {
  const next = clock + Math.max(elapsed, 0) * speed;
  if (loop) return {
    clock: next % (duration + 2),
    playing: true
  };
  return {
    clock: Math.min(next, duration),
    playing: next < duration
  };
}

// Persistent monitoring diagnostics from executed states, independent of ROI labels.
// A sensing pass has C^2 >= 0.25 for at least two samples. Gaps < 3 s stay one pass.
export function persistenceMetrics(data) {
  const {mission:m, config:c, environment:e} = data;
  const visits = [], recoveries = [], intervals = [];
  const passHistory = Array.from({length:m.x.length},()=>new Uint8Array(e.cells.length));
  e.cells.forEach((cell, k) => {
    let count = 0, lastNear = -Infinity, run = 0, eligible = false, lastPass = null;
    let met = m.q[0][k] >= e.targets[k], lossStart = null, recovered = 0;
    m.x.forEach((x, i) => {
      const near = Math.exp(-((x[0]-cell[0])**2+(x[1]-cell[1])**2)/c.sensor_sigma**2) >= .25;
      if (near) {
        if (!run) eligible = !count || (i-lastNear)*c.dt >= 3;
        run++;
        if (run === 2 && eligible) {
          const start = (i-1)*c.dt;
          if (lastPass !== null) intervals.push(start-lastPass);
          lastPass = start;
          count++;
        }
        lastNear = i;
      } else run = 0;
      passHistory[i][k] = Math.min(count,255);
      if (m.q[i][k] < e.targets[k]) {
        if (met && lossStart === null) lossStart = i;
      } else {
        if (met && lossStart !== null && (i-lossStart)*c.dt >= 1) recovered++;
        met = true;
        lossStart = null;
      }
    });
    visits.push(count);
    recoveries.push(recovered);
  });
  intervals.sort((a,b)=>a-b);
  return {
    visits, recoveries, passHistory,
    visitedCells: visits.filter(v=>v>0).length,
    revisitedCells: visits.filter(v=>v>1).length,
    recoveredCells: recoveries.filter(v=>v>0).length,
    medianRevisitSeconds: intervals.length ? (intervals[Math.floor((intervals.length-1)/2)] + intervals[Math.floor(intervals.length/2)]) / 2 : null
  };
}
