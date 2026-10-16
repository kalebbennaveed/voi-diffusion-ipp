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
export function pairedRecord(records, record) {
  return records.find(r => r.group === record.group && r.seed === record.seed && r.method === (
    record
    .method === 'current_belief' ? 'proposed' : 'current_belief')) || null;
}
export function metricText(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : 'Infeasible';
}

export function methodLabel(data) {
  if (data.metrics.method === 'current_belief') return 'No anticipated learning';
  if (!data.config.measurements) return 'Proposed · measurements off';
  if (data.metrics.method === 'cem') return 'CEM · anticipates learning';
  if (data.metrics.method === 'mppi') return 'MPPI · anticipates learning';
  return 'Proposed · anticipates learning';
}

// Keep method identity and colors stable even when the baseline record is selected.
export function comparisonPair(data, other) {
  if (data.metrics.method === 'current_belief' && other) return [other, data];
  return [data, other];
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
    costHistory: cost
  };
}
