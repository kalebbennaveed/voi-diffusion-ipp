export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
export function deficit(q, targets) {
  return q.reduce((s, v, i) => s + Math.max(0, targets[i] - v), 0) / q.length;
}
export function cumulativeCost(q, targets, dt) {
  const c = [0];
  for (let i = 1; i < q.length; i++) c.push(c.at(-1) + dt * (deficit(q[i - 1], targets) + deficit(q[i],
    targets)) / 2);
  return c;
}
export function beliefIndex(step, H, count) {
  return clamp(Math.floor(step / H), 0, count - 1);
}
export function pairedRecord(records, record) {
  return records.find(r => r.group === record.group && r.seed === record.seed && r.method === (record
    .method === 'current_belief' ? 'proposed' : 'current_belief')) || null;
}
export function metricText(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : 'Infeasible';
}
