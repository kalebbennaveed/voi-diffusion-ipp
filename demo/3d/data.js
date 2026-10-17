// Display-only transforms. The recorded planar states and forecasts are authoritative.
import {clamp, playbackFrame} from '../model.js';

export const FLIGHT_HEIGHT = 0.58; // Illustrative, never used for collision or task metrics.
export const toScene = (point, height = FLIGHT_HEIGHT) => [point[0], height, -point[1]];
export const duration = data => (data.mission.x.length - 1) * data.config.dt;
export const replanSeconds = data => data.config.stage_steps * data.config.dt;

export function frameAt(data, seconds) {
  const frame = playbackFrame(data, seconds);
  const boundary = Math.floor(clamp(seconds, 0, duration(data)) / replanSeconds(data) + 1e-8);
  return {...frame, beliefBoundary: Math.min(boundary, data.beliefs.length - 1),
    belief: data.beliefs[Math.min(boundary, data.beliefs.length - 1)],
    period: Math.min(boundary, data.planning.length - 1)};
}

export function regionIndices(environment, region) {
  const center = environment[region], size = environment.region_size;
  if (!center || !size) return [];
  return environment.cells.flatMap((p, i) => Math.abs(p[0] - center[0]) <= size[0] / 2 &&
    Math.abs(p[1] - center[1]) <= size[1] / 2 ? [i] : []);
}
export const regionMean = (q, indices) => indices.length ? indices.reduce((s,i) => s+q[i],0)/indices.length : null;

// The clarity ODE uses H²/R, so relative information weight is exp(-r²/sigma²).
export const sensorWeight = (distance,sigma) => Math.exp(-((distance/sigma)**2));
export const sensorRadius = (sigma,weight=.5) => sigma*Math.sqrt(-Math.log(weight));
export function sensingStatus(data,frame){
  const cells=data.environment.cells, sigma=data.config.sensor_sigma;
  const regions=['roi_a','roi_b'].map(key=>{
    const ids=regionIndices(data.environment,key);
    const covered=ids.filter(k=>sensorWeight(Math.hypot(cells[k][0]-frame.x[0],cells[k][1]-frame.x[1]),sigma)>=.5).length;
    return {clarity:regionMean(frame.q,ids),covered,total:ids.length};
  });
  const nearest=cells.reduce((best,p,k)=>Math.hypot(p[0]-frame.x[0],p[1]-frame.x[1])<Math.hypot(cells[best][0]-frame.x[0],cells[best][1]-frame.x[1])?k:best,0);
  return {regions,nearest};
}

export function motionForces(data,index){
  const k=Math.max(0,Math.min(index,data.mission.control.length-1));
  const a=data.mission.x[k],b=data.mission.x[k+1],dt=data.config.dt;
  const control=data.mission.control[k],dragCoefficient=data.config.linear_drag||0;
  // Position differences give the same RK4-weighted mean velocity used by the
  // integrator. Recover its interval-mean disturbance, including linear drag.
  const net=[(b[2]-a[2])/dt,(b[3]-a[3])/dt];
  const drag=[-dragCoefficient*(b[0]-a[0])/dt,-dragCoefficient*(b[1]-a[1])/dt];
  const wind=net.map((v,j)=>v-control[j]-drag[j]);
  const speed=Math.hypot(a[2],a[3]);
  const along=speed>.03?(wind[0]*a[2]+wind[1]*a[3])/speed:0;
  return {wind,control,drag,net,speed,along,interval:k};
}

export function diffusionAt(data, period, level) {
  const p = data.planning[clamp(period, 0, data.planning.length - 1)];
  const l = clamp(level, 0, p.candidate_paths.length - 1);
  return {period: p, level: l, candidates: p.candidate_paths[l], costs: p.costs[l],
    incumbent: p.incumbent_paths[l], mean: p.mean_paths[l], selected: p.selected_path,
    rejected: !Number.isFinite(p.selected_cost), fallback: Boolean(p.backup_used)};
}

// Shared physical scales across both arms. Colors encode numbers, not method identity.
export function fieldColor(value, layer) {
  const stops = layer === 'deficit' ? [[34,49,61],[157,81,67],[244,187,111]] :
    layer === 'uncertainty' ? [[34,49,61],[91,104,169],[220,189,239]] :
    [[36,52,67],[45,125,131],[103,217,187],[228,239,167]];
  const v = clamp(value,0,1)*(stops.length-1), i = Math.min(Math.floor(v),stops.length-2), t=v-i;
  return stops[i].map((c,k)=>Math.round(c+(stops[i+1][k]-c)*t));
}

export function gridShape(environment) {
  const nx = new Set(environment.cells.map(p=>p[0])).size;
  const ny = environment.cells.length / nx;
  if (!Number.isInteger(ny)) throw new Error('The recording does not contain a rectangular grid.');
  return {nx,ny};
}
