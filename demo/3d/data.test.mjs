import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {toScene,frameAt,regionIndices,regionMean,diffusionAt,fieldColor,gridShape,sensorRadius,sensorWeight,sensingStatus,motionForces} from './data.js';
import {missionSummary} from '../model.js';
import {DISPLAY_CEILING,terrainHeight,ridgeVertex,cameraSightClear,chasePosition} from './terrain-math.js';

test('sensor rings describe H squared information weight, not a hard field of view',()=>{
  for(const sigma of [.2,.5,1.2])for(const weight of [.1,.5,.9]){
    assert.ok(Math.abs(sensorWeight(sensorRadius(sigma,weight),sigma)-weight)<1e-12);
  }
  assert.ok(sensorWeight(sensorRadius(.5,.1)*2,.5)>0);
  const data={config:{sensor_sigma:.5},environment:{roi_a:[0,0],roi_b:[2,0],region_size:[.8,.8],cells:[[0,0],[.4,0],[2,0]]}};
  const s=sensingStatus(data,{x:[0,0,0,0],q:[.8,.6,.2]});
  assert.deepEqual(s.regions[0],{clarity:.7,covered:2,total:2});
  assert.equal(s.regions[1].covered,0);assert.equal(s.nearest,0);
});

test('motion residual recovers disturbance with drag, not just net acceleration',()=>{
  const dt=.05,drag=1.3,u=[.4,-.8],wind=[-.7,1.2],v=[.6,-.2],p=[1,-2];
  const terminal=u.map((x,i)=>(x+wind[i])/drag),decay=Math.exp(-drag*dt);
  const vn=v.map((x,i)=>terminal[i]+(x-terminal[i])*decay);
  const pn=p.map((x,i)=>x+terminal[i]*dt+(v[i]-terminal[i])*(1-decay)/drag);
  const data={config:{dt,linear_drag:drag},mission:{x:[[...p,...v],[...pn,...vn]],control:[u]}};
  const f=motionForces(data,200);
  f.wind.forEach((x,i)=>assert.ok(Math.abs(x-wind[i])<1e-12));
  f.net.forEach((x,i)=>assert.ok(Math.abs(x-f.wind[i]-f.control[i]-f.drag[i])<1e-12));
  assert.equal(f.interval,0);
});

test('chase camera keeps the aircraft visible without changing its position or flight altitude',()=>{
  const obstacles=[[0,0,.5],[1.8,-.5,.4]];
  for(let angle=0;angle<Math.PI*2;angle+=.17){
    const origin=[Math.cos(angle)*.7,.58,Math.sin(angle)*.7],before=[...origin];
    for(let heading=0;heading<Math.PI*2;heading+=.31){
      const eye=chasePosition(origin,heading,obstacles);
      assert.ok(eye.every(Number.isFinite));
      assert.ok(cameraSightClear(origin,eye,obstacles));
      assert.ok(eye[1]>terrainHeight(eye[0],-eye[2]));
      assert.deepEqual(origin,before);
    }
  }
  assert.equal(cameraSightClear([-.8,.58,0],[.8,.9,0],obstacles),false);
});

test('background ridge geometry stays bounded and display altitude has no climb action',()=>{
  const obstacle=[.37,-.62,.5];let peak=0;
  for(let level=0;level<7;level++)for(let angle=0;angle<Math.PI*2;angle+=.03){
    const [x,h,y]=ridgeVertex(obstacle,level,angle);
    assert.ok(Math.hypot(x-obstacle[0],y-obstacle[1])<=obstacle[2]+1e-12);
    assert.ok(Number.isFinite(h));peak=Math.max(peak,h);
  }
  assert.ok(peak>DISPLAY_CEILING*1.15);
  assert.ok(peak<1.1,'natural low escarpment, not the former tall spire');
  assert.equal(toScene([1,2,2000,-50])[1],toScene([-2,1,-2000,50])[1]);
  for(let x=-2.7;x<2.7;x+=.1)for(let y=-2.1;y<2.1;y+=.1){
    assert.ok(terrainHeight(x,y)+.29<toScene([x,y])[1]-.1,'valley trees remain below the displayed aircraft');
  }
});

test('planar coordinates preserve distances and handedness in the display plane',()=>{
  assert.deepEqual(toScene([2,-3],.5),[2,.5,3]);
  const a=[-1,2],b=[3,-1],sa=toScene(a),sb=toScene(b);
  assert.equal(Math.hypot(a[0]-b[0],a[1]-b[1]),Math.hypot(...sa.map((v,i)=>v-sb[i])));
});

test('display interpolation never grants the next POAM update early',()=>{
  const data={config:{dt:.5,stage_steps:2},mission:{x:[[0,0,1,0],[.5,0,1,0],[1,0,1,0],[1.5,0,1,0],[2,0,1,0]],q:[[0],[.2],[.4],[.6],[.8]]},beliefs:['prior','first','second'],planning:[{},{}]};
  assert.equal(frameAt(data,.999).belief,'prior');
  assert.equal(frameAt(data,1).belief,'first');
  assert.equal(frameAt(data,2).belief,'second');
  assert.equal(frameAt(data,2).period,1);
  assert.deepEqual(frameAt(data,100).x,[2,0,1,0]);
  assert.deepEqual(frameAt(data,.25).q,[.1]);
});

test('regional clarity is averaged over the recorded rectangle, not a point ROI',()=>{
  const e={cells:[[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0]],roi_a:[-.5,-.5],region_size:[1,1]};
  assert.deepEqual(gridShape(e),{nx:3,ny:2});
  assert.deepEqual(regionIndices(e,'roi_a'),[0,1,3,4]);
  assert.equal(regionMean([0,.2,1,.6,.8,1],regionIndices(e,'roi_a')),.4);
  assert.equal(regionMean([1],[]),null);
});

test('rejected selected forecasts and fallback are explicit, never called executed motion',()=>{
  const p={candidate_paths:[[[[0,0],[1,0]]]],costs:[[null]],incumbent_paths:[[[0,0],[.1,0]]],mean_paths:[[[0,0],[.2,0]]],selected_path:[[0,0],[1,1]],selected_cost:null,backup_used:true};
  const r=diffusionAt({planning:[p]},99,99);
  assert.equal(r.level,0);assert.equal(r.rejected,true);assert.equal(r.fallback,true);
  assert.equal(r.candidates,p.candidate_paths[0]);assert.equal(r.selected,p.selected_path);
});

test('layer scales are fixed across methods and bounded at their declared extremes',()=>{
  for(const layer of ['clarity','deficit','uncertainty']){
    assert.deepEqual(fieldColor(-10,layer),fieldColor(0,layer));
    assert.deepEqual(fieldColor(10,layer),fieldColor(1,layer));
    assert.notDeepEqual(fieldColor(0,layer),fieldColor(1,layer));
  }
});

const path=new URL('../../results/demo/live/budgeted_persistent_two_region_2s_01_proposed_8101/replay.json',import.meta.url);
test('the complete local mission keeps identical endpoints, POAM and task cost in 3D',{skip:!fs.existsSync(path)},()=>{
  const d=JSON.parse(fs.readFileSync(path,'utf8'));
  assert.deepEqual(frameAt(d,120).x,d.mission.x.at(-1));
  assert.deepEqual(frameAt(d,120).q,d.mission.q.at(-1));
  assert.equal(frameAt(d,120).belief,d.beliefs.at(-1));
  assert.ok(Math.abs(missionSummary(d).cost-d.metrics.integrated_deficit)<1e-6);
  assert.deepEqual(gridShape(d.environment),{nx:41,ny:33});
  for(const region of ['roi_a','roi_b'])assert.ok(regionIndices(d.environment,region).length>50);
  for(let p=0;p<d.planning.length;p++)for(let l=0;l<d.planning[p].candidate_paths.length;l++){
    const trace=diffusionAt(d,p,l);
    assert.equal(trace.candidates,d.planning[p].candidate_paths[l]);
    assert.equal(trace.incumbent,d.planning[p].incumbent_paths[l]);
  }
});
