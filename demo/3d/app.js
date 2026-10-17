import {FlightScene} from './scene.js';
import {frameAt, duration, replanSeconds, regionIndices, regionMean, diffusionAt, FLIGHT_HEIGHT, sensingStatus, motionForces, gridShape, fieldColor, sensorRadius} from './data.js';
import {missionSummary, pairedRecord} from '../model.js';

const $=id=>document.getElementById(id);
const DEFAULT='budgeted_persistent_two_region_2s_01_proposed_8101';
const COLORS=[0x36cbb4,0xe7b468], CSS_COLORS=['#56cfba','#e5b264'];
const query=new URLSearchParams(location.search);
const state={mode:'execution',time:0,playing:false,speed:4,period:0,level:0,levelTime:0,follow:['aircraft','helicopter'].includes(query.get('camera')),sensor:query.get('camera')==='sensor',windFlow:true,clarityLayer:'clarity',ready:false};
let records=[],record,scenes=[],data=[],summaries=[],regions=[],histories=[],forecastKey='',frameKey='',lastFrame=0,lastInfo=0;
const cache=new Map();let activeCamera=0;

function error(message){$('error').textContent=message;$('error').hidden=false;$('status').textContent='Replay unavailable';state.playing=false;$('play').disabled=true;}
document.addEventListener('scene-error',event=>error(event.detail));
async function getJSON(url){const response=await fetch(url);if(!response.ok)throw new Error(`Could not load ${url} (${response.status}).`);return response.json();}
async function replay(id){if(!cache.has(id))cache.set(id,getJSON(`/api/replay/${encodeURIComponent(id)}`));return cache.get(id);}
const number=(v,digits=2)=>Number.isFinite(v)?v.toFixed(digits):'—';

function updateURL(){
  if(!record)return;
  const params=new URLSearchParams({run:record.id,baseline:$('baseline').value});
  $('open-2d').href=`/?${params}`;
  if(state.follow)params.set('camera','aircraft');
  if(state.sensor)params.set('camera','sensor');
  history.replaceState(null,'',`/3d/?${params}`);
}
async function loadPair(){
  state.ready=false;state.playing=false;$('play').disabled=true;$('baseline').disabled=true;$('error').hidden=true;
  $('status').textContent='Loading recorded flights…';
  try{
    const other=pairedRecord(records,record,$('baseline').value);
    if(!other)throw new Error('A matched recording is not available for this comparator. The original 2D demo contains the full catalog.');
    const next=await Promise.all([replay(record.id),replay(other.id)]);
    if(next.some(d=>!d.planning?.[0]?.candidate_paths||!d.beliefs?.length))throw new Error('This recording has no exported diffusion forecasts or wind beliefs. Choose the complete two-region mission in the original demo.');
    scenes.forEach(s=>s.dispose());scenes=[];data=next;
    summaries=data.map(missionSummary);regions=data.map(d=>['roi_a','roi_b'].map(k=>regionIndices(d.environment,k)));
    histories=data.map((d,i)=>d.mission.q.map(q=>regions[i].map(indices=>regionMean(q,indices))));
    data.forEach((d,i)=>{const s=new FlightScene($(i?'scene-b':'scene-a'),d,COLORS[i]);scenes.push(s);
      s.controls.addEventListener('start',()=>{activeCamera=i;state.follow=false;$('follow').setAttribute('aria-pressed','false');updateURL();});});
    $('status').textContent='Loading aircraft and terrain…';
    await Promise.all(scenes.map(s=>s.ready));
    const limit=Math.min(...data.map(duration));$('timeline').max=limit;$('timeline').value=0;
    $('period').replaceChildren(...data[0].planning.map((_,i)=>new Option(`${(i*replanSeconds(data[0])).toFixed(0)} s · replan ${i+1}`,String(i))));
    const blind=other.method==='wind_blind_planner';
    $('baseline-title').textContent=blind?'Wind-blind planner':'Without anticipated learning';
    $('baseline-subtitle').textContent=blind?'Wind-aware safety during execution':'Wind-aware planning with the current belief';
    $('comparison-note').textContent=blind?'Both aircraft use wind-aware safety during execution. The right planner chooses its nominal path in still air; the left also models wind and future learning. This comparison combines both effects.':'Both aircraft plan with wind and learn from real observations. Only the left planner anticipates how future measurements can change later decisions. This isolates anticipated learning.';
    const gain=100*(summaries[1].cost-summaries[0].cost)/summaries[1].cost;
    $('outcome-title').textContent=`${Math.abs(gain).toFixed(1)}% ${gain>=0?'lower':'higher'} cost`;
    $('outcome-detail').textContent=`Proposed ${summaries[0].cost.toFixed(3)} · comparator ${summaries[1].cost.toFixed(3)}. ${blind?'Combined wind-aware planning and anticipation effect.':'Anticipation effect.'} One development seed; not statistical evidence.`;
    $('outcome-safety').textContent=`Physical contact: ${summaries.map((s,i)=>`${i?'comparator':'proposed'} ${s.collision?'yes':'none recorded'}`).join(' · ')}.`;
    $('source-note').textContent=`Recordings: ${record.id} and ${other.id}. Playback uses the existing data and starts no JAX experiments. Renderer and controls are vendored locally; no CDN connection is needed.`;
    state.time=0;state.period=0;state.level=0;state.levelTime=0;state.ready=true;forecastKey='';frameKey='';
    state.mode='execution';setMode('execution');$('status').textContent=`${limit.toFixed(0)} s recorded · synchronized replay`;
    $('play').disabled=false;updateURL();refresh(true);
  }catch(e){error(`${e.message} Open the original 2D demo using the link above.`);console.error(e);}
  finally{$('baseline').disabled=false;}
}

function setMode(mode){
  state.mode=mode;state.playing=false;state.levelTime=0;
  document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode)));
  $('diffusion-controls').hidden=mode!=='denoising';
  if(state.ready){
    if(mode==='final')state.time=Math.min(...data.map(duration));
    if(mode==='denoising'){
      state.period=Math.min(Math.floor(state.time/replanSeconds(data[0])),data[0].planning.length-1);
      state.time=state.period*replanSeconds(data[0]);state.level=0;$('period').value=state.period;$('level').value=0;
    }
    forecastKey='';refresh(true);
  }
  updateTransport();
}
function updateTransport(){
  $('play').textContent=state.playing?'Ⅱ Pause':state.mode==='denoising'?'▶ Play denoising':state.mode==='final'?'▶ Replay mission':'▶ Play mission';
  $('clock').textContent=`${state.time.toFixed(1)} / ${Number($('timeline').max).toFixed(1)} s`;
  $('timeline').value=state.time;$('time-readout').textContent=`${state.time.toFixed(1)} s`;
}
function forecasts(){
  const key=`${state.mode}:${state.period}:${state.level}:${$('candidates').checked}`;
  if(key===forecastKey)return;forecastKey=key;
  data.forEach((d,i)=>{
    const p=d.planning[Math.min(state.period,d.planning.length-1)],paths=[];
    const blind=d.metrics.method==='wind_blind_planner';
    const forecastType=blind?'Still-air nominal forecast; execution uses wind-aware RPCBF.':'RPCBF forecast in one sampled planning world; actual wind can differ.';
    if(state.mode==='denoising'){
      const trace=diffusionAt(d,state.period,state.level);
      if($('candidates').checked)trace.candidates.forEach((points,k)=>paths.push({points,color:Number.isFinite(trace.costs[k])?0xb8b0ed:0xe76a63,opacity:.37,height:FLIGHT_HEIGHT+.008}));
      paths.push({points:trace.mean,color:0xf0f3f7,opacity:.8,height:FLIGHT_HEIGHT+.016});
      paths.push({points:trace.incumbent,color:COLORS[i],height:FLIGHT_HEIGHT+.025});
      if(state.level===p.candidate_paths.length-1)paths.push({points:trace.selected,color:trace.rejected?0xf17a6c:COLORS[i],height:FLIGHT_HEIGHT+.035});
      const costs=trace.costs.filter(Number.isFinite),best=costs.length?Math.min(...costs):null;
      $(i?'forecast-b':'forecast-a').textContent=`Level ${trace.level+1}: ${costs.length}/${trace.costs.length} finite candidates · best sampled cost ${number(best,4)}. ${forecastType}${trace.fallback?' Recorded fallback used.':''}${trace.rejected?' Selected plan was rejected; forecast is not an executed plan.':''}`;
    }else if(state.mode==='execution'){
      paths.push({points:p.selected_path,color:Number.isFinite(p.selected_cost)?COLORS[i]:0xf17a6c,opacity:.4});
      $(i?'forecast-b':'forecast-a').textContent=`${(state.period*replanSeconds(d)).toFixed(0)} s replan · ${(p.selected_path.length-1)*(d.config.planning_dt||d.config.dt)} s forecast. ${forecastType}${p.backup_used?' Recorded fallback used.':''}${!Number.isFinite(p.selected_cost)?' Selected forecast rejected.':''}`;
    }else $(i?'forecast-b':'forecast-a').textContent=`Complete executed path · minimum physical clearance ${number(summaries[i].clearance,3)} m · integrated deficit ${number(summaries[i].cost,3)}.`;
    scenes[i].setForecast(paths);
  });
  $('level-readout').textContent=`Level ${state.level+1} / ${data[0].planning[state.period].candidate_paths.length}`;
  $('path-legend').textContent=state.mode==='denoising'?'Violet: proposals · white: denoised mean · method color: incumbent · red: rejected':'Solid trail: executed motion · thin line: selected 10 s forecast';
}

function refresh(force=false){
  if(!state.ready)return;
  const key=`${state.time}:${state.mode}:${state.level}:${$('layer').value}:${$('wind').value}:${$('candidates').checked}`;
  if(!force&&key===frameKey)return;frameKey=key;
  const frames=data.map(d=>frameAt(d,state.time));
  state.period=frames[0].period;
  frames.forEach((frame,i)=>scenes[i].setFrame(frame,$('layer').value,$('wind').value,state.time));
  forecasts();
  if(force||!state.playing||performance.now()-lastInfo>100){
    frames.forEach((frame,i)=>{
      const q=regions[i].map(indices=>regionMean(frame.q,indices));
      const clearance=frame.index===0?null:data[i].mission.clearance[Math.min(frame.index-1,data[i].mission.clearance.length-1)];
      $(i?'metrics-b':'metrics-a').innerHTML=[['Clarity A',number(q[0])],['Clarity B',number(q[1])],['Deficit cost',number(summaries[i].costHistory[frame.index],3)],['Clearance',`${number(clearance,2)} m`]].map(([label,value])=>`<span>${label}<strong${label==='Clearance'&&clearance<0?' class="unsafe"':''}>${value}</strong></span>`).join('');
    });
    updateTransport();drawChart('deficit-chart','deficit');drawChart('clarity-chart','clarity');drawMissionHUD(frames);lastInfo=performance.now();
  }
}

function drawMissionHUD(frames){
  const status=frames.map((f,i)=>sensingStatus(data[i],f));
  $('deficit-now').innerHTML=frames.map((f,i)=>`<span>${i?'Comparator':'Proposed'} ${number(summaries[i].deficitHistory[f.index][1],3)} · accumulated ${number(summaries[i].costHistory[f.index],3)}</span>`).join('');
  $('site-status').innerHTML=['North wetland · A','South wetland · B'].map((name,k)=>`<div class="site-row"><span>${name}</span>${status.map((s,i)=>{const r=s.regions[k];return `<div class="freshness ${i?'comparator':''}"><div class="freshness-bar"><i style="width:${Math.max(0,Math.min(100,r.clarity*100))}%"></i></div><b>${number(r.clarity)}</b><small>${i?'Comparator':'Proposed'} · ${r.covered?`sensing ${r.covered}/${r.total} cells`:'outside 50% footprint'}</small></div>`;}).join('')}</div>`).join('');
  frames.forEach((f,i)=>{
    const arm=i?'b':'a',s=status[i],forces=motionForces(data[i],f.index),wind=Math.hypot(...forces.wind),control=Math.hypot(...forces.control);
    drawWindMap(`wind-map-${arm}`,data[i],f);
    drawClarityMap(arm,data[i],f,s);
    const mean=f.belief.mean[s.nearest],std=f.belief.std[s.nearest];
    $(`wind-readout-${arm}`).textContent=`At aircraft: |μ| ${number(Math.hypot(...mean))} · σ ${number(std)} m/s²`;
    let effect=$(`scene-${arm}`).querySelector('.wind-effect');if(!effect){effect=document.createElement('div');effect.className='wind-effect';$(`scene-${arm}`).append(effect);}
    const description=wind<.08?'Weak disturbance':forces.speed<.04?'Wind push at low ground speed':forces.along>.15?'Wind assists current travel':forces.along<-.15?'Wind opposes current travel':'Crosswind push';
    effect.innerHTML=`<h4>${description}</h4><p><span class="push">Wind ${number(wind)} m/s²</span> · <span class="control">Control ${number(control)} m/s²</span></p><p>Drag ${number(Math.hypot(...forces.drag))} · net ${number(Math.hypot(...forces.net))} m/s²</p><p>Ground speed ${number(forces.speed)} m/s</p><small>Wind reconstructed from recorded motion + control.<br>Orange / blue arrows: wind / control; tilt illustrative.</small>`;
    $(`scene-hint-${arm}`).textContent=state.sensor?'Downward view from aircraft · red footprint: sensing strength':'Drag to orbit · red footprint: sensing · streaks: physical wind';
  });
}

function drawClarityMap(arm,d,frame,status){
  const canvas=$(`clarity-map-${arm}`),rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height;if(!w||!h)return;
  const ratio=Math.min(devicePixelRatio,2);if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);}
  const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);
  const deficit=state.clarityLayer==='deficit',limit=deficit?Math.max(...data.flatMap(d=>d.environment.targets),.001):1;
  const [lo,hi]=d.environment.bounds,{nx,ny}=gridShape(d.environment),pad=8,scale=Math.min((w-2*pad)/(hi[0]-lo[0]),(h-2*pad)/(hi[1]-lo[1]));
  const X=x=>w/2+(x-(lo[0]+hi[0])/2)*scale,Y=y=>h/2-(y-(lo[1]+hi[1])/2)*scale;
  const cw=(hi[0]-lo[0])*scale/(nx-1),ch=(hi[1]-lo[1])*scale/(ny-1);
  ctx.save();ctx.beginPath();ctx.rect(X(lo[0])-cw/2,Y(hi[1])-ch/2,(hi[0]-lo[0])*scale+cw,(hi[1]-lo[1])*scale+ch);ctx.clip();
  d.environment.cells.forEach((p,k)=>{
    const value=deficit?Math.max(0,d.environment.targets[k]-frame.q[k])/limit:frame.q[k];
    ctx.fillStyle=`rgb(${fieldColor(value,state.clarityLayer).join(',')})`;ctx.fillRect(X(p[0])-cw/2,Y(p[1])-ch/2,cw+1,ch+1);
  });
  ctx.fillStyle='#101b23';d.environment.obstacles.forEach(([x,y,r])=>{ctx.beginPath();ctx.arc(X(x),Y(y),r*scale,0,Math.PI*2);ctx.fill();});
  const radius=sensorRadius(d.config.sensor_sigma)*scale;
  ctx.beginPath();ctx.arc(X(frame.x[0]),Y(frame.x[1]),radius,0,Math.PI*2);ctx.fillStyle='#ff554426';ctx.fill();ctx.strokeStyle='#ff7766';ctx.lineWidth=1.5;ctx.stroke();
  for(const [key,label] of [['roi_a','A'],['roi_b','B']]){
    const [x,y]=d.environment[key],[rx,ry]=d.environment.region_size;
    ctx.strokeStyle='#0c1824';ctx.lineWidth=2.5;ctx.strokeRect(X(x-rx/2),Y(y+ry/2),rx*scale,ry*scale);
    ctx.strokeStyle='#eff4e2';ctx.lineWidth=.8;ctx.strokeRect(X(x-rx/2),Y(y+ry/2),rx*scale,ry*scale);
    ctx.font='bold 10px system-ui';ctx.lineWidth=3;ctx.strokeStyle='#12242e';ctx.strokeText(label,X(x-rx/2)+3,Y(y+ry/2)+11);ctx.fillStyle='#fff9df';ctx.fillText(label,X(x-rx/2)+3,Y(y+ry/2)+11);
  }
  ctx.beginPath();ctx.arc(X(frame.x[0]),Y(frame.x[1]),2.4,0,Math.PI*2);ctx.fillStyle='#fff5dc';ctx.fill();ctx.strokeStyle='#8d2828';ctx.lineWidth=1;ctx.stroke();ctx.restore();
  ctx.fillStyle='#d4e6df';ctx.font='8px system-ui';ctx.fillText('N ↑',w-22,8);
  $(`clarity-title-${arm}`).firstChild.textContent=deficit?'Clarity deficit ':'Whole-field clarity ';
  canvas.setAttribute('aria-label',`${arm==='a'?'Proposed':'Comparator'} whole-field ${state.clarityLayer} at ${state.time.toFixed(1)} seconds`);
  $(`clarity-scale-${arm}`).style.background=deficit?'linear-gradient(90deg,#22313d,#9d5143,#f4bb6f)':'linear-gradient(90deg,#243443,#2d7d83,#67d9bb,#e4efa7)';
  $(`clarity-low-${arm}`).textContent=deficit?'0 · target met':'0 · stale';
  $(`clarity-high-${arm}`).textContent=deficit?`${limit.toFixed(2)} · deficit`:'1 · clear';
  $(`clarity-readout-${arm}`).textContent=`Mean q: A ${number(status.regions[0].clarity)} · B ${number(status.regions[1].clarity)} · ${state.time.toFixed(1)} s`;
}

function drawWindMap(id,d,frame){
  const canvas=$(id),rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height;if(!w||!h)return;
  const ratio=Math.min(devicePixelRatio,2);if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);}
  const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);
  const [lo,hi]=d.environment.bounds,{nx,ny}=gridShape(d.environment),pad=8,s=Math.min((w-2*pad)/(hi[0]-lo[0]),(h-2*pad)/(hi[1]-lo[1]));
  const X=x=>w/2+(x-(lo[0]+hi[0])/2)*s,Y=y=>h/2-(y-(lo[1]+hi[1])/2)*s;
  const cw=(hi[0]-lo[0])*s/(nx-1),ch=(hi[1]-lo[1])*s/(ny-1);
  d.environment.cells.forEach((p,k)=>{const rgb=fieldColor(frame.belief.std[k],'uncertainty');ctx.fillStyle=`rgb(${rgb.join(',')})`;ctx.fillRect(X(p[0])-cw/2,Y(p[1])-ch/2,cw+1,ch+1);});
  ctx.fillStyle='#17212a';d.environment.obstacles.forEach(([x,y,r])=>{ctx.beginPath();ctx.arc(X(x),Y(y),r*s,0,Math.PI*2);ctx.fill();});
  ctx.strokeStyle='#d3dfda';ctx.lineWidth=.7;for(const key of ['roi_a','roi_b']){const [x,y]=d.environment[key],[rx,ry]=d.environment.region_size;ctx.strokeRect(X(x-rx/2),Y(y+ry/2),rx*s,ry*s);}
  ctx.strokeStyle='#ecf3e7';ctx.lineWidth=1;
  for(let j=2;j<ny;j+=5)for(let i=2;i<nx;i+=5){const k=j*nx+i,p=d.environment.cells[k];if(d.environment.obstacles.some(o=>Math.hypot(p[0]-o[0],p[1]-o[1])<o[2]))continue;
    const [vx,vy]=frame.belief.mean[k],mag=Math.hypot(vx,vy);if(mag<.015)continue;const len=Math.min(.22*mag,.48)*s,dx=vx/mag,dy=-vy/mag,x=X(p[0]),y=Y(p[1]),ex=x+dx*len,ey=y+dy*len;
    ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(ex,ey);ctx.lineTo(ex-dx*3-dy*2,ey-dy*3+dx*2);ctx.moveTo(ex,ey);ctx.lineTo(ex-dx*3+dy*2,ey-dy*3-dx*2);ctx.stroke();
  }
  ctx.strokeStyle='#ffdf88';ctx.beginPath();ctx.arc(X(frame.x[0]),Y(frame.x[1]),3,0,Math.PI*2);ctx.stroke();ctx.fillStyle='#e8f0ef';ctx.font='8px system-ui';ctx.fillText('N ↑',w-22,9);
}

function drawChart(id,kind){
  const c=$(id),rect=c.getBoundingClientRect(),ratio=Math.min(devicePixelRatio,2),w=rect.width,h=rect.height;
  if(c.width!==Math.round(w*ratio)||c.height!==Math.round(h*ratio)){c.width=Math.round(w*ratio);c.height=Math.round(h*ratio);}
  const ctx=c.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);
  const pad={l:33,r:8,t:8,b:24},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b,maxTime=Number($('timeline').max);
  const max=kind==='clarity'?1:Math.max(...summaries.flatMap(s=>s.deficitHistory.map(p=>p[1])))*1.1;
  const X=t=>pad.l+pw*t/maxTime,Y=v=>pad.t+ph*(1-v/max);
  ctx.font='9px system-ui';ctx.lineWidth=1;ctx.fillStyle='#8ba0ad';ctx.strokeStyle='#2c3b45';
  for(let i=0;i<=2;i++){const v=max*i/2,y=Y(v);ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();ctx.fillText(v.toFixed(2),0,y+3);}
  for(let t=0;t<=maxTime;t+=maxTime/4)ctx.fillText(`${Math.round(t)}s`,X(t)-7,h-5);
  if(kind==='clarity'){ctx.setLineDash([3,5]);ctx.strokeStyle='#80938b';ctx.beginPath();ctx.moveTo(X(0),Y(.7));ctx.lineTo(X(maxTime),Y(.7));ctx.stroke();ctx.setLineDash([]);}
  data.forEach((d,arm)=>{
    const end=Math.min(Math.round(state.time/d.config.dt),d.mission.x.length-1);
    for(let region=0;region<(kind==='clarity'?2:1);region++){
      ctx.strokeStyle=CSS_COLORS[arm];ctx.lineWidth=1.5;ctx.setLineDash(region?[4,4]:[]);ctx.beginPath();
      for(let k=0;k<=end;k+=2){const value=kind==='clarity'?histories[arm][k][region]:summaries[arm].deficitHistory[k][1];
        if(value===null)continue;const x=X(k*d.config.dt),y=Y(value);if(k===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}
      ctx.stroke();
    }
  });ctx.setLineDash([]);ctx.strokeStyle='#d1e0e577';ctx.beginPath();ctx.moveTo(X(state.time),pad.t);ctx.lineTo(X(state.time),pad.t+ph);ctx.stroke();
}

function loop(now){
  const elapsed=lastFrame?Math.min((now-lastFrame)/1000,.15):0;
  if(now-lastFrame<1000/30){requestAnimationFrame(loop);return;}
  lastFrame=now;
  if(state.ready&&!document.hidden){
    if(state.playing){
      if(state.mode==='denoising'){
        state.levelTime+=elapsed*state.speed/4;
        if(state.levelTime>=1.25){state.levelTime=0;const last=data[0].planning[state.period].candidate_paths.length-1;
          if(state.level===last)state.playing=false;else state.level++;
          $('level').value=state.level;forecastKey='';refresh(true);}
      }else{state.time=Math.min(state.time+elapsed*state.speed,Number($('timeline').max));if(state.time===Number($('timeline').max))state.playing=false;}
    }
    refresh();
    scenes[activeCamera].render(elapsed,now,state.playing,state.follow,state.sensor,state.windFlow);
    if(!state.follow)scenes[1-activeCamera].syncCamera(scenes[activeCamera]);
    scenes[1-activeCamera].render(elapsed,now,state.playing,state.follow,state.sensor,state.windFlow);
  }
  requestAnimationFrame(loop);
}

document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.mode)));
$('play').addEventListener('click',()=>{
  if(!state.ready)return;
  if(state.mode==='final')setMode('execution');
  if(state.mode==='denoising'&&state.level===data[0].planning[state.period].candidate_paths.length-1){state.level=0;$('level').value=0;state.levelTime=0;}
  if(state.time>=Number($('timeline').max))state.time=0;
  state.playing=!state.playing;updateTransport();
});
$('restart').addEventListener('click',()=>{state.playing=false;state.level=0;$('level').value=0;state.levelTime=0;
  if(state.mode==='final')setMode('execution');if(state.mode!=='denoising')state.time=0;forecastKey='';refresh(true);});
$('speed').addEventListener('change',()=>state.speed=Number($('speed').value));
$('baseline').addEventListener('change',loadPair);
$('timeline').addEventListener('input',()=>{state.time=Number($('timeline').value);state.playing=false;
  if(state.mode==='final')setMode('execution');
  if(state.mode==='denoising'){state.period=Math.min(Math.floor(state.time/replanSeconds(data[0])),data[0].planning.length-1);state.time=state.period*replanSeconds(data[0]);$('period').value=state.period;}
  forecastKey='';refresh(true);
});
$('period').addEventListener('change',()=>{state.playing=false;state.period=Number($('period').value);state.time=state.period*replanSeconds(data[0]);state.level=0;$('level').value=0;state.levelTime=0;forecastKey='';refresh(true);});
$('level').addEventListener('input',()=>{state.playing=false;state.level=Number($('level').value);state.levelTime=0;forecastKey='';refresh(true);});
$('candidates').addEventListener('change',()=>{forecastKey='';refresh(true);});
document.querySelectorAll('[data-clarity-layer]').forEach(button=>button.addEventListener('click',()=>{
  state.clarityLayer=button.dataset.clarityLayer;
  document.querySelectorAll('[data-clarity-layer]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.clarityLayer===state.clarityLayer)));refresh(true);
}));
for(const id of ['layer','wind'])$(id).addEventListener('change',()=>{
  const scales={regions:'Region clarity 0 → 1 (overlay)',clarity:'Clarity 0 → 1 (overlay)',deficit:'Deficit 0 → 0.70',uncertainty:'POAM σ 0 → 1 m/s² (clipped)',none:'Red footprint: sensing strength · flow: physical wind direction'};
  const gradients={regions:'linear-gradient(90deg,#243443,#2d7d83,#67d9bb,#e4efa7)',clarity:'linear-gradient(90deg,#243443,#2d7d83,#67d9bb,#e4efa7)',deficit:'linear-gradient(90deg,#22313d,#9d5143,#f4bb6f)',uncertainty:'linear-gradient(90deg,#22313d,#5b68a9,#dcbdef)',none:'none'};
  $('scale-label').replaceChildren(document.createTextNode(scales[$('layer').value]+' '));const bar=document.createElement('i');bar.className='color-scale';bar.style.background=gradients[$('layer').value];$('scale-label').append(bar);refresh(true);
});
function cameraButtons(){
  $('follow').setAttribute('aria-pressed',String(state.follow));$('sensor').setAttribute('aria-pressed',String(state.sensor));scenes.forEach(s=>s.needsRender=true);updateURL();refresh(true);
}
for(const id of ['overview','valley','top'])$(id).addEventListener('click',()=>{state.follow=false;state.sensor=false;scenes.forEach(s=>s.setView(id));cameraButtons();});
$('follow').addEventListener('click',()=>{state.follow=!state.follow;state.sensor=false;cameraButtons();});
$('sensor').addEventListener('click',()=>{state.sensor=!state.sensor;state.follow=false;cameraButtons();});
$('flow').addEventListener('click',()=>{state.windFlow=!state.windFlow;$('flow').setAttribute('aria-pressed',String(state.windFlow));scenes.forEach(s=>s.needsRender=true);});
$('layout').addEventListener('change',()=>{
  const value=$('layout').value;document.querySelector('.flight-deck').className=`flight-deck ${value==='split'?'':`single-${value}`}`;
  activeCamera=value==='baseline'?1:0;scenes.forEach(s=>{s.resize();s.needsRender=true;});refresh(true);
});
$('fullscreen').addEventListener('click',async()=>{
  try{if(document.fullscreenElement)await document.exitFullscreen();else await $('flight-workspace').requestFullscreen();}
  catch(e){$('status').textContent='Full screen unavailable in this browser; the large view is still available.';}
});
document.addEventListener('fullscreenchange',()=>{$('fullscreen').textContent=document.fullscreenElement?'⛶ Exit full screen':'⛶ Full screen';scenes.forEach(s=>s.resize());refresh(true);});
window.addEventListener('resize',()=>{if(state.ready)refresh(true);});
window.addEventListener('pagehide',()=>scenes.forEach(s=>s.dispose()),{once:true});

async function start(){
  try{
    $('follow').setAttribute('aria-pressed',String(state.follow));
    $('sensor').setAttribute('aria-pressed',String(state.sensor));
    records=await getJSON('/api/catalog');record=records.find(r=>r.id===(query.get('run')||DEFAULT));
    if(record&&record.method!=='proposed')record=records.find(r=>r.group===record.group&&r.seed===record.seed&&r.method==='proposed');
    if(!record)throw new Error('The requested proposed mission is not available locally.');
    if(['current_belief','wind_blind_planner'].includes(query.get('baseline')))$('baseline').value=query.get('baseline');
    await loadPair();requestAnimationFrame(loop);
  }catch(e){error(e.message);console.error(e);}
}
start();
