import * as THREE from 'three';
import {createDrone} from './drone.js';
import {createFlightTrail} from './flight-trail.js?v=rrt-20261021-runtime';
const names={proposed:'Proposed',blind:'Disturbance-blind planning',blind_filter:'Blind planning + filter',aware:'Expected-cost planning',optimistic:'Optimism without archive mixture',fading:'Fading optimism',no_learning:'Proposed · learning disabled',proposed_crn:'Proposed · shared random draws'};
const colors={proposed:0x2378d3,proposed_crn:0x2378d3,blind:0xd66b31,blind_filter:0xd66b31};
const $=s=>document.querySelector(s);
const lastAt=(a,t)=>{let lo=0,hi=a.length-1;while(lo<hi){const m=Math.ceil((lo+hi)/2);if(a[m].t<=t)lo=m;else hi=m-1}return a[lo]};
export async function install(preview,initial){
 const rootManifest=await fetch(new URLSearchParams(location.search).has('archive')?'data/archive_manifest.json':'data/manifest.json',{cache:'no-store'}).then(r=>r.json());const study=(rootManifest.studies||[]).find(s=>s.id===new URLSearchParams(location.search).get('study'));const manifest=study?await fetch('data/'+study.manifest,{cache:'no-store'}).then(r=>r.json()):rootManifest;const assetPrefix=manifest.asset_prefix||'';preview.resultManifest=manifest;if(manifest.notice){const notice=document.createElement('div');notice.className='archive-notice';notice.textContent=manifest.notice;document.querySelector('.workspace').before(notice);document.body.classList.toggle('archived-recording',!!manifest.archived)}const analysis=manifest.analysis?await fetch('data/'+assetPrefix+manifest.analysis,{cache:'no-store'}).then(r=>r.json()):{runs:[]};
 const layout=$('#layout-select').value,available=manifest.runs.filter(r=>r.layout===layout);
 if(manifest.runs.length)for(const option of $('#layout-select').options){if(!manifest.runs.some(r=>r.layout===option.value)){option.disabled=true;option.textContent+=' · no recorded MBD flight'}}
 const box=document.createElement('div');box.className='replay-panel';box.innerHTML=`<div class="section-label">RECORDED FLIGHTS</div><label class="field-label" for="case-select">Wind environment</label><select id="case-select"><option value="forward">Forward · eastward wind in A</option><option value="reverse">Reversed · westward wind in A</option><option value="crosswind">Opposing and lateral flow in A/C</option></select><div id="run-status" class="fine">Loading recorded flights…</div><div id="variant-control" hidden><label class="field-label" for="proposed-variant">Proposed variant</label><select id="proposed-variant"><option value="proposed">Standard posterior sampling</option><option value="proposed_crn">Shared random draws · ablation</option></select></div><label class="field-label" for="map-method">Spatial maps for</label><select id="map-method"><option value="proposed">Proposed</option><option value="blind">Disturbance-blind planning</option></select><label class="switch-row"><span>Both flight trails</span><input id="trails" type="checkbox" checked></label><label class="field-label" for="trail-window">Trail history</label><select id="trail-window"><option value="45">Recent 45 seconds</option><option value="all">Entire recording</option></select><button class="text-button" id="results-toggle">Measured performance <span>＋</span></button><div id="results" hidden></div>`;
 $('.sidebar').insertBefore(box,$('.sidebar').children[2]);
 if(rootManifest.studies?.length){
  box.insertAdjacentHTML('afterbegin','<label class="field-label" for="study-select">Experiment</label><select id="study-select"></select>');
  $('#study-select').add(new Option('30 s horizon · 3.1 m/s wind',''));
  for(const item of rootManifest.studies)$('#study-select').add(new Option(item.title,item.id));
  $('#study-select').value=study?.id||'';
  $('#study-select').onchange=()=>{const url=new URL(location.href);url.searchParams.set('study',$('#study-select').value);location.assign(url)};
 }

 $('#map-method').previousElementSibling.insertAdjacentHTML('beforebegin','<label class="field-label" for="baseline-select">Compare proposed with</label><select id="baseline-select"></select>');
 $('#trail-window').insertAdjacentHTML('afterend','<label class="switch-row"><span>Show planned motion</span><input id="show-plan" type="checkbox"></label><p id="plan-caption" class="fine" hidden></p>');
 $('#maps-toggle').insertAdjacentHTML('beforebegin','<button id="performance-toggle">Performance</button>');
 $('#performance-toggle').onclick=()=>{document.body.classList.remove('controls-collapsed');$('#focus').textContent='Hide settings';$('#focus').setAttribute('aria-expanded','true');$('#results').hidden=false;$('#results-toggle span').textContent='−';$('#results-toggle').scrollIntoView({block:'nearest',behavior:'smooth'})};
 const transport=document.createElement('div');transport.className='transport';transport.hidden=true;transport.innerHTML=`<div class="follow-bar" aria-label="Flight camera"><button data-follow="proposed">Follow proposed</button><button data-follow="blind">Follow blind</button><button data-flight-overview>Overview</button><span class="follow-hint">Drag to orbit</span></div><div class="transport-top"><div class="method-key"><span class="proposed">Proposed</span><span class="blind">Disturbance-blind</span></div><span id="flight-status"></span></div><div class="transport-controls"><button id="play" aria-label="Play recorded simulation">Play</button><button id="rewind" aria-label="Restart recording">↶</button><input id="timeline" aria-label="Recorded simulation time" type="range" min="0" max="180" step="0.1" value="0"><output id="time-readout">0.0 s</output><select id="play-speed" aria-label="Playback speed"><option value="1">1×</option><option value="2">2×</option><option value="5" selected>5×</option><option value="10">10×</option></select></div><div class="transport-note" id="run-config"></div>`;
 $('#scene').append(transport);
 const flights=new THREE.Group();preview.scene.add(flights);const objects={};let activeProposed=new URLSearchParams(location.search).get('variant')||'proposed',activeBaseline=new URLSearchParams(location.search).get('baseline')||'blind';let runs={},playing=false,time=0,duration=0,lastNow=performance.now(),lastMap=-1,loading=0,followMode=null;const followOffset=new THREE.Vector3(-4.2,1.15,0);let followHeading=0;const followDesired=new THREE.Vector3(),followTarget=new THREE.Vector3();
 // Coordinate reflection maps physics (east,south,up) to Three (east,up,south).
 const basis=new THREE.Matrix4().set(1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1),rmat=new THREE.Matrix4();
 const plannedMotion=new THREE.Group();preview.scene.add(plannedMotion);let drawnPlan='';
 function drawPlan(method,run){
  plannedMotion.visible=$('#show-plan').checked&&!!run;
  $('#plan-caption').hidden=!plannedMotion.visible;
  if(!plannedMotion.visible)return;
  const plan=lastAt(run.plans,time),identifier=method+':'+run.case+':'+plan.t;
  if(identifier!==drawnPlan){
   for(const child of plannedMotion.children){child.geometry.dispose();child.material.dispose()}
   plannedMotion.clear();
   const samples=plan.points.map(p=>({p:[p[0],p[1],run.environment.flight_height]}));
   const line=createFlightTrail(samples,0x8a519e);line.material.uniforms.width.value=3.5;line.renderOrder=4;line.userData.showThrough(samples.length-1);plannedMotion.add(line);drawnPlan=identifier;
  }
  const belief=method==='blind'||method==='blind_filter'?'zero wind':'the learned mean wind';
  $('#plan-caption').textContent=`Purple: ${names[method]} prediction at ${plan.t.toFixed(0)} s, using ${belief}. Blue and orange remain recorded flight paths. Selected from ${plan.decision_origin}.`;
 }
 function object(method,run){
  const group=new THREE.Group(),drone=createDrone();drone.scale.setScalar(.55/.56);group.add(drone);group.matrixAutoUpdate=false;flights.add(group);
  const line=createFlightTrail(run.frames,colors[method]);flights.add(line);
  const footprint=new THREE.Mesh(new THREE.RingGeometry(.97,1,64),new THREE.MeshBasicMaterial({color:colors[method],transparent:true,opacity:.8,depthWrite:false,side:THREE.DoubleSide}));footprint.rotation.x=-Math.PI/2;flights.add(footprint);
  return {group,drone,line,footprint};
 }
 function draw(){
  const method=$('#map-method').value,selected=runs[method];
  drawPlan(method,selected);
  for(const [key,run] of Object.entries(runs)){
   if(!objects[key])continue;const obj=objects[key];if(key!==activeBaseline&&key!==activeProposed){for(const k of ['group','line','footprint'])obj[k].visible=false;continue;}const a=lastAt(run.frames,time),ix=run.frames.indexOf(a),b=run.frames[Math.min(ix+1,run.frames.length-1)],s=Math.max(0,Math.min(1,(time-a.t)/Math.max(b.t-a.t,1e-9)));
   const position=new THREE.Vector3(...a.p).lerp(new THREE.Vector3(...b.p),s),quat=new THREE.Quaternion(...a.quat).slerp(new THREE.Quaternion(...b.quat),s);
   rmat.compose(position,quat,new THREE.Vector3(1,1,1));obj.group.matrix.multiplyMatrices(basis,rmat);obj.group.matrixWorldNeedsUpdate=true;
   obj.drone.userData.rotors.forEach((r,i)=>r.rotation.z=time*(i%2?-65:65));obj.line.userData.showThrough(ix,$('#trail-window').value==='all'?0:Math.max(0,ix-450));obj.line.visible=$('#trails').checked;obj.footprint.position.set(position.x,.21,position.y);obj.footprint.scale.setScalar(position.z);obj.group.visible=!(run.failure&&time>run.completed_seconds);obj.footprint.visible=$('#sensor').checked&&obj.group.visible;
  }

  if(selected&&(Math.abs(time-lastMap)>.15||(time===duration&&lastMap!==time))){const f=lastAt(selected.frames,time),belief=lastAt(selected.beliefs,time);preview.mapState={...initial,environment:selected.environment,q:selected.failure&&time>selected.completed_seconds?f.q.map((q,i)=>q/(1+(selected.environment.process_noise||initial.environment.process_noise)[i]*q*(time-selected.completed_seconds))):f.q,position:selected.failure&&time>selected.completed_seconds?null:f.p,mean:belief.mean,std:belief.std,t:time,beliefTime:belief.t,activeMethod:names[method]+(selected.failure&&time>selected.completed_seconds?' · sensing stopped':'')};preview.drawMaps();lastMap=time;}
  $('#timeline').value=time;$('#time-readout').textContent=time.toFixed(1)+' / '+duration.toFixed(0)+' s';
  $('#flight-status').textContent=Object.entries(runs).filter(([k,r])=>(k===activeBaseline||k===activeProposed)&&r.failure&&time>=r.completed_seconds).map(([k,r])=>names[k]+': '+r.failure).join(' · ');
 }
 function physicalField(env){preview.windAt=(x,z)=>{const [rows,cols]=env.grid_shape,a=env.grid[0],b=env.grid.at(-1),u=Math.max(0,Math.min(cols-1,(x-a[0])/(b[0]-a[0])*(cols-1))),v=Math.max(0,Math.min(rows-1,(z-a[1])/(b[1]-a[1])*(rows-1))),i=Math.floor(u),j=Math.floor(v),answer=[0,0];for(const [dx,wx]of[[0,1-u+i],[1,u-i]])for(const[dz,wz]of[[0,1-v+j],[1,v-j]]){const w=env.wind[Math.min(j+dz,rows-1)*cols+Math.min(i+dx,cols-1)];answer[0]+=wx*wz*w[0];answer[1]+=wx*wz*w[1]}return answer};$('.wind-badge').textContent='Physical wind · peak '+Math.max(...env.wind.map(w=>Math.hypot(...w))).toFixed(1)+' m/s';$('#wind-pattern').value=$('#case-select').value;$('#wind-pattern').disabled=true;$('#wind-pattern').previousElementSibling.textContent='Simulation wind field';$('#wind-controls .fine').textContent='Particles follow the spatial air-velocity field used by the simulation.';}
 async function load(){
  const id=++loading,kind=$('#case-select').value;playing=false;$('#play').textContent='Play';$('#play').setAttribute('aria-label','Play recorded simulation');time=0;lastMap=-1;transport.hidden=true;$('#run-status').textContent='Loading recorded flights…';
  plannedMotion.visible=false;drawnPlan='';
  flights.clear();Object.values(objects).forEach(o=>Object.values(o).forEach(x=>{x.geometry?.dispose();x.material?.dispose()}));for(const k in objects)delete objects[k];runs={};
  const matching=available.filter(r=>r.case===kind),loaded=await Promise.all(matching.map(async r=>[r.method,await fetch('data/'+assetPrefix+r.file,{cache:'no-store'}).then(v=>v.json())]));if(id!==loading)return;
  runs=Object.fromEntries(loaded);preview.runs=runs;
  const variants=['proposed','proposed_crn'].filter(k=>runs[k]);if(!runs[activeProposed])activeProposed=variants[0]||'proposed';$('#variant-control').hidden=variants.length<2;$('#proposed-variant').value=activeProposed;const baselines=['blind','blind_filter'].filter(k=>runs[k]);if(!runs[activeBaseline])activeBaseline=baselines[0]||'blind';$('#baseline-select').innerHTML=baselines.map(k=>`<option value="${k}">${names[k]}</option>`).join('');$('#baseline-select').value=activeBaseline;const primary=[activeProposed,activeBaseline].filter(k=>runs[k]);preview.drone.visible=preview.sensor.visible=!primary.length;preview.recordedPlayback=!!primary.length;
  if(!primary.length){$('#run-status').textContent='No completed flight recording for this selection yet. Maps show the initial condition.';const init=await fetch(`data/initial_${layout}_${kind}.json`,{cache:'no-store'}).then(r=>r.json());preview.mapState={...init,t:0,activeMethod:'Initial condition'};physicalField(init.environment);preview.drawMaps();return;}
  for(const key of Object.keys(runs).filter(k=>colors[k]))objects[key]=object(key,runs[key]);
  document.querySelectorAll('[data-follow]').forEach(button=>{button.hidden=button.dataset.follow==='proposed'?!runs[activeProposed]:!runs[activeBaseline]});
  $('#map-method').innerHTML=primary.map(k=>`<option value="${k}">${names[k]}</option>`).join('');$('#map-method').value=primary.includes(activeProposed)?activeProposed:primary[0];
  for(const key of ['proposed','blind'])document.querySelector('.method-key .'+key).hidden=!(key==='proposed'?runs[activeProposed]:runs[activeBaseline]);const r=runs[primary[0]];duration=Math.max(...primary.map(k=>runs[k].mission_seconds));$('#timeline').max=duration;transport.hidden=false;
  $('#run-status').textContent=`${primary.length===2?'Both primary methods':names[primary[0]]} · ${r.mission_seconds} s mission · seed ${r.seed}. ${matching.some(x=>x.development)?'Development run.':''}`;
  $('.header-note').textContent='Recorded nonlinear flights';
  transport.title=`${r.config.horizon} s planning horizon · ${r.config.dt} s planning step · 0.01 s physics · ${r.config.iterations} × ${r.config.proposals} MBD · cosine · temperature ${r.config.temperature}`;
  $('#run-config').textContent=`${r.config.initialization==='rrt'?'RRT-guided MBD':'MBD'} · ${r.config.horizon} s horizon · ${r.config.iterations} × ${r.config.proposals} · dt ${r.config.dt} s · replan every ${r.config.commit} s`;
  const blind=runs.blind,valid=!manifest.archived&&blind?.completed&&!blind?.plan_only;
  updateComparison();
  $('#results').innerHTML=`<p class="fine">Time-integrated mean clarity deficit. Lower is better. These are individual simulation runs.</p><p class="fine">Both blind baselines estimate wind online for execution, but assume zero wind when planning. “Blind planning + filter” uses the same execution filter as proposed.</p><p class="fine">Planning horizon: ${r.config.horizon} s; planning step: ${r.config.dt} s; nonlinear physics: ${r.model.execution_step} s.<br>Decision coordinates: ${r.model.decision_dimension}; ${r.plans[0].evaluations} optimization evaluations per call, plus independent selection.<br>Replanning: ${r.config.commit} s. Optimization / selection fields: ${r.config.worlds} / ${r.config.validation_worlds}. Proposed favorable fraction: ${r.config.optimism}.</p><table><thead><tr><th>Method</th><th>Deficit·s</th><th>vs blind</th></tr></thead><tbody>${Object.entries(runs).map(([k,v])=>`<tr><td>${names[k]}</td><td>${v.cost.toFixed(2)}</td><td>${valid&&v.completed&&v.mission_seconds===blind.mission_seconds?((blind.cost-v.cost)/blind.cost*100).toFixed(1)+'%':'—'}</td></tr>`).join('')}</tbody></table><p class="fine">Positive percentage = less deficit. Failures retain the unsensed remainder of the mission in their cost.</p>${Object.entries(runs).map(([k,v])=>{const times=v.plans.filter((p,i)=>i>=2&&!p.compilation_included&&!v.plans[i-1].compilation_included).map(p=>p.total_planning_seconds??p.seconds).sort((a,b)=>a-b);return `<div class="compute-row"><b>${names[k]}</b><br>${v.success?'Completed without detected contact':v.failure}<br>Median planning: ${times.length?((times[Math.floor((times.length-1)/2)]+times[Math.floor(times.length/2)])/2).toFixed(2):'—'} s (compile warm-up excluded)<br>Tracking RMS: ${v.rms_tracking_error.toFixed(2)} m</div>`}).join('')}<p class="fine">Planning computation is outside the simulation clock. This is not a real-time demonstration.</p>`;
  if(manifest.timing)$('#results').insertAdjacentHTML('beforeend','<p class="fine"><a href="data/'+assetPrefix+manifest.timing+'" target="_blank" rel="noopener">Timing benchmark and reproduction commands</a> · Compilation and fixed-state validation reported separately.</p>');
  if(r.config.initialization==='rrt')$('#results').insertAdjacentHTML('beforeend','<p class="fine"><a href="data/'+assetPrefix+'execution_integrity.json" target="_blank" rel="noopener">Flight reconstruction audit</a> · Positions, attitudes and mission costs checked against the 100 Hz simulation records.</p>');
  const evidence=analysis.runs.filter(r=>r.layout===layout&&r.case===kind&&colors[r.method]);if(evidence.length){
   if(r.mission_seconds>60)$('#results').insertAdjacentHTML('beforeend','<div class="section-label">EARLY AND LATER PERFORMANCE</div><p class="fine">Mean clarity deficit in each time window; lower is better.</p><table><thead><tr><th>Method</th><th>0–60 s</th><th>60–'+r.mission_seconds+' s</th></tr></thead><tbody>'+evidence.map(e=>'<tr><td>'+names[e.method]+'</td><td>'+(e.first_minute_cost/60).toFixed(3)+'</td><td>'+e.later_mean_deficit.toFixed(3)+'</td></tr>').join('')+'</tbody></table>');
$('#results').insertAdjacentHTML('beforeend','<div class="section-label">OBSERVED BEHAVIOR</div><p class="fine">'+(kind==='crosswind'?'A and C contain opposing and lateral flow; B is calm. Monitoring four sites requires repeated transfers.':'A contains an axial wind; B and C are calm. A longer sheltered route can leave more time for inspection.')+'</p>'+evidence.map(r=>`<div class="compute-row"><b>${names[r.method]}</b><br>${r.transfers.length} completed cross-district transfers.<br><details><summary>Passage durations</summary>${r.transfers.map(t=>`Passage ${t.passage} ${t.direction}: ${t.seconds.toFixed(1)} s`).join('<br>')}</details><br>Inspection time: ${r.regions.map(g=>g.name+' '+g.seconds.toFixed(0)+' s').join('; ')}.<br>Mean wind std: 1.80 → ${r.final_mean_wind_std.toFixed(2)} m/s.<br>Minimum surface clearance: ${r.minimum_clearance.toFixed(3)} m.${r.mean_position_prediction_error!=null?'<br>Mean '+r.prediction_error_interval.toFixed(0)+' s position prediction error: '+r.mean_position_prediction_error.toFixed(2)+' m.':''}<br>First-minute deficit integral: ${r.first_minute_cost.toFixed(2)}.<br>Mean deficit over the remaining ${Math.max(0,r.mission_seconds-60).toFixed(0)} s: ${r.later_mean_deficit.toFixed(3)}.${r.late_deficit_reduction_percent!=null?'<br>Later deficit vs blind: '+r.late_deficit_reduction_percent.toFixed(1)+'% reduction.':''}</div>`).join(''));}
  $('#wind-pattern').value=kind;$('#wind-pattern').disabled=true;$('#wind-pattern').previousElementSibling.textContent='Recorded physical field';$('#wind-controls .fine').textContent='Particle motion follows the same synthetic spatial wind field used by the flight dynamics. The planner receives only its learned belief.';$('.wind-badge').textContent='Physical wind · peak '+Math.max(...r.environment.wind.map(w=>Math.hypot(...w))).toFixed(1)+' m/s';$('#wind').closest('label').querySelector('span').textContent='Physical wind particles';
  preview.windAt=(x,z)=>{const e=r.environment,cols=e.grid_shape[1],rows=e.grid_shape[0],a=e.grid[0],last=e.grid.at(-1),ix=Math.max(0,Math.min(cols-1,(x-a[0])/(last[0]-a[0])*(cols-1))),iz=Math.max(0,Math.min(rows-1,(z-a[1])/(last[1]-a[1])*(rows-1))),i=Math.floor(ix),j=Math.floor(iz),u=ix-i,v=iz-j;const out=[0,0];for(const [dx,wx]of [[0,1-u],[1,u]])for(const[dz,wz]of[[0,1-v],[1,v]]){const w=e.wind[Math.min(j+dz,rows-1)*cols+Math.min(i+dx,cols-1)];out[0]+=wx*wz*w[0];out[1]+=wx*wz*w[1]}return out};
  $('#wind').dispatchEvent(new Event('change'));draw();follow(runs[activeProposed]?'proposed':'blind',true);
 }

 function follow(key,instant=false){
  followMode=key;preview.following=!!key;preview.cancelCameraTransition();
  document.querySelectorAll('[data-follow]').forEach(b=>{const active=b.dataset.follow===key;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active))});
  $('[data-flight-overview]').classList.toggle('active',!key);
  if(key){$('#view-name').textContent=key==='blind'?'Following · '+names[activeBaseline]:'Following · proposed';$('#view-caption').textContent='Perspective chase camera · follows the drone through the streets';preview.controls.minDistance=.45;updateFollow(1)}
 }
 function updateFollow(factor=.18){
  if(!followMode)return;
  const key=followMode==='proposed'?activeProposed:activeBaseline,run=runs[key];
  const obj=objects[key];
  if(obj&&run){const a=lastAt(run.frames,time),i=run.frames.indexOf(a),b=run.frames[Math.min(i+1,run.frames.length-1)],u=Math.min(1,Math.max(0,(time-a.t)/Math.max(.001,b.t-a.t)));followTarget.set(a.p[0]+u*(b.p[0]-a.p[0]),a.p[2]+u*(b.p[2]-a.p[2]),a.p[1]+u*(b.p[1]-a.p[1]));}
  else followTarget.copy(preview.drone.position);
  // Third-person chase view: remain behind the measured motion at a low
  // elevation. Near a wall, shorten the camera boom; never switch overhead.
  if(run){
   const earlier=lastAt(run.frames,Math.max(0,time-.6)),current=lastAt(run.frames,time);
   let vx=current.p[0]-earlier.p[0],vz=current.p[1]-earlier.p[1];
   if(Math.hypot(vx,vz)<.025){vx=current.velocity[0];vz=current.velocity[1]}
   if(Math.hypot(vx,vz)>.04){
    const desired=Math.atan2(vz,vx),delta=Math.atan2(Math.sin(desired-followHeading),Math.cos(desired-followHeading));
    followHeading+=delta*(factor===1?1:.08);
   }else if(time===0)followHeading=0;
  }
  const blocked=offset=>preview.world.buildings.some(b=>{
   const c=Math.cos(b.rotation),s=Math.sin(b.rotation),steps=Math.max(2,Math.ceil(Math.hypot(...offset)/.08));
   for(let j=1;j<=steps;j++){
    const t=j/steps,x=followTarget.x+offset[0]*t-b.x,z=followTarget.z+offset[2]*t-b.z,y=followTarget.y+offset[1]*t;
    if(y>b.totalHeight+.12)continue;
    if(Math.abs(x*c-z*s)<b.width/2+.12&&Math.abs(x*s+z*c)<b.depth/2+.12)return true;
   }return false;
  });
  followOffset.set(-4.2*Math.cos(followHeading),1.15,-4.2*Math.sin(followHeading));
  // Restrict distance, not viewing angle, to keep perspective in narrow turns.
  while(blocked(followOffset.toArray())&&followOffset.length()>.55)followOffset.multiplyScalar(.88);
  followDesired.copy(followTarget).add(followOffset);
  if(factor!==1){
   followDesired.lerpVectors(preview.camera.position,followDesired,factor);
   const offset=followDesired.clone().sub(followTarget);
   // A smoothed camera can cut across a building corner. Use the clear boom
   // immediately there, rather than showing the inside of a wall.
   if(blocked(offset.toArray()))followDesired.copy(followTarget).add(followOffset);
  }
  preview.camera.position.copy(followDesired);preview.controls.target.copy(followTarget);
  preview.controls.update();
 }
 const resetCamera=$('#reset').onclick;$('#reset').onclick=()=>followMode?follow(followMode,true):resetCamera();
 document.querySelectorAll('[data-follow]').forEach(b=>b.onclick=()=>follow(b.dataset.follow));
 $('[data-flight-overview]').onclick=()=>{follow(null);preview.controls.minDistance=1.1;preview.setCamera('overview')};
 function updateComparison(){
  const baseline=runs[activeBaseline],proposed=runs[activeProposed];$('.method-key .blind').textContent=names[activeBaseline];document.querySelector('[data-follow="blind"]').textContent=activeBaseline==='blind_filter'?'Follow blind + filter':'Follow blind';
  const label=activeProposed==='proposed_crn'?'Proposed · shared draws':'Proposed';
  if(!manifest.archived&&proposed?.completed&&baseline?.completed&&proposed.mission_seconds===baseline.mission_seconds){const gain=100*(baseline.cost-proposed.cost)/baseline.cost;$('.method-key .proposed').textContent=`Full mission: ${label} · ${Math.abs(gain).toFixed(1)}% ${gain>=0?'less':'more'} deficit`;$('.method-key').title='Complete mission comparison, independent of playback position';}else $('.method-key .proposed').textContent=label;
 }
 $('#proposed-variant').onchange=()=>{
  activeProposed=$('#proposed-variant').value;const previous=$('#map-method').value;
  $('#map-method').innerHTML=[activeProposed,activeBaseline].filter(k=>runs[k]).map(k=>`<option value="${k}">${names[k]}</option>`).join('');
  $('#map-method').value=previous===activeBaseline?activeBaseline:activeProposed;
  const url=new URL(location.href);url.searchParams.set('variant',activeProposed);history.replaceState(null,'',url);
  lastMap=-1;updateComparison();draw();
 };
 $('#baseline-select').onchange=()=>{
  activeBaseline=$('#baseline-select').value;
  $('#map-method').innerHTML=[activeProposed,activeBaseline].filter(k=>runs[k]).map(k=>`<option value="${k}">${names[k]}</option>`).join('');
  const url=new URL(location.href);url.searchParams.set('baseline',activeBaseline);history.replaceState(null,'',url);
  lastMap=-1;updateComparison();draw();if(followMode)follow(followMode,true);
 };
 $('#show-plan').onchange=draw;
 $('#results-toggle').onclick=()=>{$('#results').hidden=!$('#results').hidden;$('#results-toggle span').textContent=$('#results').hidden?'＋':'−'};
 $('#play').onclick=()=>{playing=!playing;if(time>=duration)time=0;lastNow=performance.now();$('#play').textContent=playing?'Pause':'Play';$('#play').setAttribute('aria-label',playing?'Pause recorded simulation':'Play recorded simulation')};
 $('#rewind').onclick=()=>{time=0;lastMap=-1;draw()};$('#timeline').oninput=()=>{time=Number($('#timeline').value);lastMap=-1;draw();updateFollow(1)};$('#map-method').onchange=()=>{lastMap=-1;draw()};$('#trails').onchange=draw;$('#trail-window').onchange=draw;preview.controls.addEventListener('start',()=>follow(null));document.querySelectorAll('[data-camera]').forEach(b=>b.addEventListener('click',()=>{followMode=null;preview.following=false;document.querySelectorAll('[data-follow]').forEach(x=>{x.classList.remove('active');x.setAttribute('aria-pressed','false')})}));
 $('#sensor').addEventListener('change',()=>{if(preview.recordedPlayback){preview.drone.visible=preview.sensor.visible=false;draw()}});
 const requested=new URLSearchParams(location.search).get('case')||'forward',recordedCases=[...new Set(available.map(r=>r.case))];
 const caseNames={forward:'Forward · eastward wind in A',reverse:'Reversed · westward wind in A',crosswind:'Opposing and lateral flow in A/C'};
 if(recordedCases.length)$('#case-select').innerHTML=recordedCases.map(k=>`<option value="${k}">${caseNames[k]}</option>`).join('');
 $('#case-select').value=recordedCases.includes(requested)?requested:recordedCases[0]||requested;
 const canonical=new URL(location.href);canonical.searchParams.set('case',$('#case-select').value);history.replaceState(null,'',canonical);
 $('#case-select').onchange=()=>{const url=new URL(location.href);url.searchParams.set('case',$('#case-select').value);history.replaceState(null,'',url);load()};
 preview.seek=t=>{time=Math.max(0,Math.min(duration,t));lastMap=-1;draw()};preview.playback=()=>({time,duration,playing});
 preview.tick=now=>{const delta=Math.min(.2,(now-lastNow)/1000);lastNow=now;if(playing){time=Math.min(duration,time+delta*Number($('#play-speed').value));if(time>=duration){playing=false;$('#play').textContent='Play';$('#play').setAttribute('aria-label','Play recorded simulation')}draw()}updateFollow()};
 await load();
}
