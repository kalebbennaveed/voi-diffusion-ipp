import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {createDrone} from './models.js';
import {values as plotValues,drawBelief,crossings} from './diagnostics.js';
import {buildEnvironment,createSensor,updateSensor,createFlow,updateFlow} from './environment.js';
const $=s=>document.querySelector(s), COLORS={proposed:'#71d5b3',aware:'#79b7ed',blind:'#ecaa85'};
const PATH_COLORS={proposed:'#087d61',aware:'#176bb8',blind:'#b64d29'};
const names={proposed:'Proposed',aware:'Disturbance-aware',blind:'Disturbance-blind'};
const descriptions={forward:['1a · Forward wind in passage A','A 3.8 m/s eastward wind fills narrow passage A. The planner can use different passages outward and back; the sensing footprint shows actual coverage.'],reverse:['1b · Same environment, reversed wind','The entire disturbance vector in A is reversed: the wind now blows westward. Geometry, initial conditions and the three controllers are unchanged.'],crosswind:['2 · A sheltered alternative','Passage B has zero wind velocity. Headwind and opposing lateral wind in narrow A and C reduce controllable ground speed; B remains calm.'],commitment:['3 · Is the crossing worth its cost?','The right inspection region is larger. Strong eastward wind makes a return costly; the monitoring objective decides whether crossing pays off.']};
let runs={},views=[],t=0,playing=false,last=0,follow=false,selected='forward',duration=360;
let manifest;let dirty=true,lastRender=0,loadSequence=0;
function material(color,extra={}){return new THREE.MeshStandardMaterial({color,roughness:.82,...extra})}
function box(scene,size,pos,color,extra={}){const m=new THREE.Mesh(new THREE.BoxGeometry(...size),material(color,extra));m.position.set(...pos);m.castShadow=true;m.receiveShadow=true;scene.add(m);return m;}
function textSprite(text,color='#183746',scale=2){const c=document.createElement('canvas');c.width=512;c.height=128;const a=c.getContext('2d');a.font='600 78px system-ui';a.textAlign='center';a.textBaseline='middle';if(['A','B','C'].includes(text)){a.fillStyle='#153247';a.beginPath();a.roundRect(197,10,118,108,18);a.fill();a.fillStyle='#ffffff'}else{a.fillStyle=color}a.fillText(text,256,64);const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),depthTest:false,transparent:true,fog:false,toneMapped:false}));s.scale.set(scale*4,scale,1);s.renderOrder=9;return s;}
function line(scene,points,color,dashed=false){const g=new THREE.BufferGeometry().setFromPoints(points.map(p=>new THREE.Vector3(...p)));const m=dashed?new THREE.LineDashedMaterial({color,dashSize:.5,gapSize:.3,depthTest:true}):new THREE.LineBasicMaterial({color,fog:false,toneMapped:false});const l=new THREE.Line(g,m);if(dashed)l.computeLineDistances();scene.add(l);return l;}
const c0=new THREE.Color('#733957'),c1=new THREE.Color('#ffbc77'),c2=new THREE.Color('#bde6ad'),c3=new THREE.Color('#369a95');
function clarityColor(q){if(q<.33)return c0.clone().lerp(c1,q/.33);if(q<.67)return c1.clone().lerp(c2,(q-.33)/.34);return c2.clone().lerp(c3,(q-.67)/.33)}
function uncertaintyColor(s){return new THREE.Color('#1f6374').lerp(new THREE.Color('#f3ca7b'),Math.min(s/1.8,1))}
function makeView(method,run){
  const card=document.createElement('article');card.className='method';card.style.setProperty('--method',COLORS[method]);
  const base=runs.blind?.cost, gain=base?100*(base-run.cost)/base:0;
  card.innerHTML=`<div class="method-head"><div><h3>${names[method]}</h3><small>${method==='proposed'?'Anticipates future observations':method==='aware'?'Plans with the current wind belief':'Plans without a disturbance field'}</small></div><div class="card-actions"><span class="chip">${run.success?'COMPLETED':'FAILED'}</span><button class="enlarge" aria-label="Enlarge ${names[method]} view">⤢</button></div></div><div class="viewport"><span class="corner">3D FLIGHT · METRES</span><span class="inset-label">ACTUAL ATTITUDE</span><div class="failure-overlay" hidden></div></div><div class="metrics"><div><strong>${run.cost.toFixed(2)}</strong><small>MISSION COST ↓</small></div><div><strong class="gain ${method==='blind'?'neutral':gain>=0?'improved':'worse'}">${method==='blind'?'baseline':!base?'pending':`${gain>=0?'−':'+'}${Math.abs(gain).toFixed(1)}%`}</strong><small>VS. BLIND COST</small></div><div><strong>${run.rms_tracking_error.toFixed(2)} m</strong><small>TRACKING RMS</small></div></div><div class="live"><span class="live-cost">Cost: 0.00</span><span class="live-std">Wind σ: 1.80 m/s</span></div>`;
  $('#comparison').append(card);card.querySelector('.enlarge').onclick=()=>{const expanded=card.classList.toggle('is-large');$('#comparison').classList.toggle('has-large',expanded);card.querySelector('.enlarge').textContent=expanded?'Compare all three':'⤢';card.querySelector('.enlarge').setAttribute('aria-label',expanded?'Compare all three':`Enlarge ${names[method]} view`)};const host=card.querySelector('.viewport');
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=.95;host.prepend(renderer.domElement);
  const scene=new THREE.Scene();scene.background=new THREE.Color('#a9b7c0');scene.fog=new THREE.Fog('#a9b7c0',120,220);
  scene.add(new THREE.HemisphereLight(0xc7d5e1,0x5d6462,.9));const sun=new THREE.DirectionalLight(0xfff1df,2.7);sun.position.set(-5,-15,45);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);Object.assign(sun.shadow.camera,{left:-35,right:35,top:35,bottom:-35,near:1,far:100});sun.shadow.bias=-.0005;scene.add(sun);
  buildEnvironment(scene,run.environment);
  // Batch repeated building details without changing their visual geometry.
  const groups=new Map();for(const mesh of [...scene.children]){if(!mesh.isMesh||mesh.geometry.type!=='BoxGeometry')continue;const p=mesh.geometry.parameters,key=[p.width,p.height,p.depth,mesh.material.color.getHex(),mesh.material.roughness,mesh.material.metalness,mesh.material.map?.uuid].join('/');if(!groups.has(key))groups.set(key,[]);groups.get(key).push(mesh)}
  for(const group of groups.values()){if(group.length<4)continue;const first=group[0],batch=new THREE.InstancedMesh(first.geometry,first.material,group.length);group.forEach((m,i)=>{m.updateMatrix();batch.setMatrixAt(i,m.matrix);scene.remove(m);if(i){m.geometry.dispose();if(m.material!==first.material)m.material.dispose()}});batch.castShadow=true;batch.receiveShadow=true;scene.add(batch)}
  const camera=new THREE.OrthographicCamera(-25,25,18,-18,.05,180);camera.up.set(0,0,1);camera.position.set(20,7,80);
  const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(20,14,1);controls.minDistance=3;controls.maxDistance=95;controls.minZoom=.65;controls.maxZoom=8;controls.maxPolarAngle=Math.PI*.47;controls.enableDamping=true;controls.addEventListener('change',()=>{dirty=true});controls.update();
  const inset=new THREE.PerspectiveCamera(40,1,.01,150);inset.up.set(0,0,1);
  const drone=createDrone();drone.scale.setScalar(.08/.17);scene.add(drone);
  const ring=new THREE.Mesh(new THREE.TorusGeometry(.70,.07,8,40),new THREE.MeshBasicMaterial({color:PATH_COLORS[method],fog:false,toneMapped:false}));scene.add(ring);
  const cellGeometry=new THREE.PlaneGeometry(.84,.84);const cells=run.environment.cells.map(p=>{const m=new THREE.Mesh(cellGeometry,new THREE.MeshBasicMaterial({color:clarityColor(.35),transparent:true,opacity:.9,fog:false,toneMapped:false}));m.position.set(p[0],p[1],.09);scene.add(m);return m});
  const uncertainty=[];for(const p of run.environment.grid){if(run.environment.buildings.some(r=>p[0]>r[0]-.2&&p[0]<r[2]+.2&&p[1]>r[1]-.2&&p[1]<r[3]+.2))continue;const g=new THREE.PlaneGeometry(1.9,1.9);const m=new THREE.Mesh(g,new THREE.MeshBasicMaterial({color:0xf3ca7b,transparent:true,opacity:.65,depthWrite:false,fog:false,toneMapped:false}));m.position.set(p[0],p[1],.12);m.userData.gridIndex=run.environment.grid.indexOf(p);scene.add(m);uncertainty.push(m)}
  const flow=createFlow(scene,run.environment),sensor=createSensor(scene);
  const arrows=new THREE.Group();scene.add(arrows);
  const points=run.frames.map(f=>new THREE.Vector3(...f.p));const trail=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:PATH_COLORS[method],fog:false,toneMapped:false}));scene.add(trail);
  let refline=line(scene,[[6,6,3],[6,6,3]],0xffffff,true);
  const resize=()=>{if(!host.clientWidth||!host.clientHeight)return;renderer.setSize(host.clientWidth,host.clientHeight,false);const aspect=host.clientWidth/host.clientHeight,half=Math.max(18,23/aspect);camera.left=-half*aspect;camera.right=half*aspect;camera.top=half;camera.bottom=-half;camera.updateProjectionMatrix();dirty=true};const observer=new ResizeObserver(resize);observer.observe(host);resize();
  return {method,run,card,host,renderer,scene,camera,controls,inset,drone,ring,cells,uncertainty,arrows,trail,refline,lastPlan:-1,observer,flow,sensor};
}
function indexAt(array,time){let lo=0,hi=array.length-1;while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(array[mid].t<=time)lo=mid;else hi=mid-1}return lo}
function cameraOffset(position,env,close=false){
 const offsets=close?[[.55,-.75,.45],[.55,.75,.45],[-.55,-.75,.45],[-.55,.75,.45],[.03,.02,1]]:[[4,-3,6],[4,3,6],[-4,-3,6],[-4,3,6],[3,-.3,4],[3,.3,4],[-3,-.3,4],[-3,.3,4],[.01,0,6]];
 for(const offset of offsets){let blocked=false;for(let k=1;k<=12;k++){
  const f=k/12,x=position.x+offset[0]*f,y=position.y+offset[1]*f,z=position.z+offset[2]*f;
  if(env.buildings.some(r=>x>r[0]-.12&&x<r[2]+.12&&y>r[1]-.12&&y<r[3]+.12&&z<r[4]+.4)){blocked=true;break}
 }if(!blocked)return new THREE.Vector3(...offset)}
 return new THREE.Vector3(.01,0,6);
}
function updateView(v){
 if(!v.host.clientWidth||!v.host.clientHeight)return;
 const {run,drone,renderer,scene,camera,controls,inset,host}=v;const i=indexAt(run.frames,t),f=run.frames[i],g=run.frames[Math.min(i+1,run.frames.length-1)],fraction=Math.max(0,Math.min(1,(t-f.t)/(g.t-f.t||1)));
 const stopped=!run.success&&t>run.completed_seconds;drone.visible=!stopped;v.ring.visible=!stopped;const overlay=v.card.querySelector('.failure-overlay');overlay.hidden=!stopped;overlay.textContent=`Execution stopped at ${run.completed_seconds?.toFixed(1)} s: ${run.failure}. Remaining mission cost assumes no sensing.`;
 drone.position.fromArray(f.p).lerp(new THREE.Vector3(...g.p),fraction);drone.quaternion.fromArray(f.quat).slerp(new THREE.Quaternion(...g.quat),fraction);
 drone.userData.rotors.forEach((r,k)=>r.rotation.z=t*f.motors[k]*(k%2?1:-1));
 v.ring.position.copy(drone.position);v.ring.position.z=drone.position.z+.12;v.trail.geometry.setDrawRange(0,i+1);v.cells.forEach((m,j)=>m.material.color.copy(clarityColor(f.q[j])));
 const b=run.beliefs[indexAt(run.beliefs,t)];v.uncertainty.forEach(m=>{m.visible=$('#uncertainty').checked;m.material.color.copy(uncertaintyColor(b.std[m.userData.gridIndex]))});const estimated=$('#flow-source').value==='mean';updateFlow(v.flow,estimated?b.mean:run.environment.wind,estimated?b.t:'truth',t,$('#wind-toggle').checked);updateSensor(v.sensor,drone.position,$('#footprint').checked&&!stopped);v.cells.forEach(m=>m.visible=$('#clarity-toggle').checked);
 v.card.querySelector('.live-cost').textContent=stopped?`Run stopped · mission cost including no-sensing remainder: ${run.cost.toFixed(2)}`:`Cost: ${f.cost.toFixed(2)}`;v.card.querySelector('.live-std').textContent=`Actual wind ${Math.hypot(...f.wind).toFixed(1)} m/s · ground ${Math.hypot(...f.velocity.slice(0,2)).toFixed(2)} m/s · map σ ${(b.std.reduce((a,x)=>a+x,0)/b.std.length).toFixed(2)} m/s`;
 const planIndex=indexAt(run.plans,t);if(planIndex!==v.lastPlan){v.lastPlan=planIndex;v.refline.geometry.dispose();v.refline.geometry=new THREE.BufferGeometry().setFromPoints(run.plans[planIndex].points.map(p=>new THREE.Vector3(p[0],p[1],3.04)));v.refline.computeLineDistances()}
 if(follow){camera.position.copy(drone.position).add(cameraOffset(drone.position,run.environment));controls.target.copy(drone.position)}controls.update();
 renderer.setScissorTest(false);renderer.setViewport(0,0,host.clientWidth,host.clientHeight);renderer.render(scene,camera);
 v.card.querySelector('.inset-label').hidden=stopped||!$('#attitude').checked;if(stopped||!$('#attitude').checked)return;
 const s=Math.min(135,host.clientWidth*.32),x=host.clientWidth-s-7,y=7;inset.position.copy(drone.position).add(cameraOffset(drone.position,run.environment,true));inset.lookAt(drone.position);renderer.clearDepth();renderer.setScissorTest(true);renderer.setScissor(x,y,s,s);renderer.setViewport(x,y,s,s);const overlays=[v.ring,v.arrows,v.trail,v.refline,v.flow.lines,v.sensor.group,v.sensor.rays,...scene.children.filter(o=>o.isSprite)];const visibility=overlays.map(o=>o.visible);overlays.forEach(o=>o.visible=false);renderer.render(scene,inset);overlays.forEach((o,i)=>o.visible=visibility[i]);renderer.setScissorTest(false);
}
function drawChart(target,kind){
 const W=560,H=175,L=38,R=15,T=12,B=28;
 const data=Object.entries(runs).map(([method,r])=>({method,values:plotValues(r,kind)}));
 const ymax=['left','right'].includes(kind)?.75:kind==='uncertainty'?2:Math.max(.1,...data.flatMap(d=>d.values.map(v=>v[1])))*1.08;
 const x=v=>L+(W-L-R)*v/duration,y=v=>H-B-(H-T-B)*v/ymax;
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${{cost:'Cumulative monitoring cost',uncertainty:'Wind uncertainty',left:'Left clarity deficit',right:'Right clarity deficit',error:'Wind estimation error',speed:'Ground speed'}[kind]} for all three methods">`;
 for(let i=0;i<4;i++){const v=i*ymax/3;svg+=`<path d="M${L},${y(v)}H${W-R}" stroke="#223747"/><text x="${L-7}" y="${y(v)+3}" text-anchor="end">${v.toFixed(kind==='cost'?0:1)}</text>`}
 for(let i=0;i<5;i++){const v=i*duration/4;svg+=`<text x="${x(v)}" y="${H-8}" text-anchor="middle">${v.toFixed(0)} s</text>`}
 for(const {method,values} of data){svg+=`<path d="${values.map(([a,b],i)=>`${i?'L':'M'}${x(a).toFixed(1)},${y(b).toFixed(1)}`).join('')}" fill="none" stroke="${COLORS[method]}" stroke-width="2.5"/>`}
 $(target).innerHTML=svg+`<line class="chart-cursor" x1="${x(t)}" x2="${x(t)}" y1="${T}" y2="${H-B}" stroke="#e2ce9b" stroke-dasharray="3 3"/></svg>`;
}
function updateDiagnostics(){
 const run=runs[$('#belief-method').value];if(!run)return;
 const belief=run.beliefs[indexAt(run.beliefs,t)],frame=run.frames[indexAt(run.frames,t)];
 const component=$('#belief-component').value;
 for(const kind of ['true','mean','std'])drawBelief($('#belief-'+kind),run,belief,kind,component,frame);
 $('#belief-time').textContent=`Belief available at ${belief.t.toFixed(0)} s · replay ${t.toFixed(1)} s`;
 document.querySelectorAll('.chart-cursor').forEach(line=>{line.setAttribute('x1',38+(560-38-15)*t/duration);line.setAttribute('x2',38+(560-38-15)*t/duration)});
}
async function loadCase(name){
 const sequence=++loadSequence;
 selected=name;t=0;playing=false;follow=false;$('#camera').textContent='Follow drone';dirty=true;$('#play').textContent='▶ Play';$('#comparison').innerHTML='<div class="loading">Loading recorded nonlinear simulations…</div>';
 for(const v of views){v.observer.disconnect();v.scene.traverse(o=>{o.geometry?.dispose();for(const m of (Array.isArray(o.material)?o.material:[o.material])){m?.map?.dispose();m?.dispose()}});v.renderer.dispose();v.renderer.forceContextLoss();v.controls.dispose()}views=[];$('#comparison').classList.remove('has-large');
 const entries=manifest.runs.filter(r=>r.case===name),loaded={};await Promise.all(entries.map(async e=>{const response=await fetch(e.file);if(!response.ok)throw Error(e.file);loaded[e.method]=await response.json()}));if(sequence!==loadSequence)return;runs=loaded;
 runs=Object.fromEntries(['proposed','aware','blind'].filter(m=>runs[m]).map(m=>[m,runs[m]]));
 $('#comparison').innerHTML='';for(const method of ['proposed','aware','blind']){if(runs[method])views.push(makeView(method,runs[method]));else{const card=document.createElement('article');card.className='method loading';card.textContent=`${names[method]}: simulation still running`;$('#comparison').append(card)}}
 duration=Math.max(360,...Object.values(runs).map(r=>r.mission_seconds));$('#scrub').max=duration;
 $('#prediction-duration').textContent=`${name==='commitment'?240:120} s prediction · 360 s mission`;
 $('#case-tag').textContent=`ENVIRONMENT ${name==='forward'?'1A':name==='reverse'?'1B':name==='crosswind'?'2':'3'}`;$('#case-title').textContent=descriptions[name][0];$('#case-description').textContent=descriptions[name][1];
 $('#scenarios').querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.case===name));
 const rows=Object.entries(runs).map(([m,r])=>`<tr><td>${names[m]}</td><td>${r.cost.toFixed(2)}</td><td>${r.min_clearance.toFixed(2)} m</td><td class="${r.success?'':'failure'}">${r.success?'Completed':r.failure}</td></tr>`).join('');
 const occupancy=Object.entries(runs).map(([method,r])=>{const times=[0,0,0];for(let i=1;i<r.frames.length;i++){const f=r.frames[i],dt=f.t-r.frames[i-1].t;const roi=r.environment.regions;let found=false;roi.forEach((a,k)=>{if(f.p[0]>=a[0]&&f.p[0]<=a[2]&&f.p[1]>=a[1]&&f.p[1]<=a[3]){times[k]+=dt;found=true}});if(!found)times[2]+=dt}return `<tr><td>${names[method]}</td><td>${times[0].toFixed(1)} s</td><td>${times[1].toFixed(1)} s</td><td>${times[2].toFixed(1)} s</td></tr>`}).join('');
 const passageEvents=Object.fromEntries(Object.entries(runs).map(([m,r])=>[m,crossings(r)]));
 const passageRows=Object.entries(passageEvents).map(([method,events])=>{
  const count=direction=>['A','B','C'].map(p=>{const n=events.filter(e=>e.direction===direction&&e.passage===p).length;return n?`${p}: ${n}`:null}).filter(Boolean).join(' · ')||'None';
  return `<tr><td>${names[method]}</td><td>${count('right')}</td><td>${count('left')}</td></tr>`;
 }).join('');
 const count=(direction,passage)=>(passageEvents.proposed||[]).filter(e=>e.direction===direction&&e.passage===passage).length;
 let routeExplanation='';
 if(name==='forward'&&count('right','A')&&count('left','C'))routeExplanation='The proposed run travels right with the wind through A and returns through calm C. This avoids the slow upwind traversal demonstrated in the physical passage check.';
 if(name==='reverse'&&count('right','C')&&count('left','A'))routeExplanation='The proposed run reverses circulation: calm C for rightward travel and wind-assisted A for the return. The wind direction changes which passage is useful in each direction.';
 if(name==='crosswind'&&count('right','B'))routeExplanation='The proposed run reaches the right region through calm B. Its longer geometric route avoids slow headwind progress in A/C; the table shows whether it uses the wind-assisted passages for return travel.';
 const passageTable=`<p>${routeExplanation}</p><table><thead><tr><th>EXECUTED CROSSINGS</th><th>RIGHTWARD</th><th>LEFTWARD</th></tr></thead><tbody>${passageRows}</tbody></table>`;
 const mechanism=`${passageTable}<p>Executed time in the inspection regions explains part of the cost difference. Clarity also depends on sensing distance, revisit timing and decay.</p><table><thead><tr><th>METHOD</th><th>LEFT REGION</th><th>RIGHT REGION</th><th>OUTSIDE REGIONS</th></tr></thead><tbody>${occupancy}</tbody></table>`;
 const p=runs.proposed,a=runs.aware,b=runs.blind;let explanation='';if(p&&a&&b){const pa=100*(a.cost-p.cost)/a.cost;explanation=`<p>The proposed run has ${Math.abs(pa).toFixed(1)}% ${pa>=0?'lower':'higher'} cost than disturbance-aware planning. ${pa>0?'The lower cost is observed in this run; it is not yet a multi-seed learning-effect result.':'An additional learning advantage is not established by this run.'} ${Object.values(runs).every(r=>r.success)?'All three completed the mission.':'Failed runs include the remaining mission with no further sensing in their reported cost.'}</p>`}
 if(p&&a&&b&&p.cost>b.cost&&a.cost>b.cost)explanation+='<p><b>Both informed planners have higher monitoring cost than disturbance-blind in this case. This is an unsuccessful performance comparison.</b></p>';
 drawChart('#cost-chart','cost');drawChart('#uncertainty-chart','uncertainty');drawChart('#left-deficit','left');drawChart('#right-deficit','right');drawChart('#wind-error','error');drawChart('#speed-chart','speed');
 const staysRight=passageEvents.proposed?.some(e=>e.direction==='right')&&!passageEvents.proposed?.some(e=>e.direction==='left');
 if(name==='commitment'&&staysRight)$('#case-description').textContent+=' The recorded proposed run crosses and remains on the right; it does not demonstrate the intended turn-back/stay-left behavior.';
 const crossingNote=name==='commitment'?`<p>This environment tests whether servicing the larger right region compensates for slow return travel. ${staysRight?'<b>The proposed run crosses to the right region without returning; the intended turn-back/stay-left behavior is not demonstrated.</b>':'The trails and occupancy table show the actual decision.'} No turn-back behavior is prescribed.</p>`:'';
 $('#summary').innerHTML=`<table><thead><tr><th>METHOD</th><th>COST ↓</th><th>MIN. CLEARANCE</th><th>OUTCOME</th></tr></thead><tbody>${rows}</tbody></table>${explanation}${crossingNote}${mechanism}`;history.replaceState(null,'',`#${name}`);dirty=true;
}
$('#play').onclick=()=>{playing=!playing;if(t>=duration)t=0;$('#play').textContent=playing?'Ⅱ Pause':'▶ Play'};$('#restart').onclick=()=>{t=0;dirty=true};$('#scrub').oninput=e=>{t=+e.target.value;dirty=true};$('#camera').onclick=()=>{dirty=true;follow=!follow;$('#camera').textContent=follow?'Overview':'Follow drone';for(const v of views){v.camera.zoom=follow?4:1;if(!follow){v.camera.position.set(20,7,80);v.controls.target.set(20,14,1)}v.camera.updateProjectionMatrix()}};
for(const id of ['attitude','uncertainty','wind-toggle','flow-source','footprint','clarity-toggle','belief-method','belief-component'])$('#'+id).onchange=()=>{dirty=true};
function animate(stamp){requestAnimationFrame(animate);const dt=Math.min(.1,(stamp-last)/1000);last=stamp;if(playing){t=Math.min(duration,t+dt*+$('#speed').value);dirty=true;if(t>=duration){playing=false;$('#play').textContent='▶ Play'}}if(!dirty||stamp-lastRender<33)return;dirty=false;lastRender=stamp;$('#scrub').value=t;$('#time').textContent=`${t.toFixed(1)} / ${duration.toFixed(0)} s`;views.forEach(updateView);updateDiagnostics();}
try{manifest=await(await fetch('data/manifest.json')).json();for(const name of Object.keys(descriptions)){const b=document.createElement('button');b.dataset.case=name;b.textContent=descriptions[name][0].split(' · ')[0]+' · '+({forward:'Forward',reverse:'Reversed',crosswind:'Crosswind',commitment:'Crossing decision'}[name]);b.onclick=()=>loadCase(name);$('#scenarios').append(b)}const initial=location.hash.slice(1);await loadCase(descriptions[initial]?initial:'forward');requestAnimationFrame(animate)}catch(e){$('#comparison').innerHTML=`<div class="loading">Unable to load recorded results: ${e.message}</div>`;console.error(e)}
window.urbanDemo={get state(){return {selected,time:t,views:views.length,runs:Object.keys(runs),playing}},setTime(value){t=value}};
