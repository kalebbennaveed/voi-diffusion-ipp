import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {createDrone} from './models.js';
const $=s=>document.querySelector(s), COLORS={proposed:'#71d5b3',aware:'#79b7ed',blind:'#ecaa85'};
const PATH_COLORS={proposed:'#087d61',aware:'#176bb8',blind:'#b64d29'};
const names={proposed:'Proposed',aware:'Disturbance-aware',blind:'Disturbance-blind'};
const descriptions={forward:['1a · Forward wind in passage A','A tailwind on the outward crossing becomes a headwind on the return. Routes A, B and C remain available.'],reverse:['1b · Same environment, reversed wind','The entire disturbance vector in A is reversed. Geometry, initial conditions and the three controllers are unchanged.'],crosswind:['2 · A sheltered alternative','Passage B has no wind load. Alternating lateral jets in A and adverse loading in C make the shorter passages difficult to track.'],commitment:['3 · Is the crossing worth its cost?','The right inspection region is larger. Strong eastward loads make a return costly; the monitoring objective decides whether crossing pays off.']};
let runs={},views=[],t=0,playing=false,last=0,follow=false,selected='forward',duration=120;
let manifest;let dirty=true,lastRender=0,loadSequence=0;
function material(color,extra={}){return new THREE.MeshStandardMaterial({color,roughness:.82,...extra})}
function box(scene,size,pos,color,extra={}){const m=new THREE.Mesh(new THREE.BoxGeometry(...size),material(color,extra));m.position.set(...pos);m.castShadow=true;m.receiveShadow=true;scene.add(m);return m;}
function textSprite(text,color='#183746',scale=2){const c=document.createElement('canvas');c.width=512;c.height=128;const a=c.getContext('2d');a.font='600 78px system-ui';a.textAlign='center';a.textBaseline='middle';if(['A','B','C'].includes(text)){a.fillStyle='#153247';a.beginPath();a.roundRect(197,10,118,108,18);a.fill();a.fillStyle='#ffffff'}else{a.fillStyle=color}a.fillText(text,256,64);const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),depthTest:false,transparent:true,fog:false,toneMapped:false}));s.scale.set(scale*4,scale,1);s.renderOrder=9;return s;}
function line(scene,points,color,dashed=false){const g=new THREE.BufferGeometry().setFromPoints(points.map(p=>new THREE.Vector3(...p)));const m=dashed?new THREE.LineDashedMaterial({color,dashSize:.5,gapSize:.3,depthTest:true}):new THREE.LineBasicMaterial({color,fog:false,toneMapped:false});const l=new THREE.Line(g,m);if(dashed)l.computeLineDistances();scene.add(l);return l;}
const c0=new THREE.Color('#733957'),c1=new THREE.Color('#ffbc77'),c2=new THREE.Color('#bde6ad'),c3=new THREE.Color('#369a95');
function clarityColor(q){if(q<.33)return c0.clone().lerp(c1,q/.33);if(q<.67)return c1.clone().lerp(c2,(q-.33)/.34);return c2.clone().lerp(c3,(q-.67)/.33)}
function uncertaintyColor(s){return new THREE.Color('#1f6374').lerp(new THREE.Color('#f3ca7b'),Math.min(s/1.4,1))}
function buildVillage(scene,env){
  box(scene,[64,52,.4],[20,14,-.25],0x9bab8a);box(scene,[40,28,.14],[20,14,-.07],0xb6b8aa);
  // Paving is deliberately flat; the collision layout is the recorded model.
  for(const y of [1.5,9,25])box(scene,[39, y===1.5?2.8:5.2,.025],[20,y,.025],0x76858a);
  box(scene,[6,27,.025],[10.3,14,.025],0x76858a);box(scene,[6,27,.025],[31.5,14,.025],0x76858a);
  for(const y of [1.5,9,25])for(let x=12;x<30;x+=2.8)box(scene,[1.2,.06,.02],[x,y,.05],0xc5cbbf);
  for(const [i,r] of env.buildings.entries()){
    const [x0,y0,x1,y1,h]=r,w=x1-x0,d=y1-y0,cx=(x0+x1)/2,cy=(y0+y1)/2;
    box(scene,[w+.15,d+.15,.22],[cx,cy,.11],0xc9c4b4);
    box(scene,[w,d,h],[cx,cy,h/2],i===0?0xd0c6ad:0xbfc4ba);
    box(scene,[w+.28,d+.28,.18],[cx,cy,h+.04],0x916651);
    box(scene,[w-.7,d-.7,.08],[cx,cy,h+.18],0xae8060);
    // Fastpathplanning Village vocabulary: roof cap, framed blue windows.
    for(let z=1.4;z<h-.5;z+=1.9){
      for(let x=x0+.8;x<x1-.3;x+=1.6)for(const y of [y0-.025,y1+.025]){
        box(scene,[.82,.06,.92],[x,y,z],0x293b42);box(scene,[.68,.075,.78],[x,y,z],0x8eb7c5,{metalness:.2,roughness:.3});
      }
      for(let y=y0+.8;y<y1-.3;y+=1.6)for(const x of [x0-.025,x1+.025]){
        box(scene,[.06,.82,.92],[x,y,z],0x293b42);box(scene,[.075,.68,.78],[x,y,z],0x8eb7c5,{metalness:.2,roughness:.3});
      }
    }
    for(let n=0;n<Math.floor(w/4);n++)box(scene,[1.2,1.2,.45],[x0+2+n*3,cy,h+.45],0x8a9595);
  }
  for(const [i,r] of env.regions.entries()){
    const [a,b,c,d]=r;line(scene,[[a,b,.08],[c,b,.08],[c,d,.08],[a,d,.08],[a,b,.08]],0xf1b45d);
    const label=textSprite(i?'INSPECTION · RIGHT':'INSPECTION · LEFT','#3d4c45',.60);label.position.set((a+c)/2,b-.65,.3);scene.add(label);
  }
  for(const [s,y] of [['A',9],['B',25],['C',1.5]]){const label=textSprite(s,'#e3ebe7',2.2);label.position.set(12,y,.3);scene.add(label)}
  // Non-colliding decoration stays outside the simulation boundary.
  for(let i=0;i<22;i++){
    let x=-3+(i%11)*4.8,y=i<11?-3.5:31.5;
    box(scene,[.3,.3,1.6],[x,y,.8],0x74604b);box(scene,[1.5,1.5,1.7],[x,y,2.1],i%2?0x788e5d:0x607959);
  }
  line(scene,[[0,0,.07],[40,0,.07],[40,28,.07],[0,28,.07],[0,0,.07]],0xf2e5c2);
  const bar=line(scene,[[1,27,.1],[6,27,.1]],0xf0ede4);const label=textSprite('5 m','#e6ede7',.35);label.position.set(3.5,26.4,.3);scene.add(label);
}
function makeView(method,run){
  const card=document.createElement('article');card.className='method';card.style.setProperty('--method',COLORS[method]);
  const base=runs.blind?.cost, gain=base?100*(base-run.cost)/base:0;
  card.innerHTML=`<div class="method-head"><div><h3>${names[method]}</h3><small>${method==='proposed'?'Anticipates future observations':method==='aware'?'Plans with the current wind belief':'Plans without a disturbance field'}</small></div><div class="card-actions"><span class="chip">${run.success?'COMPLETED':'FAILED'}</span><button class="enlarge" aria-label="Enlarge ${names[method]} view">⤢</button></div></div><div class="viewport"><span class="corner">3D FLIGHT · METRES</span><span class="inset-label">ACTUAL ATTITUDE</span></div><div class="metrics"><div><strong>${run.cost.toFixed(2)}</strong><small>MISSION COST ↓</small></div><div><strong class="gain ${method==='blind'?'neutral':gain>=0?'improved':'worse'}">${method==='blind'?'baseline':!base?'pending':`${gain>=0?'−':'+'}${Math.abs(gain).toFixed(1)}%`}</strong><small>VS. BLIND COST</small></div><div><strong>${run.rms_tracking_error.toFixed(2)} m</strong><small>TRACKING RMS</small></div></div><div class="live"><span class="live-cost">Cost: 0.00</span><span class="live-std">Wind σ: 1.40 m/s²</span></div>`;
  $('#comparison').append(card);card.querySelector('.enlarge').onclick=()=>{const expanded=card.classList.toggle('is-large');$('#comparison').classList.toggle('has-large',expanded);card.querySelector('.enlarge').textContent=expanded?'Compare all three':'⤢';card.querySelector('.enlarge').setAttribute('aria-label',expanded?'Compare all three':`Enlarge ${names[method]} view`)};const host=card.querySelector('.viewport');
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.2;host.prepend(renderer.domElement);
  const scene=new THREE.Scene();scene.background=new THREE.Color('#b8ccd0');scene.fog=new THREE.Fog('#b8ccd0',120,200);
  scene.add(new THREE.HemisphereLight(0xe1f0ff,0x727e59,2.2));const sun=new THREE.DirectionalLight(0xffedcc,3.5);sun.position.set(-5,-15,45);sun.castShadow=true;sun.shadow.mapSize.set(1024,1024);Object.assign(sun.shadow.camera,{left:-35,right:35,top:35,bottom:-35,near:1,far:100});sun.shadow.bias=-.0005;scene.add(sun);
  buildVillage(scene,run.environment);
  // Batch repeated windows/vegetation without changing the visual geometry.
  const groups=new Map();for(const mesh of [...scene.children]){if(!mesh.isMesh||mesh.geometry.type!=='BoxGeometry')continue;const p=mesh.geometry.parameters,key=[p.width,p.height,p.depth,mesh.material.color.getHex(),mesh.material.roughness,mesh.material.metalness].join('/');if(!groups.has(key))groups.set(key,[]);groups.get(key).push(mesh)}
  for(const group of groups.values()){if(group.length<4)continue;const first=group[0],batch=new THREE.InstancedMesh(first.geometry,first.material,group.length);group.forEach((m,i)=>{m.updateMatrix();batch.setMatrixAt(i,m.matrix);scene.remove(m);if(i){m.geometry.dispose();m.material.dispose()}});batch.castShadow=true;batch.receiveShadow=true;scene.add(batch)}
  const camera=new THREE.PerspectiveCamera(43,1,.05,180);camera.up.set(0,0,1);camera.position.set(32,0,70);
  const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(20,14,1);controls.minDistance=3;controls.maxDistance=95;controls.maxPolarAngle=Math.PI*.47;controls.enableDamping=true;controls.addEventListener('change',()=>{dirty=true});controls.update();
  const inset=new THREE.PerspectiveCamera(40,1,.01,150);inset.up.set(0,0,1);
  const drone=createDrone();drone.scale.setScalar(.08/.17);scene.add(drone);
  const ring=new THREE.Mesh(new THREE.TorusGeometry(.70,.07,8,40),new THREE.MeshBasicMaterial({color:PATH_COLORS[method],fog:false,toneMapped:false}));scene.add(ring);
  const cellGeometry=new THREE.PlaneGeometry(.84,.84);const cells=run.environment.cells.map(p=>{const m=new THREE.Mesh(cellGeometry,new THREE.MeshBasicMaterial({color:clarityColor(.35),transparent:true,opacity:.9,fog:false,toneMapped:false}));m.position.set(p[0],p[1],.09);scene.add(m);return m});
  const uncertainty=[];for(const p of run.environment.grid){if(run.environment.buildings.some(r=>p[0]>r[0]-.2&&p[0]<r[2]+.2&&p[1]>r[1]-.2&&p[1]<r[3]+.2))continue;const g=new THREE.PlaneGeometry(1.9,1.9);const m=new THREE.Mesh(g,new THREE.MeshBasicMaterial({color:0xf3ca7b,transparent:true,opacity:.65,depthWrite:false,fog:false,toneMapped:false}));m.position.set(p[0],p[1],.12);m.userData.gridIndex=run.environment.grid.indexOf(p);scene.add(m);uncertainty.push(m)}
  const arrows=new THREE.Group();scene.add(arrows);
  run.environment.grid.forEach((p,i)=>{const w=run.environment.wind[i],mag=Math.hypot(...w);if(mag<.15||i%2)return;const color=new THREE.Color('#3da8ef').lerp(new THREE.Color('#0053b8'),Math.min(mag/3.5,1));const direction=new THREE.Vector3(w[0],w[1],0).normalize(),length=1.3+.45*mag;const arrow=new THREE.Group();arrow.position.set(p[0],p[1],3.5);const stem=new THREE.Mesh(new THREE.CylinderGeometry(.11,.11,length-.35,8),new THREE.MeshBasicMaterial({color,fog:false,toneMapped:false}));stem.position.y=(length-.35)/2;const tip=new THREE.Mesh(new THREE.ConeGeometry(.32,.55,12),new THREE.MeshBasicMaterial({color,fog:false,toneMapped:false}));tip.position.y=length-.18;arrow.add(stem,tip);arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),direction);arrows.add(arrow)});
  const points=run.frames.map(f=>new THREE.Vector3(...f.p));const trail=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:PATH_COLORS[method],fog:false,toneMapped:false}));scene.add(trail);
  let refline=line(scene,[[6,6,3],[6,6,3]],0xffffff,true);
  const resize=()=>{if(!host.clientWidth||!host.clientHeight)return;renderer.setSize(host.clientWidth,host.clientHeight,false);camera.aspect=host.clientWidth/host.clientHeight;camera.updateProjectionMatrix();dirty=true};const observer=new ResizeObserver(resize);observer.observe(host);resize();
  return {method,run,card,host,renderer,scene,camera,controls,inset,drone,ring,cells,uncertainty,arrows,trail,refline,lastPlan:-1,observer};
}
function indexAt(array,time){let lo=0,hi=array.length-1;while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(array[mid].t<=time)lo=mid;else hi=mid-1}return lo}
function updateView(v){
 if(!v.host.clientWidth||!v.host.clientHeight)return;
 const {run,drone,renderer,scene,camera,controls,inset,host}=v;const i=indexAt(run.frames,t),f=run.frames[i],g=run.frames[Math.min(i+1,run.frames.length-1)],fraction=Math.max(0,Math.min(1,(t-f.t)/(g.t-f.t||1)));
 drone.position.fromArray(f.p).lerp(new THREE.Vector3(...g.p),fraction);drone.quaternion.fromArray(f.quat).slerp(new THREE.Quaternion(...g.quat),fraction);
 drone.userData.rotors.forEach((r,k)=>r.rotation.z=t*f.motors[k]*(k%2?1:-1));
 v.ring.position.copy(drone.position);v.ring.position.z=drone.position.z+.12;v.trail.geometry.setDrawRange(0,i+1);v.cells.forEach((m,j)=>m.material.color.copy(clarityColor(f.q[j])));
 const b=run.beliefs[indexAt(run.beliefs,t)];v.uncertainty.forEach(m=>{m.visible=$('#uncertainty').checked;m.material.color.copy(uncertaintyColor(b.std[m.userData.gridIndex]))});v.arrows.visible=$('#wind-toggle').checked;
 v.card.querySelector('.live-cost').textContent=`Cost: ${f.cost.toFixed(2)}`;v.card.querySelector('.live-std').textContent=`Wind σ: ${(b.std.reduce((a,x)=>a+x,0)/b.std.length).toFixed(2)} m/s²`;
 const planIndex=indexAt(run.plans,t);if(planIndex!==v.lastPlan){v.lastPlan=planIndex;v.refline.geometry.dispose();v.refline.geometry=new THREE.BufferGeometry().setFromPoints(run.plans[planIndex].points.map(p=>new THREE.Vector3(p[0],p[1],3.04)));v.refline.computeLineDistances()}
 if(follow){camera.position.lerp(drone.position.clone().add(new THREE.Vector3(7,-10,9)),.08);controls.target.lerp(drone.position,.08)}controls.update();
 renderer.setScissorTest(false);renderer.setViewport(0,0,host.clientWidth,host.clientHeight);renderer.render(scene,camera);
 const s=Math.min(135,host.clientWidth*.32),x=host.clientWidth-s-7,y=7;inset.position.copy(drone.position).add(new THREE.Vector3(.55,-.75,.45));inset.lookAt(drone.position);renderer.clearDepth();renderer.setScissorTest(true);renderer.setScissor(x,y,s,s);renderer.setViewport(x,y,s,s);const overlays=[v.ring,v.arrows,v.trail,v.refline,...scene.children.filter(o=>o.isSprite)];const visibility=overlays.map(o=>o.visible);overlays.forEach(o=>o.visible=false);renderer.render(scene,inset);overlays.forEach((o,i)=>o.visible=visibility[i]);renderer.setScissorTest(false);
}
function drawChart(target,kind){
 const W=560,H=175,L=38,R=15,T=12,B=28;
 const data=Object.entries(runs).map(([method,r])=>({method,values:kind==='cost'?r.frames.map(f=>[f.t,f.cost]):r.beliefs.map(b=>[b.t,b.std.reduce((a,x)=>a+x,0)/b.std.length])}));
 const ymax=kind==='cost'?Math.max(1,...Object.values(runs).map(r=>r.cost))*1.08:1.5;
 const x=v=>L+(W-L-R)*v/duration,y=v=>H-B-(H-T-B)*v/ymax;
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${kind==='cost'?'Cumulative monitoring cost':'Wind uncertainty'} for all three methods">`;
 for(let i=0;i<4;i++){const v=i*ymax/3;svg+=`<path d="M${L},${y(v)}H${W-R}" stroke="#223747"/><text x="${L-7}" y="${y(v)+3}" text-anchor="end">${v.toFixed(kind==='cost'?0:1)}</text>`}
 for(let i=0;i<5;i++){const v=i*duration/4;svg+=`<text x="${x(v)}" y="${H-8}" text-anchor="middle">${v.toFixed(0)} s</text>`}
 for(const {method,values} of data){svg+=`<path d="${values.map(([a,b],i)=>`${i?'L':'M'}${x(a).toFixed(1)},${y(b).toFixed(1)}`).join('')}" fill="none" stroke="${COLORS[method]}" stroke-width="2.5"/>`}
 $(target).innerHTML=svg+'</svg>';
}
async function loadCase(name){
 const sequence=++loadSequence;
 selected=name;t=0;playing=false;dirty=true;$('#play').textContent='▶ Play';$('#comparison').innerHTML='<div class="loading">Loading recorded nonlinear simulations…</div>';
 for(const v of views){v.observer.disconnect();v.scene.traverse(o=>{o.geometry?.dispose();for(const m of (Array.isArray(o.material)?o.material:[o.material])){m?.map?.dispose();m?.dispose()}});v.renderer.dispose();v.renderer.forceContextLoss();v.controls.dispose()}views=[];$('#comparison').classList.remove('has-large');
 const entries=manifest.runs.filter(r=>r.case===name),loaded={};await Promise.all(entries.map(async e=>{const response=await fetch(e.file);if(!response.ok)throw Error(e.file);loaded[e.method]=await response.json()}));if(sequence!==loadSequence)return;runs=loaded;
 runs=Object.fromEntries(['proposed','aware','blind'].filter(m=>runs[m]).map(m=>[m,runs[m]]));
 $('#comparison').innerHTML='';for(const method of ['proposed','aware','blind']){if(runs[method])views.push(makeView(method,runs[method]));else{const card=document.createElement('article');card.className='method loading';card.textContent=`${names[method]}: simulation still running`;$('#comparison').append(card)}}
 duration=Math.max(120,...Object.values(runs).map(r=>r.mission_seconds));$('#scrub').max=duration;
 $('#case-tag').textContent=`ENVIRONMENT ${name==='forward'?'1A':name==='reverse'?'1B':name==='crosswind'?'2':'3'}`;$('#case-title').textContent=descriptions[name][0];$('#case-description').textContent=descriptions[name][1];
 $('#scenarios').querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.case===name));
 const rows=Object.entries(runs).map(([m,r])=>`<tr><td>${names[m]}</td><td>${r.cost.toFixed(2)}</td><td>${r.min_clearance.toFixed(2)} m</td><td class="${r.success?'':'failure'}">${r.success?'Completed':r.failure}</td></tr>`).join('');
 const occupancy=Object.entries(runs).map(([method,r])=>{const times=[0,0,0];for(let i=1;i<r.frames.length;i++){const f=r.frames[i],dt=f.t-r.frames[i-1].t;const roi=r.environment.regions;let found=false;roi.forEach((a,k)=>{if(f.p[0]>=a[0]&&f.p[0]<=a[2]&&f.p[1]>=a[1]&&f.p[1]<=a[3]){times[k]+=dt;found=true}});if(!found)times[2]+=dt}return `<tr><td>${names[method]}</td><td>${times[0].toFixed(1)} s</td><td>${times[1].toFixed(1)} s</td><td>${times[2].toFixed(1)} s</td></tr>`}).join('');
 const mechanism=`<p>Executed time in the inspection regions explains part of the cost difference. Clarity also depends on sensing distance, revisit timing and decay.</p><table><thead><tr><th>METHOD</th><th>LEFT REGION</th><th>RIGHT REGION</th><th>OUTSIDE REGIONS</th></tr></thead><tbody>${occupancy}</tbody></table>`;
 const p=runs.proposed,a=runs.aware,b=runs.blind;let explanation='';if(p&&a&&b){const pa=100*(a.cost-p.cost)/a.cost;explanation=`<p>The proposed run has ${Math.abs(pa).toFixed(1)}% ${pa>=0?'lower':'higher'} cost than disturbance-aware planning. ${pa>0?'The lower cost is observed in this run; it is not yet a multi-seed learning-effect result.':'An additional learning advantage is not established by this run.'} ${Object.values(runs).every(r=>r.success)?'All three completed the mission.':'Failed runs include the remaining mission with no further sensing in their reported cost.'}</p>`}
 drawChart('#cost-chart','cost');drawChart('#uncertainty-chart','uncertainty');
 const crossingNote=name==='commitment'?'<p><b>The turn-back hypothesis was not observed.</b> Proposed and disturbance-aware planning both cross to the larger right inspection region. This environment therefore does not yet demonstrate the intended stay-left behaviour.</p>':'';
 $('#summary').innerHTML=`<table><thead><tr><th>METHOD</th><th>COST ↓</th><th>MIN. CLEARANCE</th><th>OUTCOME</th></tr></thead><tbody>${rows}</tbody></table>${explanation}${crossingNote}${mechanism}`;history.replaceState(null,'',`#${name}`);dirty=true;
}
$('#play').onclick=()=>{playing=!playing;if(t>=duration)t=0;$('#play').textContent=playing?'Ⅱ Pause':'▶ Play'};$('#restart').onclick=()=>{t=0;dirty=true};$('#scrub').oninput=e=>{t=+e.target.value;dirty=true};$('#camera').onclick=()=>{dirty=true;follow=!follow;$('#camera').textContent=follow?'Overview':'Follow drone';if(!follow)for(const v of views){v.camera.position.set(32,0,70);v.controls.target.set(20,14,1)}};
$('#uncertainty').onchange=()=>{dirty=true};$('#wind-toggle').onchange=()=>{dirty=true};
function animate(stamp){requestAnimationFrame(animate);const dt=Math.min(.1,(stamp-last)/1000);last=stamp;if(playing){t=Math.min(duration,t+dt*+$('#speed').value);dirty=true;if(t>=duration){playing=false;$('#play').textContent='▶ Play'}}if(!dirty||stamp-lastRender<33)return;dirty=false;lastRender=stamp;$('#scrub').value=t;$('#time').textContent=`${t.toFixed(1)} / ${duration.toFixed(0)} s`;views.forEach(updateView)}
try{manifest=await(await fetch('data/manifest.json')).json();for(const name of Object.keys(descriptions)){const b=document.createElement('button');b.dataset.case=name;b.textContent=descriptions[name][0].split(' · ')[0]+' · '+({forward:'Forward',reverse:'Reversed',crosswind:'Crosswind',commitment:'Crossing decision'}[name]);b.onclick=()=>loadCase(name);$('#scenarios').append(b)}const initial=location.hash.slice(1);await loadCase(descriptions[initial]?initial:'forward');requestAnimationFrame(animate)}catch(e){$('#comparison').innerHTML=`<div class="loading">Unable to load recorded results: ${e.message}</div>`;console.error(e)}
window.urbanDemo={get state(){return {selected,time:t,views:views.length,runs:Object.keys(runs),playing}},setTime(value){t=value}};
