import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {RGBELoader} from 'three/addons/loaders/RGBELoader.js';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {SSAOPass} from 'three/addons/postprocessing/SSAOPass.js';
import {OutputPass} from 'three/addons/postprocessing/OutputPass.js';
import {createDrone} from './drone.js';
import {layout,buildQuarter} from './world.js?v=rrt-20261021-runtime';
const $=s=>document.querySelector(s),container=$('#scene');
const panels={belief:'top',deficit:'bottom',clarity:'off',error:'off'};
const mapNames={belief:'Wind belief',deficit:'Clarity deficit map',clarity:'Clarity map',error:'Wind estimation error'};
function plotHTML(id){return `<div class="plot-title">${mapNames[id]}<button class="map-expand" aria-label="Enlarge ${mapNames[id]}" aria-expanded="false">Enlarge</button></div><div class="map-mode">${id==='belief'?'<select data-belief-mode><option value="uncertainty">Posterior uncertainty</option><option value="mean">Posterior mean speed</option><option value="truth">Physical wind speed</option></select>':''}<span class="map-time">Initial condition</span></div><canvas class="spatial-map" width="512" height="512" data-map="${id}" aria-label="${mapNames[id]} over the environment"></canvas><div class="map-legend" data-legend="${id}"></div>`}
document.addEventListener('click',e=>{const button=e.target.closest('.map-expand');if(!button)return;const dock=button.closest('.plot-dock');const opening=!dock.classList.contains('map-expanded');document.querySelectorAll('.map-expanded').forEach(el=>{el.classList.remove('map-expanded');const b=el.querySelector('.map-expand');b.textContent='Enlarge';b.setAttribute('aria-expanded','false')});dock.classList.toggle('map-expanded',opening);button.textContent=opening?'Close':'Enlarge';button.setAttribute('aria-expanded',String(opening));button.setAttribute('aria-label',(opening?'Close ':'Enlarge ')+mapNames[dock.querySelector('canvas').dataset.map]);});
document.addEventListener('keydown',e=>{if(e.key==='Escape')document.querySelector('.map-expanded .map-expand')?.click()});
function updatePanels(){for(const slot of ['top','bottom']){const id=Object.keys(panels).find(k=>panels[k]===slot),el=$('#dock-'+slot);el.hidden=!id;if(id)el.innerHTML=plotHTML(id)}document.querySelectorAll('[data-panel]').forEach(e=>e.value=panels[e.dataset.panel]);window.dispatchEvent(new Event('map-panels-changed'));}
document.querySelectorAll('[data-panel]').forEach(e=>e.addEventListener('change',()=>{if(e.value!=='off')for(const k in panels)if(k!==e.dataset.panel&&panels[k]===e.value)panels[k]='off';panels[e.dataset.panel]=e.value;updatePanels()}));updatePanels();
$('#layout-select').value=new URLSearchParams(location.search).get('layout')||'two';$('#layout-select').onchange=()=>{const u=new URL(location.href);u.searchParams.set('layout',$('#layout-select').value);location.href=u};

async function start(){
 const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance'});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=.92;renderer.outputColorSpace=THREE.SRGBColorSpace;container.prepend(renderer.domElement);
 const scene=new THREE.Scene();scene.background=new THREE.Color('#b9ccd5');scene.fog=new THREE.FogExp2('#becdcf',.00085);
 const camera=new THREE.PerspectiveCamera(40,1,.08,550);camera.position.set(48,59,72);
 const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,2,-3);controls.enableDamping=true;controls.dampingFactor=.09;controls.minDistance=1.1;controls.maxDistance=210;controls.maxPolarAngle=Math.PI*.495;controls.screenSpacePanning=true;
 const hemi=new THREE.HemisphereLight(0xe5efff,0x737a7d,.6);scene.add(hemi);
 const sun=new THREE.DirectionalLight(0xfff8ee,2.5);sun.position.set(-45,70,25);sun.castShadow=true;sun.shadow.mapSize.set(4096,4096);Object.assign(sun.shadow.camera,{left:-82,right:82,top:82,bottom:-82,near:1,far:250});sun.shadow.normalBias=.035;sun.shadow.bias=-.00004;sun.shadow.radius=2;sun.target.position.set(0,0,-5);scene.add(sun,sun.target);
 const hdr=await new RGBELoader().loadAsync('assets/daylight.hdr');hdr.mapping=THREE.EquirectangularReflectionMapping;
 const pmrem=new THREE.PMREMGenerator(renderer);scene.environment=pmrem.fromEquirectangular(hdr).texture;scene.environmentIntensity=.65;scene.background=hdr;scene.backgroundBlurriness=.16;scene.backgroundIntensity=.85;
 $('#load-status').textContent='Building façades, roofs and streets';
 const district=new THREE.Group();scene.add(district);const world=await buildQuarter(district,renderer);district.scale.set(layout.scale,1,layout.scale);for(const b of world.buildings)for(const k of ['x','z','width','depth'])b[k]*=layout.scale;layout.inspectionCenters=layout.inspectionCenters.map(p=>p.map(x=>x*layout.scale));const [startX,startZ]=layout.inspectionCenters[0];renderer.shadowMap.autoUpdate=false;renderer.shadowMap.needsUpdate=true;
 renderer.info.autoReset=false;const composer=new EffectComposer(renderer);composer.addPass(new RenderPass(scene,camera));
 const ao=new SSAOPass(scene,camera,1,1);ao.kernelRadius=1.6;ao.minDistance=.001;ao.maxDistance=.13;composer.addPass(ao);composer.addPass(new OutputPass());
 const drone=createDrone();drone.scale.setScalar(.55/.56);drone.rotation.x=-Math.PI/2;drone.position.set(startX,layout.flightHeight,startZ);scene.add(drone);
 const sensor=new THREE.Group();scene.add(sensor);const regionGroup=new THREE.Group();scene.add(regionGroup);const routeGroup=new THREE.Group();scene.add(routeGroup);routeGroup.visible=false;
 const labelObjects=[];
 function label(text,pos,kind='region'){const el=document.createElement('div');el.className='world-label '+kind;el.innerHTML=text;$('#labels').append(el);labelObjects.push({el,pos:new THREE.Vector3(...pos),kind});return el}
 function line(points,color,group,dashed=false,opacity=1){const g=new THREE.BufferGeometry().setFromPoints(points.map(v=>new THREE.Vector3(...v)));const m=dashed?new THREE.LineDashedMaterial({color,dashSize:.65,gapSize:.35,transparent:true,opacity,depthTest:true}):new THREE.LineBasicMaterial({color,transparent:true,opacity});const l=new THREE.Line(g,m);if(dashed)l.computeLineDistances();group.add(l);return l}
 for(const [i,[x,z]] of layout.inspectionCenters.entries()){
  const size=layout.inspectionScale;
  const pts=[[-4.5,-5],[4.5,-5],[4.5,5],[-4.5,5],[-4.5,-5]].map(([a,b])=>[x+a*size,.17,z+b*size]);line(pts,0xd6c292,regionGroup,true,.95);
  const plane=new THREE.Mesh(new THREE.PlaneGeometry(9*size,10*size),new THREE.MeshBasicMaterial({color:0xefd59b,transparent:true,opacity:.055,depthWrite:false}));plane.rotation.x=-Math.PI/2;plane.position.set(x,.11,z);regionGroup.add(plane);
  label(`Inspection ${['west','east','northwest','southeast'][i]}`,[x,.5,z+5.4*size]);
 }
 const rad=3.2;const ring=Array.from({length:97},(_,i)=>[startX+rad*Math.cos(i*Math.PI/48),.19,startZ+rad*Math.sin(i*Math.PI/48)]);line(ring,0x318e81,sensor,false,.8);
 const coneMaterial=new THREE.MeshBasicMaterial({color:0x81c7b2,transparent:true,opacity:.035,depthWrite:false,side:THREE.DoubleSide});const cone=new THREE.Mesh(new THREE.ConeGeometry(rad,layout.flightHeight,48,1,true),coneMaterial);cone.position.set(startX,layout.flightHeight/2,startZ);sensor.add(cone);
 for(let i=0;i<4;i++){const ang=Math.PI/4+i*Math.PI/2;line([[startX,layout.flightHeight,startZ],[startX+rad*Math.cos(ang),.2,startZ+rad*Math.sin(ang)]],0x4fae99,sensor,false,.25)}
 label('Quadrotor <small>3.2 m</small>',[startX,4,startZ],'drone');
 const windGroup=new THREE.Group();scene.add(windGroup);windGroup.visible=false;
 const nWind=64,wpos=new Float32Array(nWind*6),wcol=new Float32Array(nWind*6),wg=new THREE.BufferGeometry();wg.setAttribute('position',new THREE.BufferAttribute(wpos,3));wg.setAttribute('color',new THREE.BufferAttribute(wcol,3));
 for(let i=0;i<nWind;i++){wcol.set([.32,.48,.48,.77,.91,.88],i*6)}
 const wline=new THREE.LineSegments(wg,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.40,depthWrite:false}));windGroup.add(wline);
 const particles=Array.from({length:nWind},(_,i)=>({x:layout.scale*(-25+50*((i*.618)%1)),z:layout.scale*(i%2?6+4*((i*.417)%1):23+10*((i*.333)%1))}));let windTime=0;
 function animateWind(t){if(window.quarterPreview?.windAt){const dt=Math.min(.1,t-windTime)*(window.quarterPreview.playback?.().playing?Number(document.querySelector('#play-speed').value):1);windTime=t;for(let i=0;i<nWind;i++){const p=particles[i],w=window.quarterPreview.windAt(p.x,p.z);p.x+=dt*w[0];p.z+=dt*w[1];if(Math.abs(p.x)>26*layout.scale||p.z<5.5*layout.scale||p.z>34*layout.scale){p.x=layout.scale*(w[0]>=0?-24:24);p.z=layout.scale*(i%2?6+4*((i*.417)%1):23+10*((i*.333)%1))}const y=2+(i%7)*.45;wpos.set([p.x,y,p.z,p.x-w[0]*.25,y,p.z-w[1]*.25],i*6)}wg.attributes.position.needsUpdate=true;wg.computeBoundingSphere();return;}}
 const cameras={overview:{p:[12,150,70],target:[0,0,8],name:'Neighborhood overview',caption:'Offset inspection sites · three narrow passages'},street:{p:[-29,4.6,8],target:[20,4.5,8],name:'Central street · passage A',caption:'The same geometry, viewed near flight height'},top:{p:[0,156,10.01],target:[0,0,10],name:'District layout',caption:'A · central street     B · northern street     C · canal street'},square:{p:[-35,7.3,18.5],target:[-26,3,6],name:'West inspection square',caption:'0.55 m quadrotor · sensing footprint shown for scale'}};
 let transition=null,currentCamera='overview';
 function setCamera(k,instant=false){currentCamera=k;const original=cameras[k],factor=layout.scale,c={...original,p:original.p.map((v,i)=>v*(i===1?(k==='overview'||k==='top'?factor:1):factor)),target:original.target.map((v,i)=>v*(i===1?1:factor))};if(instant){camera.position.set(...c.p);controls.target.set(...c.target);transition=null}else transition={start:performance.now(),p:camera.position.clone(),target:controls.target.clone(),end:new THREE.Vector3(...c.p),endTarget:new THREE.Vector3(...c.target)};$('#view-name').textContent=c.name;$('#view-caption').textContent=c.caption;document.querySelectorAll('[data-camera]').forEach(e=>e.classList.toggle('active',e.dataset.camera===k));}
 document.querySelectorAll('[data-camera]').forEach(e=>e.addEventListener('click',()=>setCamera(e.dataset.camera)));$('#reset').onclick=()=>setCamera(currentCamera);controls.addEventListener('start',()=>transition=null);
 $('#focus').onclick=()=>{document.body.classList.toggle('controls-collapsed');$('#focus').textContent=document.body.classList.contains('controls-collapsed')?'Settings':'Hide settings';$('#focus').setAttribute('aria-expanded',String(!document.body.classList.contains('controls-collapsed')))};
 $('#maps-toggle').onclick=()=>{document.body.classList.toggle('maps-hidden');$('#maps-toggle').textContent=document.body.classList.contains('maps-hidden')?'Show maps':'Hide maps'};
 $('#regions').onchange=()=>{regionGroup.visible=$('#regions').checked};$('#sensor').onchange=()=>{sensor.visible=$('#sensor').checked};
 $('#wind').onchange=()=>{windGroup.visible=$('#wind').checked;$('#wind-controls').hidden=!windGroup.visible;$('.wind-badge').hidden=!windGroup.visible};$('#ao').onchange=()=>ao.enabled=$('#ao').checked;
 $('#sun').oninput=()=>{const a=Number($('#sun').value)/100;sun.position.set(-60+100*a,42+48*Math.sin(a*Math.PI),55-30*a);sun.color.setHSL(.10,.16+(1-a)*.18,.92);sun.intensity=2.8;world.water.material.uniforms.sunDirection.value.copy(sun.position).normalize();renderer.shadowMap.needsUpdate=true;$('#sun-value').textContent=a<.3?'Low afternoon':a>.72?'High daylight':'Afternoon'};
 const resize=()=>{const w=container.clientWidth,h=container.clientHeight;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);composer.setSize(w,h);ao.setSize(Math.floor(w*renderer.getPixelRatio()),Math.floor(h*renderer.getPixelRatio()))};new ResizeObserver(resize).observe(container);resize();
 const project=new THREE.Vector3();let frames=0,last=0,renderSum=0;
 function animate(now){requestAnimationFrame(animate);if(now-last<32)return;last=now;const startTime=performance.now(),t=now/1000;renderer.info.reset();
  if(transition){const f=Math.min(1,(now-transition.start)/900),s=f*f*(3-2*f);camera.position.lerpVectors(transition.p,transition.end,s);controls.target.lerpVectors(transition.target,transition.endTarget,s);if(f===1)transition=null;}controls.update();world.animate(t);if(windGroup.visible)animateWind(t);window.quarterPreview?.tick?.(now);
  drone.userData.rotors.forEach((r,i)=>r.rotation.z=t*(i%2?-65:65));
  for(const l of labelObjects){const enabled=window.quarterPreview?.following?false:l.kind==='route'?routeGroup.visible:l.kind==='drone'?drone.visible&&!window.quarterPreview?.recordedPlayback:regionGroup.visible;project.copy(l.pos).project(camera);l.el.hidden=!enabled||project.z>1||project.x<-1||project.x>1||project.y<-1||project.y>1;if(!l.el.hidden){l.el.style.left=((project.x+1)/2*container.clientWidth)+'px';l.el.style.top=((-project.y+1)/2*container.clientHeight)+'px';}}
  composer.render();frames++;renderSum+=performance.now()-startTime;
 }
 document.querySelectorAll('.dimension-list strong')[0].textContent=(56*layout.scale).toFixed(1)+' m E · '+(18*layout.scale).toFixed(1)+' m N';document.querySelectorAll('.dimension-list strong')[1].textContent=(5*layout.scale).toFixed(1)+' / '+(6*layout.scale).toFixed(1)+' m';document.body.classList.add('loaded');window.quarterPreview={ready:true,scene,camera,controls,world,layout,renderer,drone,sensor,regionGroup,routeGroup,windGroup,panels,setCamera,cancelCameraTransition:()=>transition=null,statistics:()=>({...world.statistics,drawCalls:renderer.info.render.calls,frames,meanFrameMilliseconds:renderSum/Math.max(frames,1)})};setCamera('overview',true);requestAnimationFrame(animate);const maps=await import('./spatial-maps.js?v=rrt-20261021-runtime');await maps.install(window.quarterPreview);
}
start().catch(e=>{console.error(e);$('#load-status').textContent='A scene asset failed to load.';$('#error-banner').hidden=false;$('#error-banner').textContent=e.message;window.previewError=String(e)});
