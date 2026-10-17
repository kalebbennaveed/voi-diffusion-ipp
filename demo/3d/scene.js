import * as THREE from 'three';
import {OrbitControls} from './vendor/three/OrbitControls.js';
import {GLTFLoader} from './vendor/three/GLTFLoader.js';
import {FLIGHT_HEIGHT,toScene,gridShape,fieldColor,sensorRadius,sensorWeight,motionForces} from './data.js';
import {makeLandscape,overlayGeometry} from './terrain.js';
import {terrainHeight,chasePosition,cameraSightClear} from './terrain-math.js';

function label(text){
  const c=document.createElement('canvas');c.width=256;c.height=64;
  const ctx=c.getContext('2d');ctx.fillStyle='#172b2dcc';ctx.beginPath();ctx.roundRect(2,2,252,60,8);ctx.fill();
  ctx.font='500 18px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle='#e8f1e8';ctx.fillText(text,128,33);
  const t=new THREE.CanvasTexture(c);t.colorSpace=THREE.SRGBColorSpace;
  const s=new THREE.Sprite(new THREE.SpriteMaterial({map:t,depthTest:false,transparent:true}));s.scale.set(.75,.1875,1);return s;
}
function line(points,color,opacity=1,height=FLIGHT_HEIGHT){
  const positions=points.map(p=>new THREE.Vector3(...toScene(p,typeof height==='function'?height(p):height)));
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(positions),new THREE.LineBasicMaterial({color,transparent:opacity<1,opacity,toneMapped:false}));
}
function disposeObjects(group){
  const geometries=new Set(),materials=new Set(),textures=new Set();
  group.traverse(o=>{if(o.geometry)geometries.add(o.geometry);if(o.material)(Array.isArray(o.material)?o.material:[o.material]).forEach(m=>materials.add(m));});
  materials.forEach(m=>{for(const v of Object.values(m))if(v?.isTexture)textures.add(v);m.dispose();});
  geometries.forEach(g=>g.dispose());textures.forEach(t=>t.dispose());group.clear();
}
function skyEnvironment(renderer){
  const canvas=document.createElement('canvas');canvas.width=256;canvas.height=128;const ctx=canvas.getContext('2d');
  const g=ctx.createLinearGradient(0,0,0,128);g.addColorStop(0,'#8aafc6');g.addColorStop(.48,'#dbe0d2');g.addColorStop(.53,'#667860');g.addColorStop(1,'#333b29');ctx.fillStyle=g;ctx.fillRect(0,0,256,128);
  const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;texture.mapping=THREE.EquirectangularReflectionMapping;
  const generator=new THREE.PMREMGenerator(renderer),target=generator.fromEquirectangular(texture);texture.dispose();generator.dispose();return target;
}

export class FlightScene {
  constructor(host,data,accent){
    this.host=host;this.data=data;this.accent=accent;this.disposed=false;this.needsRender=true;
    this.scene=new THREE.Scene();this.scene.background=new THREE.Color(0xb7c4c5);this.scene.fog=new THREE.FogExp2(0xb7c4c5,.043);
    this.renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'low-power'});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));this.renderer.shadowMap.enabled=true;this.renderer.shadowMap.type=THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;this.renderer.toneMapping=THREE.ACESFilmicToneMapping;this.renderer.toneMappingExposure=1.15;
    host.appendChild(this.renderer.domElement);this.environment=skyEnvironment(this.renderer);this.scene.environment=this.environment.texture;
    this.renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();host.dispatchEvent(new CustomEvent('scene-error',{bubbles:true,detail:'The 3D graphics context was lost. Reload to restore the terrain view.'}));});
    this.camera=new THREE.PerspectiveCamera(43,1,.04,70);
    this.sensorCamera=new THREE.PerspectiveCamera(110,1,.01,40);this.sensorCamera.up.set(0,0,-1);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;this.controls.dampingFactor=.12;this.controls.minDistance=.65;this.controls.maxDistance=20;this.controls.maxPolarAngle=Math.PI/2-.04;
    this.controls.addEventListener('change',()=>{this.needsRender=true;});
    this.scene.add(new THREE.HemisphereLight(0xd7e8f2,0x566044,.9));
    const sun=new THREE.DirectionalLight(0xffeed5,2.6);sun.position.set(-3,7,4);sun.castShadow=true;
    Object.assign(sun.shadow.camera,{left:-4.3,right:4.3,top:4.3,bottom:-4.3,near:.1,far:19});sun.shadow.mapSize.set(1536,1536);sun.shadow.normalBias=.009;sun.shadow.bias=-.00012;this.scene.add(sun);
    const ready=()=>{if(!this.disposed)this.needsRender=true;};
    this.landscape=makeLandscape(data.environment,ready);this.scene.add(this.landscape.group);
    this.shape=gridShape(data.environment);this.fieldBytes=new Uint8Array(this.shape.nx*this.shape.ny*4);
    this.fieldTexture=new THREE.DataTexture(this.fieldBytes,this.shape.nx,this.shape.ny,THREE.RGBAFormat);this.fieldTexture.colorSpace=THREE.SRGBColorSpace;
    this.fieldTexture.magFilter=THREE.NearestFilter;this.fieldTexture.minFilter=THREE.NearestFilter;
    this.ground=new THREE.Mesh(overlayGeometry(data.environment,this.shape.nx,this.shape.ny),new THREE.MeshBasicMaterial({map:this.fieldTexture,transparent:true,depthWrite:false,toneMapped:false,polygonOffset:true,polygonOffsetFactor:-2}));this.scene.add(this.ground);
    const env=data.environment;this.regionLabels=[];
    for(const [key,name] of [['roi_a','NORTH WETLAND · A'],['roi_b','SOUTH WETLAND · B']]){
      if(!env[key]||!env.region_size)continue;const [x,y]=env[key],[w,h]=env.region_size;
      const points=[];const corners=[[x-w/2,y-h/2],[x+w/2,y-h/2],[x+w/2,y+h/2],[x-w/2,y+h/2],[x-w/2,y-h/2]];
      for(let j=0;j<4;j++)for(let k=0;k<18;k++){const t=k/18;points.push(corners[j].map((v,a)=>v+(corners[j+1][a]-v)*t));}points.push(corners[0]);
      const boundary=line(points,0xd7ddc7,.55,p=>terrainHeight(...p)+.025);
      boundary.material.dispose();boundary.material=new THREE.LineDashedMaterial({color:0xd7ddc7,dashSize:.05,gapSize:.035,transparent:true,opacity:.6});boundary.computeLineDistances();this.scene.add(boundary);
      const s=label(name);s.position.set(...toScene([x,y+h/2+.1],terrainHeight(x,y)+.17));this.scene.add(s);this.regionLabels.push(s);
    }
    this.flight={group:new THREE.Group(),body:new THREE.Group()};this.flight.group.add(this.flight.body);this.scene.add(this.flight.group);
    this.ready=new GLTFLoader().loadAsync('/3d/assets/aircraft.glb').then(gltf=>{
      if(this.disposed){disposeObjects(gltf.scene);return;}
      const bounds=new THREE.Box3().setFromObject(gltf.scene),size=bounds.getSize(new THREE.Vector3()),center=bounds.getCenter(new THREE.Vector3());
      const model=gltf.scene;model.position.sub(center);
      const normalized=new THREE.Group();normalized.add(model);const scale=.54/Math.max(size.x,size.y,size.z);normalized.scale.setScalar(scale);
      normalized.rotation.y=Math.PI/2;
      model.traverse(o=>{if(o.isMesh){o.castShadow=true;o.receiveShadow=true;}});
      this.flight.body.add(normalized);this.flight.mixer=new THREE.AnimationMixer(model);
      gltf.animations.forEach(clip=>this.flight.mixer.clipAction(clip).play());this.needsRender=true;
    });
    this.footprintRings=[.5,.1].map(weight=>{
      const radius=sensorRadius(data.config.sensor_sigma,weight);
      const ring=line(Array.from({length:65},(_,i)=>[Math.cos(i*Math.PI/32)*radius,Math.sin(i*Math.PI/32)*radius]),weight===.5?0xff4a3f:0xffa396,weight===.5?1:.5,.04);
      ring.userData.radius=radius;this.scene.add(ring);return ring;
    });
    const footprintPixels=new Uint8Array(128*128*4),extent=sensorRadius(data.config.sensor_sigma,.01);
    for(let y=0;y<128;y++)for(let x=0;x<128;x++){
      const radius=Math.hypot((x/127-.5)*2*extent,(y/127-.5)*2*extent),weight=sensorWeight(radius,data.config.sensor_sigma);
      footprintPixels.set([255,38,26,Math.round(170*weight*Math.max(0,Math.min(1,(extent-radius)/(extent*.1))))],(y*128+x)*4);
    }
    const footprintTexture=new THREE.DataTexture(footprintPixels,128,128,THREE.RGBAFormat);footprintTexture.colorSpace=THREE.SRGBColorSpace;footprintTexture.magFilter=THREE.LinearFilter;footprintTexture.needsUpdate=true;
    const footprintGeometry=new THREE.PlaneGeometry(extent*2,extent*2,30,30);footprintGeometry.rotateX(-Math.PI/2);
    this.footprintBase=footprintGeometry.attributes.position.array.slice();
    this.footprintFill=new THREE.Mesh(footprintGeometry,new THREE.MeshBasicMaterial({map:footprintTexture,transparent:true,depthWrite:false,side:THREE.DoubleSide,toneMapped:false,polygonOffset:true,polygonOffsetFactor:-3}));this.footprintFill.renderOrder=3;this.scene.add(this.footprintFill);
    this.sensorFrustum=new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position',new THREE.Float32BufferAttribute(new Float32Array(24),3)),new THREE.LineBasicMaterial({color:0xff6655,transparent:true,opacity:.4,toneMapped:false}));this.scene.add(this.sensorFrustum);
    this.sensorBeam=new THREE.Mesh(new THREE.BufferGeometry().setAttribute('position',new THREE.Float32BufferAttribute(new Float32Array(36),3)),new THREE.MeshBasicMaterial({color:0xff5146,transparent:true,opacity:.035,depthWrite:false,side:THREE.DoubleSide,toneMapped:false}));this.scene.add(this.sensorBeam);
    this.forceGroup=new THREE.Group();this.scene.add(this.forceGroup);
    this.forceArrows=[0xffae68,0x8ccfff].map(color=>{const arrow=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(),.1,color,.06,.035);this.forceGroup.add(arrow);return arrow;});
    this.windStreaks=new THREE.Mesh(new THREE.BufferGeometry().setAttribute('position',new THREE.Float32BufferAttribute(new Float32Array(120*18),3)),new THREE.MeshBasicMaterial({color:0xfff1d9,transparent:true,opacity:.58,depthWrite:false,side:THREE.DoubleSide,toneMapped:false}));this.scene.add(this.windStreaks);
    this.windSeeds=Array.from({length:120},(_,i)=>{const k=(i*487+93)%env.cells.length,p=env.cells[k],v=data.physical_wind[k];return {p,v,phase:(i*.618)%1,height:.24+(i%7)*.068};});
    this.windsocks=['roi_a','roi_b'].map(key=>{
      const [cx,cy]=env[key],x=cx+.5,y=cy-.42;
      const k=env.cells.reduce((best,p,i)=>Math.hypot(p[0]-x,p[1]-y)<Math.hypot(env.cells[best][0]-x,env.cells[best][1]-y)?i:best,0),[wx,wy]=data.physical_wind[k];
      const pole=new THREE.Mesh(new THREE.CylinderGeometry(.006,.009,.28,8),new THREE.MeshStandardMaterial({color:0xb9b7a7,roughness:.7}));pole.position.set(x,terrainHeight(x,y)+.14,-y);this.scene.add(pole);
      const flag=new THREE.Group();flag.position.set(x,terrainHeight(x,y)+.27,-y);flag.rotation.y=Math.atan2(wy,wx);
      for(let j=0;j<5;j++){const section=new THREE.Mesh(new THREE.CylinderGeometry(.038-j*.004,.034-j*.004,.046,12,1,true),new THREE.MeshStandardMaterial({color:j%2?0xe8e1c9:0xe5693b,roughness:.9,side:THREE.DoubleSide}));section.rotation.z=-Math.PI/2;section.position.x=j*.045+.022;flag.add(section);}this.scene.add(flag);
      return {flag,heading:Math.atan2(wy,wx),strength:Math.hypot(wx,wy)};
    });
    this.trail=line(data.mission.x,accent,.85);this.trail.geometry.setDrawRange(0,1);this.scene.add(this.trail);
    this.projection=line(data.mission.x,accent,.32,p=>terrainHeight(p[0],p[1])+.02);this.projection.geometry.setDrawRange(0,1);this.scene.add(this.projection);
    this.tip=line([[0,0],[0,0]],accent);this.scene.add(this.tip);this.forecasts=new THREE.Group();this.scene.add(this.forecasts);
    this.arrows=[];
    for(let iy=1;iy<this.shape.ny;iy+=4)for(let ix=1;ix<this.shape.nx;ix+=4){const k=iy*this.shape.nx+ix,p=env.cells[k];
      if(env.obstacles.some(o=>Math.hypot(o[0]-p[0],o[1]-p[1])<o[2]+.04))continue;
      const arrow=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(...toScene(p,terrainHeight(...p)+.14)),.1,0xcce8e4,.04,.027);
      for(const m of [arrow.line.material,arrow.cone.material]){m.transparent=true;m.opacity=.75;m.toneMapped=false;}
      this.scene.add(arrow);this.arrows.push({k,arrow});}
    this.observer=new ResizeObserver(()=>this.resize());this.observer.observe(host);this.resize();this.setView('overview');
  }
  resize(){const w=this.host.clientWidth,h=this.host.clientHeight;if(!w||!h)return;this.renderer.setSize(w,h,false);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();this.needsRender=true;}
  setView(view){this.controls.target.set(0,.25,0);this.camera.position.set(...(view==='top'?[0,8,.001]:view==='valley'?[11,9,6]:[7.4,6.3,3.0]));
    this.camera.position.sub(this.controls.target).multiplyScalar(Math.max(1,1.25/this.camera.aspect)).add(this.controls.target);this.controls.update();}
  syncCamera(other){this.camera.position.copy(other.camera.position);this.controls.target.copy(other.controls.target);this.camera.quaternion.copy(other.camera.quaternion);this.controls.update();}
  setFrame(frame,layer,wind){
    this.needsRender=true;this.currentFrame=frame;const {x,q,belief,index}=frame;
    this.flight.group.position.set(...toScene(x));if(Math.hypot(x[2],x[3])>.025)this.flight.group.rotation.y=Math.atan2(x[3],x[2]);
    this.forces=motionForces(this.data,index);
    const heading=this.flight.group.rotation.y,[ux,uy]=this.forces.control;
    const localX=ux*Math.cos(heading)+uy*Math.sin(heading),localY=-ux*Math.sin(heading)+uy*Math.cos(heading);
    // Illustrative bank/pitch responds to recorded control, never changes position.
    this.flight.body.rotation.z=THREE.MathUtils.clamp(-Math.atan(localX/9.81)*1.8,-.18,.18);
    this.flight.body.rotation.x=THREE.MathUtils.clamp(Math.atan(localY/9.81)*1.8,-.18,.18);
    this.footprintRings.forEach(ring=>{
      const a=ring.geometry.attributes.position,r=ring.userData.radius;
      for(let i=0;i<a.count;i++){const px=x[0]+Math.cos(i*Math.PI/32)*r,py=x[1]+Math.sin(i*Math.PI/32)*r;a.setXYZ(i,px,terrainHeight(px,py)+.04,-py);}
      a.needsUpdate=true;ring.geometry.computeBoundingSphere();
    });
    const fill=this.footprintFill.geometry.attributes.position;
    for(let i=0;i<fill.count;i++){const px=x[0]+this.footprintBase[i*3],py=x[1]-this.footprintBase[i*3+2];fill.setXYZ(i,px,terrainHeight(px,py)+.036,-py);}fill.needsUpdate=true;this.footprintFill.geometry.computeBoundingSphere();
    const frustum=this.sensorFrustum.geometry.attributes.position,r=sensorRadius(this.data.config.sensor_sigma);
    for(let i=0;i<4;i++){const px=x[0]+Math.cos(i*Math.PI/2)*r,py=x[1]+Math.sin(i*Math.PI/2)*r;frustum.setXYZ(i*2,x[0],FLIGHT_HEIGHT-.04,-x[1]);frustum.setXYZ(i*2+1,px,terrainHeight(px,py)+.04,-py);}frustum.needsUpdate=true;this.sensorFrustum.geometry.computeBoundingSphere();
    const beam=this.sensorBeam.geometry.attributes.position;for(let i=0;i<4;i++){beam.setXYZ(i*3,x[0],FLIGHT_HEIGHT-.04,-x[1]);for(let j=0;j<2;j++){const a=(i+j)*Math.PI/2,px=x[0]+Math.cos(a)*r,py=x[1]+Math.sin(a)*r;beam.setXYZ(i*3+j+1,px,terrainHeight(px,py)+.04,-py);}}beam.needsUpdate=true;this.sensorBeam.geometry.computeBoundingSphere();
    this.forceGroup.position.set(x[0],FLIGHT_HEIGHT+.13,-x[1]);
    this.forceArrows.forEach((arrow,i)=>{const v=i?this.forces.control:this.forces.wind,mag=Math.hypot(...v);arrow.visible=mag>.015;if(arrow.visible){const len=Math.min(mag*.20,.65);arrow.setDirection(new THREE.Vector3(v[0],0,-v[1]).normalize());arrow.setLength(len,Math.min(.06,len*.3),Math.min(.035,len*.2));}});
    this.trail.geometry.setDrawRange(0,index+1);this.projection.geometry.setDrawRange(0,index+1);
    const tip=this.tip.geometry.attributes.position;tip.setXYZ(0,...toScene(this.data.mission.x[index]));tip.setXYZ(1,...toScene(x));tip.needsUpdate=true;this.tip.geometry.computeBoundingSphere();
    const key=`${index}:${layer}`;
    if(key!==this.layerKey){q.forEach((v,k)=>{
      const target=this.data.environment.targets[k],value=layer==='deficit'?Math.max(0,target-v)/.7:layer==='uncertainty'?belief.std[k]:v;
      const color=fieldColor(value,layer),alpha=layer==='none'?0:layer==='regions'?(target>0?125:0):110;
      this.fieldBytes.set([...color,alpha],k*4);
    });this.fieldTexture.needsUpdate=true;this.layerKey=key;}
    const windKey=`${frame.beliefBoundary}:${wind}`;
    if(windKey!==this.windKey){const field=wind==='truth'?this.data.physical_wind:belief.mean;
      for(const {k,arrow} of this.arrows){const [u,v]=field[k],mag=Math.hypot(u,v);arrow.visible=wind!=='off'&&mag>.015;
        if(arrow.visible){const length=Math.min(mag*.22,.48);arrow.setDirection(new THREE.Vector3(u,0,-v).normalize());arrow.setLength(length,Math.min(.045,length*.3),Math.min(.028,length*.2));arrow.setColor(wind==='truth'?0xffbe7d:0xc4e6e7);}}
      this.windKey=windKey;
    }
  }
  setForecast(paths){disposeObjects(this.forecasts);for(const {points,color=this.accent,opacity=1,height=FLIGHT_HEIGHT} of paths)if(points?.length>1)this.forecasts.add(line(points,color,opacity,height));this.needsRender=true;}
  renderSensor(width,height){
    const x=this.currentFrame.x,h=terrainHeight(x[0],x[1]);
    this.sensorCamera.position.set(x[0],FLIGHT_HEIGHT-.04,-x[1]);this.sensorCamera.lookAt(x[0],h,-x[1]);
    this.sensorCamera.aspect=width/height;this.sensorCamera.fov=THREE.MathUtils.radToDeg(2*Math.atan(sensorRadius(this.data.config.sensor_sigma,.1)/(FLIGHT_HEIGHT-.04-h)));this.sensorCamera.updateProjectionMatrix();
    const hidden=[this.flight.group,this.forecasts,this.trail,this.projection,this.tip,this.sensorFrustum,this.sensorBeam,this.forceGroup,this.windStreaks,...this.regionLabels,...this.arrows.map(a=>a.arrow)],visible=hidden.map(o=>o.visible);
    hidden.forEach(o=>o.visible=false);this.renderer.shadowMap.autoUpdate=false;this.renderer.render(this.scene,this.sensorCamera);this.renderer.shadowMap.autoUpdate=true;
    hidden.forEach((o,i)=>o.visible=visible[i]);
  }
  render(delta,clock,spinning,follow,sensor=false,windFlow=true){
    if(!this.host.clientWidth||!this.host.clientHeight)return;
    this.regionLabels.forEach(s=>{s.visible=!follow&&!sensor;});this.controls.enabled=!sensor;
    if(spinning)this.flight.mixer?.update(delta);
    if(follow&&this.currentFrame){
      const heading=this.flight.group.rotation.y,rotation=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0),heading);
      const center=this.flight.group.position;
      const desired=new THREE.Vector3(...chasePosition(center.toArray(),heading,this.data.environment.obstacles));
      const target=new THREE.Vector3(.15,-.20,0).applyQuaternion(rotation).add(center);
      const amount=1-Math.exp(-delta*5);this.camera.position.lerp(desired,amount);this.controls.target.lerp(target,amount);
      if(!cameraSightClear(center.toArray(),this.camera.position.toArray(),this.data.environment.obstacles))this.camera.position.copy(desired);
    }
    if(this.currentFrame&&this.forces){
      const positions=this.windStreaks.geometry.attributes.position;
      this.windStreaks.visible=windFlow;
      // Moving direction cues only; acceleration is not interpreted as air velocity.
      const time=clock/1000;
      this.windSeeds.forEach(({p,v,phase,height},i)=>{const mag=Math.hypot(...v),dx=mag>1e-8?v[0]/mag:1,dz=mag>1e-8?-v[1]/mag:0;
        const progress=((phase+time*(.25+Math.min(mag,2)*.45))%1-.5)*1.3,px=p[0]+dx*progress,pz=-p[1]+dz*progress,length=.18+Math.min(mag,2)*.15;
        const direction=new THREE.Vector3(dx,0,dz),view=this.camera.position.clone().sub(new THREE.Vector3(px,height,pz)).normalize(),side=new THREE.Vector3().crossVectors(direction,view).normalize().multiplyScalar(.0035);
        const a=[px,height,pz],b=[px+dx*length*.5+side.x,height+side.y,pz+dz*length*.5+side.z],c=[px+dx*length,height,pz+dz*length],d=[px+dx*length*.5-side.x,height-side.y,pz+dz*length*.5-side.z];
        [a,b,c,a,c,d].forEach((point,j)=>positions.setXYZ(i*6+j,...point));
      });positions.needsUpdate=true;this.windStreaks.geometry.computeBoundingSphere();
      this.windsocks.forEach(({flag,heading,strength},i)=>{flag.rotation.y=heading+(windFlow?Math.sin(time*5+i)*.07:0);flag.rotation.z=-.1/(strength+.1)+(windFlow?Math.sin(time*7+i)*.035:0);});
    }
    this.controls.update();if(this.needsRender||spinning||windFlow){
      const w=this.host.clientWidth,h=this.host.clientHeight;this.renderer.setScissorTest(false);this.renderer.setViewport(0,0,w,h);
      if(sensor)this.renderSensor(w,h);else this.renderer.render(this.scene,this.camera);
      this.needsRender=false;
    }
  }
  dispose(){this.disposed=true;this.observer.disconnect();this.controls.dispose();this.flight.mixer?.stopAllAction();disposeObjects(this.scene);this.environment.dispose();this.renderer.dispose();this.renderer.domElement.remove();}
}
