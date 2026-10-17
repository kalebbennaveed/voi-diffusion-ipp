import * as THREE from 'three';
import {mergeGeometries} from './vendor/three/BufferGeometryUtils.js';
import {terrainHeight,riverDistance,noise,fbm,ridgeVertex} from './terrain-math.js';

function texture(loader,name,kind,repeat,onReady){
  const t=loader.load(`/3d/assets/${name}_${kind}_1k.jpg`,onReady);
  t.wrapS=t.wrapT=THREE.RepeatWrapping;t.repeat.set(repeat,repeat);t.anisotropy=4;
  if(kind==='diff')t.colorSpace=THREE.SRGBColorSpace;return t;
}
function landMaterial(name,repeat,onReady){
  const loader=new THREE.TextureLoader();return new THREE.MeshStandardMaterial({
    map:texture(loader,name,'diff',repeat,onReady),normalMap:texture(loader,name,'nor_gl',repeat,onReady),
    roughnessMap:texture(loader,name,'rough',repeat,onReady),roughness:1,metalness:0,
    normalScale:new THREE.Vector2(.65,.65),vertexColors:true
  });
}
function surface(width,depth,nx,ny,height){
  const g=new THREE.PlaneGeometry(width,depth,nx,ny);g.rotateX(-Math.PI/2);
  const p=g.attributes.position,colors=new Float32Array(p.count*3);
  for(let i=0;i<p.count;i++){
    const x=p.getX(i),y=-p.getZ(i),h=height(x,y);p.setY(i,h);
    const shade=.68+.38*fbm(x*1.2+8,y*1.2);colors.set([shade*.92,shade,shade*.82],i*3);
  }
  g.setAttribute('color',new THREE.BufferAttribute(colors,3));g.computeVertexNormals();return g;
}
function ridge(obstacle,index,mat,distant=false){
  const positions=[],uv=[],colors=[],indices=[],segments=96,levels=7;
  for(let j=0;j<levels;j++)for(let a=0;a<=segments;a++){
    const angle=a/segments*Math.PI*2,[x,h,y]=ridgeVertex(obstacle,j,angle,index);
    const base=terrainHeight(obstacle[0],obstacle[1]);
    // Seat the outer skirt below the local hillside, rather than scaling its base upward.
    const elevation=distant?terrainHeight(x,y)-.12+(h-base)*(1.2+index*.03):h;
    positions.push(x,elevation,-y);uv.push(a/segments*2,h*1.35);
    const shade=.66+.35*noise(a*.25+index,j*.7);
    colors.push(shade,shade*.99,shade*.9);
    if(j<levels-1&&a<segments){const n=j*(segments+1)+a;indices.push(n,n+1,n+segments+1,n+1,n+segments+2,n+segments+1);}
  }
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));g.setAttribute('uv',new THREE.Float32BufferAttribute(uv,2));g.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));g.setIndex(indices);g.computeVertexNormals();
  const m=new THREE.Mesh(g,mat);m.castShadow=true;m.receiveShadow=true;return m;
}

function wetland(center,index){
  const group=new THREE.Group(),[x,y]=center;
  const patch=(cx,cy,rx,ry,material,lift)=>{
    const p=[cx,terrainHeight(cx,cy)+lift,-cy],uv=[.5,.5],indices=[];
    for(let i=0;i<=64;i++){
      const a=i*Math.PI/32,r=.84+.12*noise(Math.cos(a)*3+index,Math.sin(a)*3+index);
      const px=cx+rx*Math.cos(a)*r,py=cy+ry*Math.sin(a)*r;
      p.push(px,terrainHeight(px,py)+lift,-py);uv.push(.5+Math.cos(a)*.5,.5+Math.sin(a)*.5);
      if(i<64)indices.push(0,i+1,i+2);
    }
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(p,3));g.setAttribute('uv',new THREE.Float32BufferAttribute(uv,2));g.setIndex(indices);g.computeVertexNormals();
    const m=new THREE.Mesh(g,material);m.receiveShadow=true;group.add(m);
  };
  const mud=new THREE.MeshStandardMaterial({color:0x5e6046,roughness:1,side:THREE.DoubleSide});
  const water=new THREE.MeshPhysicalMaterial({color:0x263f38,roughness:.34,clearcoat:.6,side:THREE.DoubleSide});
  patch(x-.18,y+.20,.43,.22,mud,.012);patch(x-.18,y+.20,.34,.15,water,.02);
  patch(x+.28,y-.22,.29,.16,mud,.012);patch(x+.28,y-.22,.22,.10,water,.02);
  const reedMat=new THREE.MeshStandardMaterial({color:0x767342,roughness:1}),reedGeo=new THREE.ConeGeometry(.007,.09,3);
  for(let k=0;k<75;k++){
    const a=k*2.399,r=.9+.12*noise(k,index),px=x-.18+Math.cos(a)*.40*r,py=y+.20+Math.sin(a)*.20*r;
    const reed=new THREE.Mesh(reedGeo,reedMat);reed.position.set(px,terrainHeight(px,py)+.045,-py);reed.rotation.z=(noise(k+10,index)-.5)*.3;group.add(reed);
  }
  // Static survey gauge and short timber access deck: context, not new hazards.
  const timber=new THREE.MeshStandardMaterial({color:0x85806b,roughness:1});
  for(let k=0;k<7;k++){
    const px=x+.25+k*.025,py=y+.34;
    const plank=new THREE.Mesh(new THREE.BoxGeometry(.022,.014,.14),timber);plank.position.set(px,terrainHeight(px,py)+.027,-py);plank.receiveShadow=true;group.add(plank);
  }
  for(let k=0;k<6;k++){
    const post=new THREE.Mesh(new THREE.BoxGeometry(.018,.024,.018),new THREE.MeshStandardMaterial({color:k%2?0xe2dfc6:0x965d42,roughness:1}));
    post.position.set(x+.27,terrainHeight(x+.27,y+.25)+.012+k*.024,-y-.25);group.add(post);
  }
  return group;
}

function pumpStation(obstacle,index){
  const [x,y,r]=obstacle,group=new THREE.Group(),scale=r/.5;
  // The foundation circumradius is 0.478; roof eaves reach 0.482, both below 0.5.
  // All elevated geometry remains within the recorded circular safety envelope.
  group.position.set(x,terrainHeight(x,y),-y);group.rotation.y=Math.atan2(.6,1);group.scale.setScalar(scale);
  const textureCanvas=document.createElement('canvas');textureCanvas.width=textureCanvas.height=128;
  const ctx=textureCanvas.getContext('2d'),pixels=ctx.createImageData(128,128);
  for(let k=0;k<128*128;k++){const v=185+Math.round(noise(k%128,Math.floor(k/128))*24);pixels.data.set([v,v,v-5,255],k*4);}ctx.putImageData(pixels,0,0);
  ctx.strokeStyle='#a9aaa1';ctx.lineWidth=.7;for(let i=0;i<4;i++){ctx.beginPath();ctx.moveTo(0,i*32);ctx.lineTo(128,i*32);ctx.stroke();}
  const concreteTexture=new THREE.CanvasTexture(textureCanvas);concreteTexture.colorSpace=THREE.SRGBColorSpace;concreteTexture.wrapS=concreteTexture.wrapT=THREE.RepeatWrapping;concreteTexture.repeat.set(2,2);
  const concrete=new THREE.MeshStandardMaterial({map:concreteTexture,color:0xd2d1bf,roughness:.95});
  const steel=new THREE.MeshStandardMaterial({color:0x63716d,metalness:.45,roughness:.67});
  const roofMat=new THREE.MeshStandardMaterial({color:index?0x4e5b5b:0x58656c,metalness:.45,roughness:.68,side:THREE.DoubleSide});
  const glass=new THREE.MeshStandardMaterial({color:0x203b45,metalness:.3,roughness:.28});
  const trim=new THREE.MeshStandardMaterial({color:0xa3aaa5,metalness:.25,roughness:.65});
  const box=(w,h,d,px,py,pz,material)=>{const mesh=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),material);mesh.position.set(px,py,pz);mesh.castShadow=true;mesh.receiveShadow=true;group.add(mesh);return mesh;};
  box(.76,.12,.58,0,.005,0,concrete);
  box(.62,.66,.42,-.035,.39,0,concrete);
  box(.18,.26,.34,.29,.19,0,concrete);
  box(.20,.025,.36,.29,.33,0,roofMat);
  const roofPositions=[-.38,.72,-.27,.31,.72,-.27,.31,.85,0,-.38,.85,0,-.38,.85,0,.31,.85,0,.31,.72,.27,-.38,.72,.27];
  const roofGeometry=new THREE.BufferGeometry();roofGeometry.setAttribute('position',new THREE.Float32BufferAttribute(roofPositions,3));roofGeometry.setIndex([0,1,2,0,2,3,4,5,6,4,6,7]);roofGeometry.computeVertexNormals();
  const roof=new THREE.Mesh(roofGeometry,roofMat);roof.castShadow=true;roof.receiveShadow=true;group.add(roof);
  // Gable infill, eaves and standing seams give the roof a believable construction.
  for(const side of [-.346,.276]){
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute([side,.72,-.21,side,.85,0,side,.72,.21],3));g.computeVertexNormals();const m=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:0xb6bbb2,roughness:.9,side:THREE.DoubleSide}));group.add(m);
  }
  box(.71,.022,.025,-.035,.719,-.27,trim);box(.71,.022,.025,-.035,.719,.27,trim);
  const seamGeo=new THREE.BufferGeometry(),seams=[];
  for(let k=0;k<12;k++){const px=-.37+k*.06;seams.push(px,.724,-.27,px,.853,0,px,.853,0,px,.724,.27);}
  seamGeo.setAttribute('position',new THREE.Float32BufferAttribute(seams,3));group.add(new THREE.LineSegments(seamGeo,new THREE.LineBasicMaterial({color:0x8a9592,transparent:true,opacity:.65})));
  for(const side of [-1,1])for(const px of [-.22,-.04,.14]){
    box(.115,.10,.014,px,.55,side*.217,trim);box(.094,.078,.016,px,.55,side*.222,glass);
    box(.004,.078,.018,px,.55,side*.225,trim);
  }
  box(.125,.225,.014,-.14,.175,.218,steel);box(.008,.008,.015,-.10,.17,.231,trim);
  box(.14,.045,.17,-.14,.075,.28,concrete);
  for(let k=0;k<7;k++)box(.012,.012,.22,.386,.12+k*.025,0,steel);
  const signCanvas=document.createElement('canvas');signCanvas.width=256;signCanvas.height=64;const signCtx=signCanvas.getContext('2d');signCtx.fillStyle='#3a5654';signCtx.fillRect(0,0,256,64);signCtx.fillStyle='#eef0df';signCtx.font='500 22px sans-serif';signCtx.textAlign='center';signCtx.fillText('RIVER PUMP STATION',128,28);signCtx.font='16px sans-serif';signCtx.fillText(index?'SOUTH WORKS':'NORTH WORKS',128,51);
  const signTexture=new THREE.CanvasTexture(signCanvas);signTexture.colorSpace=THREE.SRGBColorSpace;
  box(.27,.07,.008,.005,.36,.22,new THREE.MeshStandardMaterial({map:signTexture,roughness:.8}));
  box(.07,.08,.07,-.22,.83,-.07,steel);
  return group;
}

// Dispersed alpine farmsteads follow the wider hillsides, not the mission boundary.
// Static meshes are merged by material so extra detail adds only a few draw calls.
function settlement(environment){
  const [lo,hi]=environment.bounds,group=new THREE.Group(),lots=[],lanes=[],batches=new Map();
  const material=(color,metalness=0)=>new THREE.MeshStandardMaterial({color,roughness:metalness?.65:.92,metalness});
  const materials={stone:material(0x8d8e82),trim:material(0xc3bfae),glass:material(0x33494c,.2),wood:material(0x67503b),road:material(0x656453),rubber:material(0x30352e)};
  const walls=[0xc0b69c,0xb3ada0,0x9d9c8b,0x9f8a70].map(c=>material(c));
  const roofs=[0x606e70,0x765c4c,0x59615c].map(c=>material(c,.25));
  function add(geometry,mat,matrix){
    if(geometry.index){const old=geometry;geometry=geometry.toNonIndexed();old.dispose();}
    geometry.deleteAttribute('uv');if(matrix)geometry.applyMatrix4(matrix);
    if(!batches.has(mat))batches.set(mat,[]);batches.get(mat).push(geometry);
  }
  const pose=new THREE.Object3D();
  function box(w,h,d,x,y,z,mat,parent){
    pose.position.set(x,y,z);pose.rotation.set(0,0,0);pose.scale.set(1,1,1);pose.updateMatrix();
    const transform=parent?new THREE.Matrix4().multiplyMatrices(parent,pose.matrix):pose.matrix.clone();
    add(new THREE.BoxGeometry(w,h,d),mat,transform);
  }
  function ribbon(points,width){
    const vertices=[],indices=[];
    for(let i=0;i<points.length;i++){
      const [x,y]=points[i],a=points[Math.max(0,i-1)],b=points[Math.min(points.length-1,i+1)],dx=b[0]-a[0],dy=b[1]-a[1],length=Math.hypot(dx,dy)||1;
      for(const side of [-1,1]){const px=x-dy/length*width*.5*side,py=y+dx/length*width*.5*side;vertices.push(px,terrainHeight(px,py)+.013,-py);}
      if(i<points.length-1){const k=i*2;indices.push(k,k+2,k+1,k+1,k+2,k+3);}
    }
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));g.setIndex(indices);g.computeVertexNormals();add(g,materials.road);lanes.push({points,width});
  }
  function track(points,width){
    const curve=new THREE.CatmullRomCurve3(points.map(([x,y])=>new THREE.Vector3(x,0,y)));
    const samples=curve.getPoints(Math.max(18,points.length*24)).map(p=>[p.x,p.z]);
    ribbon(samples,width);return samples;
  }
  const routes=[
    [[-8.7,.2],[-6.9,1.0],[-5.5,1.6],[-4.1,2.6],[-3.0,3.7],[-1.7,4.3],[.2,4.9],[2.2,5.9],[4.4,6.7]],
    [[-8.6,-5.4],[-6.3,-5.1],[-4.8,-4.4],[-3.4,-4.0],[-1.6,-3.7],[.2,-3.6],[2.1,-3.2],[3.6,-2.0],[5.2,-.8],[6.6,.4],[8.6,2.0]],
    [[-7.4,1.0],[-7.6,2.6],[-6.8,3.9],[-5.3,4.9],[-4.4,6.7]],
    [[3.6,3.2],[4.8,3.9],[5.5,5.0],[7.2,5.9],[8.3,7.6]],
    [[-1.6,-3.7],[-1.1,-5.0],[.4,-6.2],[2.4,-6.6],[4.1,-5.8],[5.5,-4.8],[7.5,-3.8]],
  ].map(points=>track(points,.070));
  const routePoints=routes.flat();
  const nearestTrack=(x,y)=>routePoints.reduce((best,p)=>Math.hypot(p[0]-x,p[1]-y)<Math.hypot(best[0]-x,best[1]-y)?p:best);

  // Deliberately irregular, separated holdings across the whole landscape.
  // Nearby pairs form small farms; broad meadows separate the next group.
  const candidates=[
    [-7.5,.55],[-6.3,1.75],[-4.9,2.55],[-3.85,2.95],[-2.5,4.35],[-1.55,4.85],[.35,5.55],[2.7,6.55],
    [-7.85,3.0],[-6.55,4.5],[-5.15,5.45],
    [-7.1,-4.65],[-5.6,-5.5],[-4.7,-4.95],[-3.2,-3.35],[-1.55,-4.35],[-.25,-3.05],[.65,-4.15],[2.6,-3.65],
    [3.75,-2.65],[4.8,-1.15],[5.65,-.15],[6.8,.95],[8.0,1.2],
    [3.7,3.75],[4.9,4.55],[6.5,5.65],[7.5,6.6],
    [-.65,-5.45],[.7,-6.8],[2.25,-6.1],[4.4,-5.2],[6.2,-4.65],
  ];
  candidates.forEach(([seedX,seedY],i)=>{
    const barn=i%5===2,w=barn?.53:.36+(i%4)*.027,d=barn?.33:.29+(i%3)*.018,h=barn?.21:.24+(i%3)*.035;
    const near=nearestTrack(seedX,seedY),angle=Math.atan2(near[0]-seedX,seedY-near[1])+(noise(i,14)-.5)*.6;
    const c=Math.cos(angle),sn=Math.sin(angle),corners=[[-w/2,-d/2],[-w/2,d/2],[w/2,-d/2],[w/2,d/2],[0,0]];
    let placement=null;
    // Find a gentle nearby bench. Stone foundations reach the downhill ground.
    for(let k=0;k<97;k++){
      const radius=k?Math.sqrt(k/96)*.85:0,a=k*2.399,x=seedX+Math.cos(a)*radius,y=seedY+Math.sin(a)*radius;
      if((x>lo[0]-.55&&x<hi[0]+.55&&y>lo[1]-.55&&y<hi[1]+.55)||Math.abs(riverDistance(x,y))<.40)continue;
      if(lots.some(l=>Math.hypot(x-l.x,y-l.y)<l.r+.28))continue;
      const heights=corners.map(([dx,dz])=>terrainHeight(x+c*dx+sn*dz,y+sn*dx-c*dz));
      const low=Math.min(...heights),high=Math.max(...heights),score=high-low+radius*.025;
      if(!placement||score<placement.score)placement={x,y,low,high,score};
    }
    if(!placement||placement.high-placement.low>.18)return;
    const {x,y,low,high}=placement;
    const transform=new THREE.Object3D();transform.position.set(x,high,-y);transform.rotation.y=angle;transform.updateMatrix();const m=transform.matrix;
    lots.push({x,y,r:Math.hypot(w/2+.07,d/2+.13)+.21});
    box(w+.06,high-low+.11,d+.06,0,-(high-low)/2-.015,0,materials.stone,m);
    box(w,h,d,0,h/2+.04,0,barn?materials.wood:walls[i%walls.length],m);
    if(!barn)box(w+.006,h*.38,d+.006,0,h*.81+.04,0,materials.wood,m);
    const eave=h+.04,ridge=eave+(barn?.14:.18),half=w/2+.06,depth=d/2+.07;
    const roof=new THREE.BufferGeometry();roof.setAttribute('position',new THREE.Float32BufferAttribute([
      -half,eave,-depth,half,eave,-depth,half,ridge,0,-half,ridge,0,
      -half,ridge,0,half,ridge,0,half,eave,depth,-half,eave,depth],3));
    roof.setIndex([0,2,1,0,3,2,4,6,5,4,7,6]);roof.computeVertexNormals();add(roof,roofs[i%roofs.length],m);
    for(const side of [-1,1]){
      const gable=new THREE.BufferGeometry();gable.setAttribute('position',new THREE.Float32BufferAttribute([side*w/2,eave,-d/2,side*w/2,ridge,0,side*w/2,eave,d/2],3));
      if(side===-1)gable.setIndex([2,1,0]);gable.computeVertexNormals();add(gable,materials.wood,m);
      box(w+.08,.016,.015,0,eave,side*depth,materials.trim,m);
      for(const wx of [-w*.29,w*.29]){
        box(.087,.084,.014,wx,h*.68,side*(d/2+.004),materials.trim,m);
        box(.068,.064,.017,wx,h*.68,side*(d/2+.009),materials.glass,m);
        box(.005,.064,.018,wx,h*.68,side*(d/2+.01),materials.trim,m);
        if(!barn)for(const shutter of [-1,1])box(.022,.079,.012,wx+shutter*.055,h*.68,side*(d/2+.012),materials.wood,m);
      }
    }
    box(barn?.16:.065,h*.64,.015,0,h*.32+.04,d/2+.011,materials.wood,m);
    box(.12,.026,.07,0,.035,d/2+.04,materials.stone,m);
    if(!barn){
      box(.038,.10,.042,-w*.3,ridge-.015,-.05,materials.stone,m);
      // Timber balcony with open railing under broad roof eaves.
      box(w*.79,.018,.10,0,h*.50,d/2+.045,materials.wood,m);
      box(w*.79,.012,.012,0,h*.50+.065,d/2+.09,materials.wood,m);
      for(let post=0;post<7;post++)box(.009,.064,.009,(post/6-.5)*w*.76,h*.50+.032,d/2+.09,materials.wood,m);
    }
    const frontX=Math.sin(angle),frontY=-Math.cos(angle);
    const entrance=[x+frontX*(d/2+.1),y+frontY*(d/2+.1)],access=nearestTrack(x,y);
    track([entrance,[(entrance[0]+access[0])/2+.07*Math.cos(angle),(entrance[1]+access[1])/2+.07*Math.sin(angle)],access],.043);
    // A parked utility vehicle at alternate lots provides scale without synthetic motion.
    if(i%2===0){
      box(.12,.039,.057,w*.38,.055,d/2+.105,roofs[(i+1)%3],m);
      box(.060,.031,.051,w*.38+.005,.084,d/2+.105,materials.glass,m);
      for(const dx of [-.038,.038])for(const dz of [-.03,.03])box(.023,.025,.012,w*.38+dx,.031,d/2+.105+dz,materials.rubber,m);
    }
  });
  for(const [mat,geometries] of batches){
    const merged=mergeGeometries(geometries);geometries.forEach(g=>g.dispose());
    const mesh=new THREE.Mesh(merged,mat);mesh.castShadow=true;mesh.receiveShadow=true;group.add(mesh);
  }
  const excludes=(x,y)=>lots.some(l=>Math.hypot(x-l.x,y-l.y)<l.r)||lanes.some(l=>{
    return l.points.some((a,i)=>{
      if(!i)return false;const b=l.points[i-1],dx=b[0]-a[0],dy=b[1]-a[1],length2=dx*dx+dy*dy;
      const t=length2?Math.max(0,Math.min(1,((x-a[0])*dx+(y-a[1])*dy)/length2)):0;
      return Math.hypot(x-a[0]-t*dx,y-a[1]-t*dy)<l.width/2+.10;
    });
  });
  return {group,excludes};
}

// Branch clusters, not solid cone trees. A shared detailed mesh is instanced over the valley.
function pineGeometry(){
  const positions=[],uv=[];
  const quad=(a,b,c,d)=>{
    positions.push(...a,...b,...c,...a,...c,...d);
    // The first needle sprig in the photographic atlas; avoid the cone/wood islands.
    uv.push(.008,.56,.205,.56,.205,.995,.008,.56,.205,.995,.008,.995);
  };
  for(let layer=0;layer<13;layer++){
    const y=.16+layer*.061,r=(1-y)*.43;
    for(let b=0;b<9;b++){
      const angle=b/9*Math.PI*2+layer*2.4,length=r*(.8+.2*noise(layer,b));
      for(let twig=0;twig<3;twig++){
        const t=(twig+.3)/3,cx=Math.cos(angle)*length*t,cz=Math.sin(angle)*length*t;
        const yy=y-.055*t,w=.11*(1-y*.55),l=.18*(1-y*.4);
        const tx=-Math.sin(angle)*w,tz=Math.cos(angle)*w,dx=Math.cos(angle)*l,dz=Math.sin(angle)*l;
        quad([cx-tx,yy,cz-tz],[cx+tx,yy,cz+tz],[cx+dx+tx*.7,yy+.028,cz+dz+tz*.7],[cx+dx-tx*.7,yy+.028,cz+dz-tz*.7]);
        // Crossed needle clusters retain volume from a low chase-camera angle.
        quad([cx,yy-w,cz],[cx,yy+w,cz],[cx+dx,yy+w*.6,cz+dz],[cx+dx,yy-w*.6,cz+dz]);
      }
    }
  }
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));g.setAttribute('uv',new THREE.Float32BufferAttribute(uv,2));g.computeVertexNormals();return g;
}

export function makeLandscape(environment,onReady){
  const group=new THREE.Group();
  const groundMat=landMaterial('aerial_grass_rock',24,onReady);
  const ground=new THREE.Mesh(surface(56,56,480,480,terrainHeight),groundMat);ground.receiveShadow=true;group.add(ground);
  environment.obstacles.forEach((o,i)=>group.add(pumpStation(o,i)));
  const pipeMat=new THREE.MeshStandardMaterial({color:0x687d79,metalness:.4,roughness:.65});
  for(const [x,y,r] of environment.obstacles){
    const qx=(x+.6*y)/1.36,qy=.6*qx,dx=qx-x,dy=qy-y,distance=Math.hypot(dx,dy);
    const start=[x+dx/distance*r*.7,y+dy/distance*r*.7],end=[qx-dx/distance*.11,qy-dy/distance*.11];
    const points=Array.from({length:15},(_,i)=>{const t=i/14,px=start[0]+(end[0]-start[0])*t,py=start[1]+(end[1]-start[1])*t;return new THREE.Vector3(px,terrainHeight(px,py)+.055,-py);});
    const pipe=new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points),20,.022,8,false),pipeMat);pipe.castShadow=true;group.add(pipe);
    const pathPositions=[],indices=[];for(let i=0;i<=14;i++){const t=i/14,px=start[0]+(end[0]-start[0])*t,py=start[1]+(end[1]-start[1])*t;for(const side of [-1,1]){const ax=px+dy/distance*(.075+side*.035),ay=py-dx/distance*(.075+side*.035);pathPositions.push(ax,terrainHeight(ax,ay)+.012,-ay);}if(i<14){const n=i*2;indices.push(n,n+1,n+2,n+1,n+3,n+2);}}
    const pathGeometry=new THREE.BufferGeometry();pathGeometry.setAttribute('position',new THREE.Float32BufferAttribute(pathPositions,3));pathGeometry.setIndex(indices);pathGeometry.computeVertexNormals();group.add(new THREE.Mesh(pathGeometry,new THREE.MeshStandardMaterial({color:0x989688,roughness:1,side:THREE.DoubleSide})));
  }
  ['roi_a','roi_b'].forEach((key,i)=>group.add(wetland(environment[key],i)));

  const riverPositions=[],riverUvs=[],riverIndices=[];
  for(let i=0;i<=200;i++){
    const x=-11+i*.11,center=.6*x+.10*Math.sin(x*1.7),w=.085+.016*Math.sin(x*2.5);
    for(const side of [-1,1]){riverPositions.push(x,-.024,-center-side*w);riverUvs.push(i*.035,(side+1)/2);}
    if(i<200){const n=i*2;riverIndices.push(n,n+1,n+2,n+1,n+3,n+2);}
  }
  const waterGeometry=new THREE.BufferGeometry();waterGeometry.setAttribute('position',new THREE.Float32BufferAttribute(riverPositions,3));waterGeometry.setAttribute('uv',new THREE.Float32BufferAttribute(riverUvs,2));waterGeometry.setIndex(riverIndices);waterGeometry.computeVertexNormals();
  const waterMaterial=new THREE.MeshPhysicalMaterial({color:0x193128,roughness:.4,metalness:0,envMapIntensity:.35,clearcoat:.9,clearcoatRoughness:.14,side:THREE.DoubleSide});
  const water=new THREE.Mesh(waterGeometry,waterMaterial);water.receiveShadow=true;group.add(water);

  const village=settlement(environment);group.add(village.group);

  let seed=802;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  const trees=[],trunks=[];for(let i=0;i<1200;i++){
    const x=(random()-.5)*18,y=(random()-.5)*18;
    if(Math.abs(riverDistance(x,y))<.15||village.excludes(x,y))continue;
    if(environment.obstacles.some(o=>Math.hypot(x-o[0],y-o[1])<o[2]*1.08))continue;
    const inField=Math.abs(x)<2.75&&Math.abs(y)<2.15;
    const inRegion=['roi_a','roi_b'].some(k=>Math.abs(x-environment[k][0])<.9&&Math.abs(y-environment[k][1])<.75);
    if(inRegion&&random()<.86)continue;
    const height=inField?.14+random()*.15:.25+random()*.45;
    const m=new THREE.Object3D();m.position.set(x,terrainHeight(x,y),-y);m.scale.set(height,height,height);m.rotation.y=random()*Math.PI*2;m.updateMatrix();trees.push(m.matrix.clone());
    const t=new THREE.Object3D();t.position.set(x,m.position.y+height*.44,-y);t.scale.set(height*.018,height*.88,height*.018);t.updateMatrix();trunks.push(t.matrix.clone());
  }
  const loader=new THREE.TextureLoader(),foliage=loader.load('/3d/assets/pine_tree_01_twig_diff_1k.jpg',onReady),alpha=loader.load('/3d/assets/pine_tree_01_twig_alpha_1k.jpg',onReady);
  foliage.colorSpace=THREE.SRGBColorSpace;
  const needles=new THREE.InstancedMesh(pineGeometry(),new THREE.MeshStandardMaterial({map:foliage,alphaMap:alpha,alphaTest:.22,color:0xb3c294,roughness:.95,side:THREE.DoubleSide}),trees.length);
  trees.forEach((m,i)=>needles.setMatrixAt(i,m));needles.castShadow=true;needles.receiveShadow=true;group.add(needles);
  const stems=new THREE.InstancedMesh(new THREE.CylinderGeometry(.7,1,1,5),new THREE.MeshStandardMaterial({color:0x4f4434,roughness:1}),trunks.length);
  trunks.forEach((m,i)=>stems.setMatrixAt(i,m));stems.castShadow=true;group.add(stems);

  // Distant ridgelines give a horizon without changing the modeled flight domain.
  const farMaterial=landMaterial('rock_face_03',4,onReady);
  for(let k=0;k<12;k++){
    const a=k*Math.PI/6,r=9+random()*2;
    const mountain=ridge([Math.cos(a)*r,Math.sin(a)*r,1.8+random()],k+3,farMaterial,true);
    group.add(mountain);
  }
  return {group,water};
}

export function overlayGeometry(environment,nx,ny){
  const [lo,hi]=environment.bounds,cx=(lo[0]+hi[0])/2,cy=(lo[1]+hi[1])/2;
  const g=surface(hi[0]-lo[0],hi[1]-lo[1],nx,ny,(x,y)=>terrainHeight(x+cx,y+cy)+.009);
  g.translate(cx,0,-cy);return g;
}
