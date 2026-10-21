import * as THREE from 'three';
import {Water} from 'three/addons/objects/Water.js';
const viewParameters=new URLSearchParams(location.search);
const initialGeometry=await fetch(`data/initial_${viewParameters.get('layout')==='multi'?'multi':'two'}.json`,{cache:'no-store'}).then(r=>r.json());

export const layout={
 scale:viewParameters.has('archive')?1:(initialGeometry.environment.horizontal_scale||1),
 inspectionScale:initialGeometry.environment.inspection_scale||1,
 name:'Canal Quarter',units:'metres',inspectionCenters:new URLSearchParams(location.search).get('layout')==='multi'?[[-28,8],[28,-10],[-28,-18],[28,27]]:[[-28,8],[28,-10]],
 passages:{A:[[-28,8],[25,8],[28,-10]],B:[[-28,8],[-25,-23],[25,-23],[28,-10]],C:[[-28,8],[-25,27],[25,27],[28,-10]]},
 flightHeight:3.2,droneSpan:.55,
 description:'Designed European neighborhood; not a reconstruction or a measured wind environment.'
};

export async function buildQuarter(scene,renderer){
 const rand=(()=>{let a=170315;return()=>{a=(Math.imul(a,1664525)+1013904223)>>>0;return a/4294967296}})();
 const loader=new THREE.TextureLoader(),maps={};
 await Promise.all(['brick','roof','slate','cobble','paving'].map(async n=>{
  const tx=await Promise.all(['diff','nor_gl','rough'].map(t=>loader.loadAsync(`assets/${n}_${t}.jpg`)));
  tx.forEach((t,i)=>{t.wrapS=t.wrapT=THREE.RepeatWrapping;t.anisotropy=Math.min(8,renderer.capabilities.getMaxAnisotropy());if(i===0)t.colorSpace=THREE.SRGBColorSpace});maps[n]=tx;
 }));
 const pbr=(n,color=0xffffff,extra={})=>new THREE.MeshStandardMaterial({map:maps[n][0],normalMap:maps[n][1],roughnessMap:maps[n][2],color,roughness:.91,normalScale:new THREE.Vector2(.55,.55),...extra});
 const brick=[0xe7d4bd,0xa9a7a3,0xc7a691,0x918f83,0xe2b9a1,0xbdaba3,0x8b8d8a].map(c=>pbr('brick',c));
 const plaster=[0xddd8c8,0xcccbbf,0xb6b7a9,0x9a9a8d].map(c=>new THREE.MeshStandardMaterial({color:c,roughness:.93,normalMap:maps.brick[1],normalScale:new THREE.Vector2(.06,.06)}));
 const roof=[pbr('roof',0xc0a295),pbr('slate',0xbbb8ad),pbr('slate',0x8f958e),pbr('roof',0xd8af98)];
 const cream=new THREE.MeshStandardMaterial({color:0xc9c4b8,roughness:.82});
 const limestone=new THREE.MeshStandardMaterial({color:0xb8b7a6,roughness:.93});
 const darkStone=new THREE.MeshStandardMaterial({color:0x777e77,roughness:.97});
 const metal=new THREE.MeshStandardMaterial({color:0x3c4846,metalness:.55,roughness:.56});
 const zinc=new THREE.MeshStandardMaterial({color:0x8e9995,metalness:.5,roughness:.65});
 const darkWood=new THREE.MeshStandardMaterial({color:0x304b43,roughness:.7});
 const browns=[0x463c31,0x4b4740,0x334743,0x56564d].map(c=>new THREE.MeshStandardMaterial({color:c,roughness:.7}));
 const glasses=[0x253d43,0x374b4c,0x425653,0x2d4146].map(c=>new THREE.MeshStandardMaterial({color:c,metalness:.35,roughness:.2,envMapIntensity:1.1}));
 const warmGlass=new THREE.MeshStandardMaterial({color:0x8c7855,emissive:0x4b3920,emissiveIntensity:.18,roughness:.38});
 const batches=new Map(),unitBox=new THREE.BoxGeometry(1,1,1),unitCylinder=new THREE.CylinderGeometry(1,1,1,8),mat=new THREE.Matrix4(),q=new THREE.Quaternion();
 const vector=new THREE.Vector3(),scale=new THREE.Vector3();
 function instance(geometry,material,size,pos,rot=[0,0,0]){
  const key=geometry.uuid+material.uuid;if(!batches.has(key))batches.set(key,{geometry,material,transforms:[]});
  q.setFromEuler(new THREE.Euler(...rot));mat.compose(vector.set(...pos),q,scale.set(...size));batches.get(key).transforms.push(mat.clone());
 }
 const box=(size,pos,material,rot=0)=>instance(unitBox,material,size,pos,[0,rot,0]);
 const cyl=(radius,height,pos,material)=>instance(unitCylinder,material,[radius,height,radius],pos);
 const staticMeshes=[];
 const mesh=(geo,material,pos=[0,0,0],rotation=0)=>{const m=new THREE.Mesh(geo,material);m.position.set(...pos);m.rotation.y=rotation;m.castShadow=m.receiveShadow=true;scene.add(m);staticMeshes.push(m);return m};
 function physicalUV(geo,tile=1.4){const p=geo.attributes.position,n=geo.attributes.normal,uv=geo.attributes.uv;for(let i=0;i<p.count;i++){let nx=Math.abs(n.getX(i)),ny=Math.abs(n.getY(i));let u=nx>.5?p.getZ(i):p.getX(i),v=ny>.5?p.getZ(i):p.getY(i);uv.setXY(i,u/tile,v/tile);}return geo}
 const ground=pbr('cobble',0x9da5a7,{map:null,roughness:1,normalScale:new THREE.Vector2(.16,.16)});
 const pavement=pbr('paving',0xb2b9b9,{map:null,roughness:1,normalScale:new THREE.Vector2(.12,.12)});
 function slab(x,z,w,d,material=ground,y=.01){const g=new THREE.PlaneGeometry(w,d);const u=g.attributes.uv;for(let i=0;i<u.count;i++)u.setXY(i,u.getX(i)*w/2,u.getY(i)*d/2);const m=mesh(g,material,[x,y,z]);m.rotation.x=-Math.PI/2;m.castShadow=false;return m}
 slab(0,-77.1,300,225.8,ground,-.12);
 slab(0,111,300,90,ground,-.12);
 slab(0,67,300,4,pavement,.06);
 slab(0,32,120,6,pavement,.06);
 // Narrow, physically dimensioned centerlines stay unblocked between the squares.
 for(const z of [8,-23,27])slab(0,z,66,z===8?8:9,ground,.02);
 for(const x of [-28,28])slab(x,2,15,58,ground,.025);
 for(const [x,z] of layout.inspectionCenters){slab(x,z,10,11,pavement,.04);for(const s of [-1,1])box([10,.08,.15],[x,.09,z+s*5.5],limestone)}
 const buildings=[];
 let id=0;
 function house(x,z,w,d,h,rotation=0,detail=true){
  const k=id++,floorCount=Math.max(2,Math.round((h-1)/3.05));h=floorCount*3.05+.65;
  const wall=k%9===1||k%13===0?plaster[k%plaster.length]:brick[k%brick.length];
  const rmat=roof[k%roof.length],rise=2.6+rand()*.8;
  const tr=(a,b,c)=>[x+a*Math.cos(rotation)+c*Math.sin(rotation),b,z-a*Math.sin(rotation)+c*Math.cos(rotation)];
  const bx=(sz,p,ma=cream)=>box(sz,tr(...p),ma,rotation);
  const cy=(r,len,p,ma)=>cyl(r,len,tr(...p),ma);
  buildings.push({x,z,width:w,depth:d,eaveHeight:h,totalHeight:h+rise,rotation});
  mesh(physicalUV(new THREE.BoxGeometry(w,h,d)),wall,[x,h/2,z],rotation);
  // Roof slopes retain metre-scaled tile UVs rather than stretching a texture over a block.
  const v=[-w/2-.16,h,-d/2-.2,0,h+rise,-d/2-.2,0,h+rise,d/2+.2,-w/2-.16,h,d/2+.2, 0,h+rise,-d/2-.2,w/2+.16,h,-d/2-.2,w/2+.16,h,d/2+.2,0,h+rise,d/2+.2];
  const rg=new THREE.BufferGeometry();rg.setAttribute('position',new THREE.Float32BufferAttribute(v,3));rg.setIndex([0,2,1,0,3,2,4,6,5,4,7,6]);
  const sl=Math.hypot(w/2,rise)/2.5,dd=(d+.4)/2.5;rg.setAttribute('uv',new THREE.Float32BufferAttribute([0,0,0,sl,dd,sl,dd,0,0,sl,0,0,dd,0,dd,sl],2));rg.computeVertexNormals();rmat.side=THREE.DoubleSide;mesh(rg,rmat,[x,0,z],rotation);
  for(const side of [-1,1]){
   const shape=new THREE.Shape();shape.moveTo(-w/2,h);shape.lineTo(w/2,h);shape.lineTo(0,h+rise);shape.closePath();
   const gg=new THREE.ShapeGeometry(shape);physicalUV(gg);const gm=mesh(gg,wall,tr(0,0,side*(d/2+.015)),rotation+(side<0?Math.PI:0));gm.material.side=THREE.DoubleSide;
   bx([w+.12,.19,.23],[0,h-.05,side*(d/2+.08)],cream);
   if(side===1&&k%5===0){
    const bell=new THREE.Shape();bell.moveTo(-w/2,h-.2);bell.lineTo(w/2,h-.2);bell.lineTo(w/2,h+.25);bell.bezierCurveTo(w*.45,h+.9,w*.19,h+.4,w*.16,h+2.45);bell.lineTo(w*.14,h+rise+.2);bell.lineTo(-w*.14,h+rise+.2);bell.bezierCurveTo(-w*.19,h+.4,-w*.45,h+.9,-w/2,h+.25);bell.closePath();
    mesh(physicalUV(new THREE.ShapeGeometry(bell)),wall,tr(0,0,d/2+.22),rotation);
    bx([w*.35,.12,.40],[0,h+rise+.25,d/2+.22],cream);
   }else if(side===1&&k%3!==1){
    for(let level=0;level<4;level++){const ww=w*(1-level*.23),yy=h+.22+level*.70;bx([ww,.74,.28],[0,yy,d/2+.08],wall);bx([ww+.10,.09,.31],[0,yy+.40,d/2+.12],cream)}
   }
  }
  for(const side of [-1,1]){
   const yy=h+1.05,face=side*(d/2+.25),ww=.58;
   bx([ww,.84,.07],[0,yy,face],glasses[k%4]);
   for(const a of [-1,1]){bx([.055,.95,.14],[a*.32,yy,face+side*.05],cream);bx([.72,.055,.14],[0,yy+a*.45,face+side*.05],cream);}
   bx([.04,.90,.14],[0,yy,face+side*.07],cream);
  }
  bx([w+.16,.35,d+.13],[0,.18,0],darkStone);
  const front=d/2+.07,back=-d/2-.07;
  const bays=Math.max(2,Math.floor(w/1.7));const step=(w-.65)/bays,ww=Math.min(1.07,step*.65);
  for(let floor=0;floor<floorCount;floor++)for(let bay=0;bay<bays;bay++){
   const xx=-w/2+.325+step*(bay+.5),yy=1.78+floor*3.05,hh=floor===0?2.04:1.77;
   for(const side of [1,-1]){
    const face=side===1?front:back;
    if(floor===0&&bay===0&&side===1){
     bx([ww+.2,2.52,.16],[xx,1.33,face],cream);bx([ww,2.38,.18],[xx,1.29,face+side*.05],darkWood);
     bx([ww*.7,.58,.025],[xx,2.05,face+side*.16],glasses[k%4]);bx([ww*.68,1.1,.035],[xx,1.08,face+side*.16],browns[k%4]);
     bx([.035,.18,.05],[xx+ww*.3,1.25,face+side*.2],zinc);bx([ww+.35,.17,.6],[xx,.09,face+side*.3],limestone);
     continue;
    }
    // Recess shadow, glass, projected frames, mullions and stone sills.
    bx([ww+.24,hh+.24,.065],[xx,yy,face],darkStone);
    bx([ww,hh,.08],[xx,yy,face+side*.035],(k+bay+floor)%11===0?warmGlass:glasses[(k+bay+floor)%4]);
    for(const s of [-1,1]){bx([.075,hh+.14,.13],[xx+s*(ww/2+.035),yy,face+side*.10],cream);bx([ww+.2,.075,.13],[xx,yy+s*(hh/2+.035),face+side*.10],cream)}
    bx([.045,hh,.13],[xx,yy,face+side*.12],cream);bx([ww,.055,.13],[xx,yy+.15,face+side*.12],cream);
    bx([ww+.38,.14,.33],[xx,yy-hh/2-.16,face+side*.14],cream);bx([ww+.34,.19,.14],[xx,yy+hh/2+.19,face+side*.08],limestone);
    if(detail&&(k+bay)%4===0&&floor>0)for(const s of [-1,1])bx([ww*.26,hh,.065],[xx+s*(ww*.72),yy,face],darkWood);
   }
  }
  // Thin gutters, rain pipes and chimney stacks replace oversized ornamental parts.
  for(const s of [-1,1]){bx([.14,.12,d+.4],[s*w/2,h-.1,0],zinc);cy(.055,h,[s*(w/2-.13),h/2,front+.11],zinc);}
  bx([.58,1.2,.65],[w*.24,h+rise-.3,-d*.19],brick[(k+1)%brick.length]);
  bx([.72,.12,.78],[w*.24,h+rise+.31,-d*.19],limestone);cy(.095,.48,[w*.24,h+rise+.58,-d*.19],browns[0]);
  if(detail&&k%2===0){
   bx([1.2,.90,1.05],[-w*.18,h+rise*.53,d*.1],wall);bx([1.38,.12,1.16],[-w*.18,h+rise*.53+.5,d*.1],zinc);
   bx([.62,.62,.08],[-w*.18,h+rise*.53,d*.1+.55],glasses[k%4]);
   for(const s of [-1,1])bx([.06,.72,.13],[-w*.18+s*.34,h+rise*.53,d*.1+.6],cream);
  }
  if(detail&&k%5===0){
   const aw=new THREE.Mesh(new THREE.PlaneGeometry(w*.59,1.05),new THREE.MeshStandardMaterial({color:k%2?0x607566:0xa88e71,side:THREE.DoubleSide,roughness:.93}));
   aw.position.set(...tr(w*.16,3.1,front+.44));aw.rotation.set(-Math.PI*.28,rotation,0);aw.castShadow=true;scene.add(aw);
   bx([w*.6,.3,.07],[w*.16,3.52,front+.1],darkWood);
  }
  // End walls need windows too: prominent on the squares and street corners.
  if(detail)for(const side of [-1,1])for(let f=1;f<floorCount;f++)for(let zz=-d/2+2;zz<d/2-1;zz+=2.9){
   bx([.08,1.6,.9],[side*(w/2+.045),1.8+f*3.05,zz],glasses[(k+f)%4]);
   bx([.18,.12,1.15],[side*(w/2+.1),.88+f*3.05,zz],cream);
   bx([.13,.06,.96],[side*(w/2+.1),1.90+f*3.05,zz],cream);
   bx([.13,1.64,.04],[side*(w/2+.1),1.8+f*3.05,zz],cream);
   for(const edge of [-1,1])bx([.13,1.72,.06],[side*(w/2+.1),1.8+f*3.05,zz+edge*.48],cream);
  }
 }
 function row(x0,x1,z,depth,h,rotation=0,detail=true){let x=x0;while(x<x1-.5){const w=Math.min(4.9+rand()*1.8,x1-x);if(w<1)break;house(x+w/2,z,w-.035,depth,h+(rand()-.5)*2.8,rotation,detail);x+=w;}}
 row(-19,19,-.875,12.75,13.2);row(-19,19,-13.625,12.75,13.8,Math.PI);
 row(-19,19,16.25,11.5,11.8,Math.PI);
 row(-40,40,-31,10,13.3);
 for(const side of [-1,1])for(let z=-23;z<31;z+=6){house(side*42,z,5.9,8,12+rand()*3.2,side<0?Math.PI/2:-Math.PI/2);}
 // Additional blocks extend the neighborhood beyond the study area.
 row(-90,90,-48,12,12.8,0,false);row(-90,90,-64,11,15.0,0,false);row(-110,110,-81,11,14,0,false);
 row(-85,85,77,12,12.7,Math.PI,false);
 for(const side of [-1,1])for(let z=-35;z<25;z+=6.5)house(side*57,z,6.42,10,13+rand()*2,side<0?Math.PI/2:-Math.PI/2,false);
 // Sidewalk lips follow the occupied building edges and leave the passages clear.
 for(const [cx,cz,w,d] of [[0,-7.25,38,25.5],[0,16.25,38,11.5],[0,-31,80,10]]){
  for(const s of [-1,1]){box([w+1.2,.12,.5],[cx,.06,cz+s*(d/2+.25)],limestone);box([.5,.12,d+1.2],[cx+s*(w/2+.25),.06,cz],limestone)}
 }
 // Quayside: stone retaining wall, cast-iron railing and a canal at the district edge.
 mesh(physicalUV(new THREE.BoxGeometry(145,2.5,.7),1.4),brick[1],[0,-1.12,35.8]);
 box([145,.24,.9],[0,.16,35.8],limestone);
 for(let x=-65;x<=65;x+=2.4){cyl(.035,.93,[x,.68,35.65],metal);box([2.4,.055,.055],[x+1.2,.97,35.65],metal);box([2.4,.045,.045],[x+1.2,.45,35.65],metal);for(let i=1;i<6;i++)box([.018,.67,.018],[x+i*.4,.65,35.65],metal)}
 const waterNormals=await loader.loadAsync('assets/waternormals.jpg');waterNormals.wrapS=waterNormals.wrapT=THREE.RepeatWrapping;
 const water=new Water(new THREE.PlaneGeometry(300,29.5),{textureWidth:512,textureHeight:512,waterNormals,sunDirection:new THREE.Vector3(-45,70,25).normalize(),sunColor:0xffe7bd,waterColor:0x1e3632,distortionScale:.9,alpha:1,fog:true});water.rotation.x=-Math.PI/2;water.position.set(0,-.6,50.6);scene.add(water);

 function boat(x,z,rot,color){const g=new THREE.Group();g.position.set(x,-.48,z);g.rotation.y=rot;scene.add(g);
  const hull=new THREE.Mesh(new THREE.SphereGeometry(1,24,10,0,Math.PI*2,Math.PI/2,Math.PI/2),new THREE.MeshStandardMaterial({color,roughness:.55,side:THREE.DoubleSide}));hull.scale.set(1.15,.7,3.6);g.add(hull);
  const deck=new THREE.Mesh(new THREE.BoxGeometry(1.85,.12,5.7),new THREE.MeshStandardMaterial({color:0x887253,roughness:.83}));deck.position.y=.03;g.add(deck);
  const cabin=new THREE.Mesh(new THREE.BoxGeometry(1.55,1.0,2.1),cream);cabin.position.set(0,.5,-.5);cabin.castShadow=true;g.add(cabin);
  const glass=new THREE.Mesh(new THREE.BoxGeometry(1.57,.53,2.13),glasses[0]);glass.position.set(0,.63,-.5);g.add(glass);
  const cover=new THREE.Mesh(new THREE.BoxGeometry(1.7,.12,2.3),darkWood);cover.position.set(0,1.06,-.5);g.add(cover);
 }
 boat(-24,39,.15,0x293f43);boat(19,39.6,-.12,0x635344);
 // Tree crowns consist of small alpha-cut leaves, not faceted spheres.
 const lc=document.createElement('canvas');lc.width=64;lc.height=64;const ctx=lc.getContext('2d');const lg=ctx.createRadialGradient(31,29,4,32,32,31);lg.addColorStop(0,'#a8b66b');lg.addColorStop(1,'#3e5e35');ctx.fillStyle=lg;ctx.beginPath();ctx.ellipse(32,32,18,30,.25,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#c2be7c';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(27,4);ctx.lineTo(37,61);ctx.stroke();
 const leafTex=new THREE.CanvasTexture(lc);leafTex.colorSpace=THREE.SRGBColorSpace;
 const leafMat=new THREE.MeshStandardMaterial({map:leafTex,alphaTest:.45,side:THREE.DoubleSide,roughness:.9});const leaves=[];
 function tree(x,z,h=8){cyl(.16,h*.62,[x,h*.31,z],browns[1]);for(let j=0;j<5;j++){const ang=j*1.256,branch=new THREE.Mesh(new THREE.CylinderGeometry(.04,.11,2.7,5),browns[1]);branch.position.set(x+Math.sin(ang)*.6,h*.58,z+Math.cos(ang)*.6);branch.rotation.z=Math.cos(ang)*.65;branch.rotation.x=Math.sin(ang)*.65;branch.castShadow=true;scene.add(branch)}
  for(let i=0;i<1300;i++){const ang=rand()*Math.PI*2,rad=Math.cbrt(rand())*2.4,y=(rand()-.5)*3.5;const p=[x+Math.cos(ang)*rad,h*.79+y,z+Math.sin(ang)*rad];q.setFromEuler(new THREE.Euler(rand()*Math.PI,rand()*Math.PI,rand()*Math.PI));mat.compose(new THREE.Vector3(...p),q,new THREE.Vector3(.22+rand()*.25,.28+rand()*.25,1));leaves.push(mat.clone());}
 }
 for(const [x,z,h] of [[-33,-14,9],[33,-13,9],[-33,24,8],[34,24,8],[-49,29,9],[49,29,9],[-28,32,8],[27,32,8],[-56,-26,9],[55,-26,9]])tree(x,z,h);
 const leafMesh=new THREE.InstancedMesh(new THREE.PlaneGeometry(1,1),leafMat,leaves.length);leaves.forEach((m,i)=>leafMesh.setMatrixAt(i,m));leafMesh.castShadow=true;leafMesh.receiveShadow=true;scene.add(leafMesh);
 // Street furniture establishes human scale without obstructing the flight corridors.
 function bench(x,z,rot=0){for(let n=0;n<5;n++)box([1.65,.055,.09],[x,.48,z+n*.10],browns[2],rot);for(const s of [-1,1]){box([.07,.48,.52],[x+s*.63,.24,z+.2],metal,rot);box([.07,.55,.07],[x+s*.63,.72,z+.46],metal,rot);}for(let n=0;n<3;n++)box([1.65,.10,.055],[x,.68+n*.15,z+.48],browns[2],rot)}
 for(const p of [[-34,14],[34,14],[-11,33],[9,33]])bench(...p);
 function lamp(x,z){cyl(.045,4.5,[x,2.25,z],metal);box([.27,.10,.27],[x,4.5,z],metal);box([.20,.36,.20],[x,4.29,z],new THREE.MeshStandardMaterial({color:0xe1dcc4,roughness:.5}));box([.36,.08,.36],[x,4.51,z],metal)}
 for(const p of [[-23,31],[23,31],[-23,-25],[23,-25],[-34,3],[34,3]])lamp(...p);
 function bicycle(x,z,rot=0){const g=new THREE.Group();g.position.set(x,.4,z);g.rotation.y=rot;scene.add(g);const rim=new THREE.TorusGeometry(.32,.016,6,24);for(const side of [-1,1]){const m=new THREE.Mesh(rim,metal);m.position.x=side*.51;m.castShadow=true;g.add(m)}const points=[[-.51,0,0],[-.15,.45,0],[.16,0,0],[.51,0,0],[.27,.5,0],[-.15,.45,0],[.16,0,0],[-.51,0,0]];for(let i=1;i<points.length;i++){const a=new THREE.Vector3(...points[i-1]),b=new THREE.Vector3(...points[i]);const m=new THREE.Mesh(new THREE.CylinderGeometry(.016,.016,a.distanceTo(b),6),darkWood);m.position.copy(a).add(b).multiplyScalar(.5);m.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),b.sub(a).normalize());g.add(m)}}
 bicycle(-22,34,.15);bicycle(-20,34,.2);bicycle(32,-4,Math.PI/2);bicycle(-34,17,Math.PI/2);
 // Flush all repeated details into instanced draws.
 for(const b of batches.values()){const im=new THREE.InstancedMesh(b.geometry,b.material,b.transforms.length);b.transforms.forEach((m,i)=>im.setMatrixAt(i,m));im.castShadow=im.receiveShadow=true;scene.add(im);}
 const mergeBuckets=new Map();
 for(const m of staticMeshes){m.updateMatrixWorld(true);const key=m.material.uuid+String(m.castShadow);if(!mergeBuckets.has(key))mergeBuckets.set(key,[]);mergeBuckets.get(key).push(m);}
 for(const group of mergeBuckets.values()){
  const positions=[],normals=[],uvs=[];
  for(const m of group){const g=m.geometry.clone().applyMatrix4(m.matrixWorld);const flat=g.index?g.toNonIndexed():g;positions.push(...flat.attributes.position.array);normals.push(...flat.attributes.normal.array);uvs.push(...flat.attributes.uv.array);scene.remove(m);g.dispose();}
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));g.setAttribute('normal',new THREE.Float32BufferAttribute(normals,3));g.setAttribute('uv',new THREE.Float32BufferAttribute(uvs,2));g.computeBoundingSphere();const m=new THREE.Mesh(g,group[0].material);m.castShadow=group[0].castShadow;m.receiveShadow=true;scene.add(m);
 }
 return {buildings,water,animate(t){water.material.uniforms.time.value=t*.22;},statistics:{buildings:buildings.length,instancedDetails:[...batches.values()].reduce((n,b)=>n+b.transforms.length,0),leaves:leaves.length}};
}
