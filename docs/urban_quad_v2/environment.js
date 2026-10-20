import * as THREE from 'three';

function surface(base,seed=9,metal=false){
 const canvas=document.createElement('canvas');canvas.width=canvas.height=512;const ctx=canvas.getContext('2d');
 let n=seed;const random=()=>{n=(1664525*n+1013904223)>>>0;return n/4294967296};
 const img=ctx.createImageData(512,512);for(let i=0;i<img.data.length;i+=4){const a=(random()-.5)*22;img.data[i]=base[0]+a;img.data[i+1]=base[1]+a;img.data[i+2]=base[2]+a;img.data[i+3]=255}ctx.putImageData(img,0,0);
 ctx.strokeStyle='rgba(26,33,37,.16)';ctx.lineWidth=metal?2:1;
 for(let x=0;x<512;x+=metal?32:128){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,512);ctx.stroke()}
 if(!metal)for(let y=0;y<512;y+=128){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(512,y);ctx.stroke()}
 const tex=new THREE.CanvasTexture(canvas);tex.colorSpace=THREE.SRGBColorSpace;tex.wrapS=tex.wrapT=THREE.RepeatWrapping;tex.repeat.set(metal?2:3,metal?2:3);tex.anisotropy=8;return tex;
}

function plaque(text){
 const c=document.createElement('canvas');c.width=256;c.height=96;const a=c.getContext('2d');a.fillStyle='rgba(11,22,29,.86)';a.fillRect(0,0,256,96);a.strokeStyle='#a9beca';a.strokeRect(2,2,252,92);a.fillStyle='#eef5f8';a.font='600 48px system-ui';a.textAlign='center';a.textBaseline='middle';a.fillText(text,128,48);
 const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),transparent:true,depthTest:false,fog:false,toneMapped:false}));sprite.scale.set(3.7,1.39,1);sprite.renderOrder=12;return sprite;
}

export function buildEnvironment(scene,env){
 const concrete=new THREE.MeshStandardMaterial({map:surface([151,151,146]),roughness:.94});
 const wall=new THREE.MeshStandardMaterial({map:surface([169,171,166],17),roughness:.83});
 const roof=new THREE.MeshStandardMaterial({map:surface([109,117,122],3,true),metalness:.55,roughness:.55});
 const steel=new THREE.MeshStandardMaterial({color:0x4e5a61,metalness:.75,roughness:.4});
 const glass=new THREE.MeshStandardMaterial({color:0x34444e,metalness:.65,roughness:.22});
 const pale=new THREE.MeshStandardMaterial({color:0xaeb1ab,roughness:.82});
 const dark=new THREE.MeshStandardMaterial({color:0x313a3c,roughness:.98});
 const paint=new THREE.MeshStandardMaterial({color:0xc1bfae,roughness:1});
 const add=(size,p,mat)=>{const m=new THREE.Mesh(new THREE.BoxGeometry(...size),mat);m.position.set(...p);m.castShadow=m.receiveShadow=true;scene.add(m);return m};
 add([100,85,.3],[20,14,-.3],dark);add([47,35,.15],[20,14,-.08],concrete);
 // Expansion joints, drainage channels and worn road markings establish scale.
 for(let x=0;x<=40;x+=5)add([.025,28,.012],[x,14,.014],dark);
 for(let y=0;y<=28;y+=4)add([40,.025,.012],[20,y,.014],dark);
 for(const y of [1.3,9,25])for(let x=10;x<32;x+=3)add([1.25,.08,.012],[x,y,.024],paint);
 for(const r of env.buildings){const [a,b,c,d,h]=r,w=c-a,l=d-b,x=(a+c)/2,y=(b+d)/2;
   add([w+.12,l+.12,.25],[x,y,.125],pale);add([w,l,h],[x,y,h/2],wall);
   add([w,l,.13],[x,y,h+.05],roof);
   for(const yy of [b+.08,d-.08])add([w,.16,.36],[x,yy,h+.18],pale);
   for(const xx of [a+.08,c-.08])add([.16,l,.36],[xx,y,h+.18],pale);
   for(let xx=a+.8;xx<c-.6;xx+=1.8)for(const yy of [b-.026,d+.026]){
     add([1.25,.08,1.05],[xx,yy,Math.min(h-1.3,3.9)],steel);add([1.1,.09,.90],[xx,yy,Math.min(h-1.3,3.9)],glass);
   }
   if(w>5){add([3.5,.06,2.7],[a+2.5,b-.05,1.4],steel);for(let z=.25;z<2.8;z+=.24)add([3.3,.07,.025],[a+2.5,b-.09,z],pale)}
   for(let xx=a+1.7;xx<c-1;xx+=4){add([1.5,1.7,.65],[xx,y,h+.42],steel);add([1.3,1.5,.07],[xx,y,h+.79],pale);for(let zz=0;zz<7;zz++)add([1.2,.04,.03],[xx,y-.6+zz*.2,h+.84],dark)}
   for(const xx of [a+.15,c-.15])add([.09,.11,h],[xx,b-.08,h/2],steel);
 }
 // Fencing and service equipment are outside the flight domain.
 for(const y of [-1.4,29.4])for(let x=-1;x<42;x+=2.8){add([.07,.07,2.1],[x,y,1.05],steel);add([2.8,.035,.05],[x+1.4,y,1.9],steel);add([2.8,.025,.04],[x+1.4,y,.55],steel);for(let k=1;k<8;k++)add([.018,.02,1.7],[x+k*.35,y,1.15],steel)}
 for(let i=0;i<6;i++)add([1.2,.8,.4],[2+i*1.7,-4,.2],i%2?steel:pale);
 for(const [s,y]of[['A',9],['B',25],['C',1.3]]){const p=plaque('PASSAGE '+s);p.position.set(10.8,y,.35);scene.add(p)}
 for(const [i,r]of env.regions.entries()){
  const [a,b,c,d]=r;const pts=[[a,b,.03],[c,b,.03],[c,d,.03],[a,d,.03],[a,b,.03]].map(v=>new THREE.Vector3(...v));
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0xd8b16e,fog:false,toneMapped:false})));
  const p=plaque(i?'INSPECTION R':'INSPECTION L');p.scale.multiplyScalar(.75);p.position.set((a+c)/2,b-1.0,.3);scene.add(p);
 }
}

export function createSensor(scene){
 const c=document.createElement('canvas');c.width=c.height=256;const a=c.getContext('2d');
 const image=a.createImageData(256,256);for(let y=0;y<256;y++)for(let x=0;x<256;x++){
   const r2=((x-127.5)*6.1/256)**2+((y-127.5)*6.1/256)**2,k=(y*256+x)*4;
   image.data.set([82,225,209,85*Math.exp(-r2/4)],k);
 }a.putImageData(image,0,0);
 const group=new THREE.Group();const mat=new THREE.MeshBasicMaterial({map:new THREE.CanvasTexture(c),transparent:true,depthWrite:false,fog:false,toneMapped:false});
 const fill=new THREE.Mesh(new THREE.PlaneGeometry(6.1,6.1),mat);group.add(fill);
 for(const [r,opacity] of [[2*Math.sqrt(Math.log(2)),.9],[2*Math.sqrt(Math.log(10)),.3]]){
  const ring=new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(Array.from({length:64},(_,i)=>new THREE.Vector3(r*Math.cos(i*Math.PI/32),r*Math.sin(i*Math.PI/32),.01))),new THREE.LineBasicMaterial({color:0x68ddcd,transparent:true,opacity,fog:false,toneMapped:false}));group.add(ring);
 }
 scene.add(group);
 const rays=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x68ddcd,transparent:true,opacity:.16,depthWrite:false,fog:false,toneMapped:false}));scene.add(rays);
 return {group,rays};
}

export function updateSensor(sensor,position,visible){
 sensor.group.visible=sensor.rays.visible=visible;sensor.group.position.set(position.x,position.y,.14);
 const r=2*Math.sqrt(Math.log(2)),points=[];for(let i=0;i<4;i++){const a=i*Math.PI/2+Math.PI/4;points.push(position.clone(),new THREE.Vector3(position.x+r*Math.cos(a),position.y+r*Math.sin(a),.16))}
 sensor.rays.geometry.dispose();sensor.rays.geometry=new THREE.BufferGeometry().setFromPoints(points);
}

export function sampleGrid(vectors,x,y){
 const gx=Math.min(18.999,Math.max(0,(x-1)/2)),gy=Math.min(12.999,Math.max(0,(y-1)/2));const ix=Math.floor(gx),iy=Math.floor(gy),fx=gx-ix,fy=gy-iy;
 const a=vectors[iy*20+ix],b=vectors[iy*20+ix+1],c=vectors[(iy+1)*20+ix],d=vectors[(iy+1)*20+ix+1];
 return [0,1].map(k=>(1-fy)*((1-fx)*a[k]+fx*b[k])+fy*((1-fx)*c[k]+fx*d[k]));
}

export function createFlow(scene,env){
 const geometry=new THREE.BufferGeometry();const count=190;geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(count*6),3));
 geometry.setAttribute('color',new THREE.BufferAttribute(new Float32Array(count*6),3));
 const material=new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.8,depthWrite:false,fog:false,toneMapped:false});
 const lines=new THREE.LineSegments(geometry,material);scene.add(lines);return {lines,env,count,paths:[],key:null};
}

export function updateFlow(flow,vectors,key,time,visible){
 flow.lines.visible=visible;if(!visible)return;
 if(flow.key!==key){
  flow.key=key;flow.paths=[];const active=flow.env.grid.filter((p,i)=>Math.hypot(...vectors[i])>.3&&!flow.env.buildings.some(r=>p[0]>r[0]&&p[0]<r[2]&&p[1]>r[1]&&p[1]<r[3]));
  for(let n=0;n<flow.count;n++){
   if(!active.length){flow.paths.push(null);continue}
   const seed=active[(n*47)%active.length];let x=seed[0]+Math.sin(n*17)*.6,y=seed[1]+Math.cos(n*13)*.6,z=2+((n*31)%100)/32;const path=[];
   for(let k=0;k<50;k++){path.push([x,y,z]);const w=sampleGrid(vectors,x,y);const nx=x+w[0]*.12,ny=y+w[1]*.12;if(!flow.env.buildings.some(r=>nx>r[0]&&nx<r[2]&&ny>r[1]&&ny<r[3])){x=nx;y=ny}}
   flow.paths.push(path);
  }
 }
 const positions=flow.lines.geometry.attributes.position,colors=flow.lines.geometry.attributes.color;
 const low=new THREE.Color('#466d91'),high=new THREE.Color('#e3edf0');
 flow.paths.forEach((path,n)=>{if(!path){positions.setXYZ(2*n,0,0,-10);positions.setXYZ(2*n+1,0,0,-10);return}const phase=((time+n*.371)%5.52)/.12,i=Math.min(45,Math.floor(phase)),f=phase-i;const a=path[i],b=path[i+1],c=path[i+4];positions.setXYZ(2*n,...a.map((v,k)=>v+(b[k]-v)*f));positions.setXYZ(2*n+1,...c);const wind=sampleGrid(vectors,...a),color=low.clone().lerp(high,Math.min(1,Math.hypot(...wind)/5));colors.setXYZ(2*n,color.r*.25,color.g*.25,color.b*.25);colors.setXYZ(2*n+1,color.r,color.g,color.b)});
 colors.needsUpdate=true;positions.needsUpdate=true;flow.lines.geometry.computeBoundingSphere();
}
