// Deterministic scenery only. These functions never enter the planner or safety filter.
export const DISPLAY_CEILING = .72;
export function noise(x,y){
  const hash=(a,b)=>{const n=Math.sin(a*127.1+b*311.7)*43758.5453;return n-Math.floor(n);};
  const ix=Math.floor(x),iy=Math.floor(y),tx=x-ix,ty=y-iy;
  const u=tx*tx*(3-2*tx),v=ty*ty*(3-2*ty);
  return (hash(ix,iy)*(1-u)+hash(ix+1,iy)*u)*(1-v)+(hash(ix,iy+1)*(1-u)+hash(ix+1,iy+1)*u)*v;
}
export function fbm(x,y){return .55*noise(x,y)+.27*noise(x*2.03+13,y*2.03-7)+.12*noise(x*4.07-3,y*4.07+19)+.06*noise(x*8.2,y*8.2);}
export function riverDistance(x,y){return y-.6*x-.10*Math.sin(x*1.7);}
export function terrainHeight(x,y){
  const valley=.025+.09*(fbm(x*1.7,y*1.7)-.5);
  const river=-.09*Math.exp(-(riverDistance(x,y)**2)/.014);
  const outside=Math.max(Math.abs(x)-3.15,Math.abs(y)-2.65,0);
  const foothills=Math.min(outside*.55,1.5)*(.35+1.6*fbm(x*.55+30,y*.55+8));
  return valley+river+foothills;
}

// Rugged cliff meshes stay inside the ORIGINAL conservative circular envelopes.
// The display height is above the fixed sensing flight band, with no climb action.
export function ridgeVertex(obstacle,level,angle,index=0){
  const [x,y,r]=obstacle;
  // Broad, irregular shoulders and an offset ridgeline, not tall rotational spires.
  const profiles=[[1,-.02],[.84,.18],[.66,.40],[.47,.62],[.30,.78],[.14,.90],[0,.96]];
  const [radius,height]=profiles[level];
  const rough=.87+.12*fbm(Math.cos(angle)*3+index*9,Math.sin(angle)*3+level*.12);
  const radial=r*radius*rough;
  const offset=(1-radius)*r*.20,axis=index*.9+.6;
  const peak=height*(.93+.10*fbm(Math.cos(angle)*2+index,Math.sin(angle)*2+level*.3));
  return [x+Math.cos(angle)*radial+offset*Math.cos(axis),terrainHeight(x,y)+peak,y+Math.sin(angle)*radial*.88+offset*Math.sin(axis)];
}

export function cameraSightClear(origin,eye,obstacles){
  const dx=eye[0]-origin[0],dz=eye[2]-origin[2],a=dx*dx+dz*dz;
  if(a<1e-10)return true;
  return obstacles.every(([x,y,r])=>{
    const px=origin[0]-x,pz=origin[2]+y;
    const t=Math.max(0,Math.min(1,-(px*dx+pz*dz)/a));
    return Math.hypot(px+t*dx,pz+t*dz)>r+.025;
  });
}
export function chasePosition(origin,heading,obstacles){
  // Avoid putting the viewing camera inside a ridge. Aircraft position is untouched.
  for(const turn of [0,.7,-.7,1.4,-1.4,Math.PI]){
    const angle=heading+turn,c=Math.cos(angle),s=Math.sin(angle);
    const eye=[origin[0]-1.12*c+.36*s,origin[1]+.47,origin[2]+1.12*s+.36*c];
    eye[1]=Math.max(eye[1],terrainHeight(eye[0],-eye[2])+.18);
    if(cameraSightClear(origin,eye,obstacles))return eye;
  }
  return [origin[0],origin[1]+1.5,origin[2]+.001];
}
