'use strict';
// Scenario-design preview only. These loops have no physical time, controls,
// dynamics integration, learned belief, clarity evaluation, or planner output.
const design = {
  status: 'Proposed scenario geometry; awaiting user confirmation',
  bounds: [[-2.8, -2.8], [2.8, 2.8]],
  regionA: [1.1, .25], regionB: [-1.1, -.25], regionSize: [1.4, 1.2],
  obstacleNormalOffset: .8, obstacleRadius: .35,
  central: {amplitude: 1.8, longitudinalWidth: .7050321357924049, transverseWidth: .35, crossFraction: .2},
  lower: {amplitude: 1.8, axialWindow: [-1.3, 1.3, .35], normalWindow: [.85, 1.95, .25]},
  units: {position: 'm', disturbance: 'm/s^2'},
  intendedRobotModel: 'p_dot = v; v_dot = u + d(p); not simulated in this preview',
  illustrativePaths: {scenario1: 'A -> central -> B -> lower -> A', scenario2: 'A -> central -> B -> upper -> A', physicalValidation: false},
  routeNormalOffsets: {central: 0, lower: 1.53, upper: -1.68}
};
const norm = Math.hypot(design.regionB[0]-design.regionA[0], design.regionB[1]-design.regionA[1]);
const t = design.regionA.map((v,i)=>(design.regionB[i]-v)/norm), n = [-t[1],t[0]];
const circles = [-1,1].map(sign=>n.map(v=>sign*design.obstacleNormalOffset*v));
function smooth(z) { z=Math.max(0,Math.min(1,z)); return z*z*z*(10+z*(-15+6*z)); }
function windowAt(x,w) { return smooth((x-w[0])/w[2])*smooth((w[1]-x)/w[2]); }
function windAt(x,y,scenario) {
  const s=x*t[0]+y*t[1], r=x*n[0]+y*n[1], w=design.central;
  const central=w.amplitude*Math.exp(-.5*((s/w.longitudinalWidth)**4+(r/w.transverseWidth)**2));
  const lower=scenario===2?design.lower.amplitude*windowAt(s,design.lower.axialWindow)*windowAt(r,design.lower.normalWindow):0;
  return t.map((v,i)=>central*(v+w.crossFraction*n[i])+lower*v);
}
function routePoint(f,h) { return design.regionA.map((v,i)=>v+f*(design.regionB[i]-v)+h*Math.sin(Math.PI*f)**2*n[i]); }
const colors={central:'#ce665d',lower:'#31a5b1',upper:'#bea03c',path:'#087f52'};
const shades=[[252,253,252],[220,236,243],[133,183,210],[49,107,150]];
function heatColor(m) { const z=Math.min(1,m/2)*3,k=Math.min(2,Math.floor(z)),a=z-k; return shades[k].map((v,i)=>Math.round(v*(1-a)+shades[k+1][i]*a)); }
function drawMap(canvas,scenario) {
  const box=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
  canvas.width=Math.round(box.width*dpr); canvas.height=Math.round(box.height*dpr);
  const c=canvas.getContext('2d');c.scale(dpr,dpr);
  const w=box.width,h=box.height,pad=34,size=Math.min(w-2*pad,h-36),left=(w-size)/2,top=5,scale=size/5.6;
  const X=x=>left+(x+2.8)*scale,Y=y=>top+(2.8-y)*scale;
  c.clearRect(0,0,w,h);
  const heat=document.createElement('canvas');heat.width=480;heat.height=480;
  const hc=heat.getContext('2d'), pixels=hc.createImageData(480,480);
  const showWind=document.querySelector('#wind').checked;
  for(let j=0;j<480;j++)for(let i=0;i<480;i++){
    const d=showWind?windAt(-2.8+(i+.5)*5.6/480,2.8-(j+.5)*5.6/480,scenario):[0,0];
    const rgb=heatColor(Math.hypot(...d)),k=(j*480+i)*4;pixels.data.set([...rgb,255],k);
  }
  hc.putImageData(pixels,0,0);c.drawImage(heat,left,top,size,size);
  c.lineWidth=.6;c.strokeStyle='#17303b12';
  for(let z=-2;z<=2;z++) {c.beginPath();c.moveTo(X(z),Y(-2.8));c.lineTo(X(z),Y(2.8));c.moveTo(X(-2.8),Y(z));c.lineTo(X(2.8),Y(z));c.stroke();}
  function linePath(h,stroke,width,dash=[]) {c.beginPath();for(let i=0;i<=240;i++){const p=routePoint(i/240,h);i?c.lineTo(X(p[0]),Y(p[1])):c.moveTo(X(p[0]),Y(p[1]));}c.strokeStyle=stroke;c.lineWidth=width;c.setLineDash(dash);c.stroke();c.setLineDash([]);}
  if(document.querySelector('#routes').checked)for(const key of ['upper','lower','central'])linePath(design.routeNormalOffsets[key],colors[key],2);
  // Rectangle boundaries are the full monitoring regions; not sensing footprints.
  for(const [label,p] of [['A',design.regionA],['B',design.regionB]]) {
    c.fillStyle='#eff3cf77';c.strokeStyle='#254650';c.lineWidth=1.8;
    c.fillRect(X(p[0]-.7),Y(p[1]+.6),1.4*scale,1.2*scale);c.strokeRect(X(p[0]-.7),Y(p[1]+.6),1.4*scale,1.2*scale);
    c.font=`600 ${Math.max(15,size/30)}px -apple-system,BlinkMacSystemFont,sans-serif`;c.fillStyle='#163d48';c.textAlign='left';c.fillText(label,X(p[0]-.7)+7,Y(p[1]+.6)+21);
  }
  if(showWind)for(let y=-2.6;y<=2.61;y+=.24)for(let x=-2.6;x<=2.61;x+=.24){
    if(circles.some(p=>Math.hypot(x-p[0],y-p[1])<design.obstacleRadius+.05))continue;
    const d=windAt(x,y,scenario),mag=Math.hypot(...d);if(mag<.08)continue;
    const len=mag*.105*scale,angle=Math.atan2(-d[1],d[0]),sx=X(x),sy=Y(y);
    c.strokeStyle='#224f70b0';c.fillStyle='#224f70b0';c.lineWidth=1.05;c.beginPath();c.moveTo(sx,sy);c.lineTo(sx+len*Math.cos(angle),sy+len*Math.sin(angle));c.stroke();
    if(len>3){const ex=sx+len*Math.cos(angle),ey=sy+len*Math.sin(angle),a=Math.min(3,len*.4);c.beginPath();c.moveTo(ex,ey);c.lineTo(ex-a*Math.cos(angle-.55),ey-a*Math.sin(angle-.55));c.lineTo(ex-a*Math.cos(angle+.55),ey-a*Math.sin(angle+.55));c.closePath();c.fill();}
  }
  for(const p of circles){c.beginPath();c.arc(X(p[0]),Y(p[1]),design.obstacleRadius*scale,0,2*Math.PI);c.fillStyle='#687b83';c.fill();c.strokeStyle='#536870';c.lineWidth=1;c.stroke();}
  if(document.querySelector('#paths').checked){
    const offset=scenario===1?design.routeNormalOffsets.lower:design.routeNormalOffsets.upper;
    for(const height of [0,offset]){linePath(height,'#ffffffd9',5.5);linePath(height,colors.path,2.8,[7,5]);}
    // Direction arrows describe only traversal order, never velocity.
    for(const [f,height,direction] of [[.36,0,1],[.42,offset,-1]]){
      const p=routePoint(f,height),q=routePoint(f+.005*direction,height),a=Math.atan2(Y(q[1])-Y(p[1]),X(q[0])-X(p[0]));
      c.save();c.translate(X(p[0]),Y(p[1]));c.rotate(a);c.fillStyle=colors.path;c.strokeStyle='white';c.lineWidth=1.4;c.beginPath();c.moveTo(7,0);c.lineTo(-5,-5);c.lineTo(-3,0);c.lineTo(-5,5);c.closePath();c.fill();c.stroke();c.restore();
    }
    c.beginPath();c.arc(X(design.regionA[0]),Y(design.regionA[1]),5.5,0,Math.PI*2);c.fillStyle=colors.path;c.fill();c.strokeStyle='white';c.lineWidth=2;c.stroke();
  }
  const small=Math.max(10,Math.min(12,size/40));
  function label(text,x,y,color){c.font=`500 ${small}px -apple-system,BlinkMacSystemFont,sans-serif`;c.textAlign='center';const tw=c.measureText(text).width;c.fillStyle='#ffffffdd';c.fillRect(X(x)-tw/2-5,Y(y)-small+1,tw+10,small+7);c.fillStyle=color;c.fillText(text,X(x),Y(y));}
  label('Upper bypass · longer',-.28,2.08,'#907323');
  label(scenario===1?'Lower bypass · relatively calm':'Lower bypass · additional wind',.34,-2.25,scenario===1?'#267f89':'#315e80');
  c.strokeStyle='#a9bab1';c.lineWidth=.8;c.strokeRect(left,top,size,size);
  c.fillStyle='#84938c';c.font='10px -apple-system,BlinkMacSystemFont,sans-serif';c.textAlign='center';
  for(let z=-2;z<=2;z++){c.fillText(z,X(z),top+size+16);c.textAlign='right';c.fillText(z,left-8,Y(z)+3);c.textAlign='center';}
  c.fillText('x (m)',left+size+17,top+size+16);c.textAlign='left';c.fillText('y (m)',Math.max(1,left-33),top+10);
}
let scheduled=false;
function draw(){scheduled=false;drawMap(document.querySelector('#map1'),1);drawMap(document.querySelector('#map2'),2);}
function schedule(){if(!scheduled){scheduled=true;requestAnimationFrame(draw);}}
document.querySelectorAll('input').forEach(input=>input.addEventListener('change',schedule));
new ResizeObserver(schedule).observe(document.querySelector('.scenarios'));
draw();
