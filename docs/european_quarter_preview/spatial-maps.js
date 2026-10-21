const palette=[[36,104,100],[112,175,146],[227,218,124],[228,143,72],[171,54,58]];
function color(value,max){const x=Math.max(0,Math.min(1,value/max))*4,i=Math.min(3,Math.floor(x)),a=x-i;return `rgb(${palette[i].map((c,k)=>Math.round(c*(1-a)+palette[i+1][k]*a)).join(',')})`}
const ramp='linear-gradient(90deg,#246864,#70af92,#e3da7c,#e48f48,#ab363a)';
export async function install(preview){
 const layout=document.querySelector('#layout-select').value;
 const initial=await fetch(`data/initial_${layout}.json`,{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Initial spatial maps unavailable');return r.json()});
 preview.mapState={...initial,t:0,label:'Initial condition',activeMethod:'Initial condition'};
 const draw=()=>{
  const state=preview.mapState,env=state.environment,q=state.q,mu=state.mean,std=state.std,truth=env.wind;
  for(const canvas of document.querySelectorAll('canvas[data-map]')){
   const kind=canvas.dataset.map,ctx=canvas.getContext('2d'),size=512;
   const extent=[env.bounds[0][0]-1,env.bounds[0][1]-1,env.bounds[1][0]+1,env.bounds[1][1]+1],scale=Math.min((size-36)/(extent[2]-extent[0]),(size-36)/(extent[3]-extent[1]));
   const ox=(size-(extent[2]-extent[0])*scale)/2,oy=(size-(extent[3]-extent[1])*scale)/2;
   const project=([x,z])=>[ox+(x-extent[0])*scale,oy+(z-extent[1])*scale];
   ctx.clearRect(0,0,size,size);ctx.fillStyle='#edf2eb';ctx.fillRect(0,0,size,size);
   const mode=canvas.parentElement.querySelector('[data-belief-mode]')?.value||'uncertainty';
   let maximum=kind==='deficit'?.75:kind==='clarity'?1:kind==='belief'&&mode==='uncertainty'?1.8:4.5;
   let unit=kind==='belief'&&mode==='uncertainty'?'Wind component standard deviation (m/s)':kind==='belief'&&mode==='truth'?'Physical wind speed (m/s)':kind==='belief'?'Posterior mean wind speed (m/s)':kind==='error'?'Wind estimation error magnitude (m/s)':kind==='deficit'?'Clarity deficit max(target − clarity, 0)':'Clarity';
   if(kind==='belief'||kind==='error'){
    const dx=env.grid[1][0]-env.grid[0][0],dz=env.grid[env.grid_shape[1]][1]-env.grid[0][1];
    for(let i=0;i<env.grid.length;i++){const point=project(env.grid[i]);let v=kind==='error'?Math.hypot(mu[i][0]-truth[i][0],mu[i][1]-truth[i][1]):mode==='uncertainty'?std[i]:Math.hypot(...(mode==='truth'?truth[i]:mu[i]));ctx.fillStyle=color(v,maximum);ctx.fillRect(point[0]-dx*scale/2,point[1]-dz*scale/2,dx*scale+1,dz*scale+1)}
   }else{
    for(let i=0;i<env.cells.length;i++){const point=project(env.cells[i]);const value=kind==='deficit'?Math.max(env.targets[i]-q[i],0):q[i];ctx.fillStyle=color(kind==='clarity'?1-value:value,maximum);ctx.fillRect(point[0]-.73*scale,point[1]-.71*scale,1.46*scale,1.42*scale)}
   }
   ctx.fillStyle='#616d6b';ctx.strokeStyle='#d8e1d9';ctx.lineWidth=1;
   for(const building of preview.world.buildings){const w=building.width,d=building.depth,c=Math.cos(building.rotation),s=Math.sin(building.rotation);const corners=[[-w/2,-d/2],[w/2,-d/2],[w/2,d/2],[-w/2,d/2]].map(([x,z])=>project([building.x+x*c+z*s,building.z-x*s+z*c]));ctx.beginPath();corners.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fill();ctx.stroke()}
   for(const [i,region] of env.regions.entries()){const [x,y]=project(region.slice(0,2)),[x2,y2]=project(region.slice(2));ctx.strokeStyle='#fcf4d7';ctx.lineWidth=1.5;ctx.setLineDash([5,3]);ctx.strokeRect(x,y,x2-x,y2-y);ctx.setLineDash([]);ctx.fillStyle='#20392f';ctx.font='bold 19px system-ui';ctx.fillText(['West','East','NW','SE'][i],x+3,y+21)}
   if(kind==='belief'){
    ctx.strokeStyle='#173e3de0';ctx.lineWidth=1.3;const bank=mode==='truth'?truth:mu;
    for(let row=1;row<env.grid_shape[0];row+=4)for(let col=1;col<env.grid_shape[1];col+=4){const i=row*env.grid_shape[1]+col,p=env.grid[i];if(env.buildings.some(b=>p[0]>b[0]&&p[0]<b[2]&&p[1]>b[1]&&p[1]<b[3]))continue;const [x,y]=project(p),[vx,vy]=bank[i],v=Math.hypot(vx,vy);if(v<.06)continue;const xx=x+vx*5,yy=y+vy*5,angle=Math.atan2(vy,vx);ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(xx,yy);ctx.lineTo(xx-4*Math.cos(angle-.5),yy-4*Math.sin(angle-.5));ctx.moveTo(xx,yy);ctx.lineTo(xx-4*Math.cos(angle+.5),yy-4*Math.sin(angle+.5));ctx.stroke()}
   }
   for(const [i,z] of env.passages.entries()){const [x,y]=project([0,z]);ctx.fillStyle='#f7faf3e8';ctx.beginPath();ctx.arc(x,y,12,0,2*Math.PI);ctx.fill();ctx.fillStyle='#234d43';ctx.font='bold 17px system-ui';ctx.textAlign='center';ctx.fillText('ABC'[i],x,y+6);ctx.textAlign='left'}
   if(state.position){const [x,y]=project([state.position[0],state.position[1]]);ctx.beginPath();ctx.arc(x,y,4,0,2*Math.PI);ctx.fillStyle='#fff';ctx.fill();ctx.strokeStyle='#174f48';ctx.lineWidth=2;ctx.stroke();ctx.beginPath();ctx.arc(x,y,state.position[2]*scale,0,2*Math.PI);ctx.lineWidth=1;ctx.stroke()}
   ctx.fillStyle='#355447';ctx.font='17px system-ui';ctx.fillText('N ↑',14,23);const bar=extent[2]-extent[0]<60?5:10;ctx.fillText(bar+' m',size-80,size-12);ctx.strokeStyle='#355447';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(size-90,size-27);ctx.lineTo(size-90+bar*scale,size-27);ctx.stroke();
   const legend=canvas.parentElement.querySelector('.map-legend');legend.innerHTML=`<div>${unit}</div><div class="color-ramp" style="background:${kind==='clarity'?'linear-gradient(90deg,#ab363a,#e48f48,#e3da7c,#70af92,#246864)':ramp}"></div><div class="legend-ends"><span>0</span><span>${maximum}</span></div><div>${kind==='belief'?'Vectors: '+(mode==='truth'?'physical wind':'posterior mean')+(mode!=='truth'&&state.beliefTime!==undefined?' · updated '+state.beliefTime.toFixed(0)+' s':''):'Gray buildings · unmonitored ground left blank'}</div>`;
   canvas.parentElement.querySelector('.map-time').textContent=`${state.activeMethod} · ${state.t.toFixed(1)} s`;
  }
 };
 preview.drawMaps=draw;window.addEventListener('map-panels-changed',draw);document.addEventListener('change',e=>{if(e.target.matches('[data-belief-mode]'))draw()});draw();
 const playback=await import('./replay.js?v=compact-control-v1');await playback.install(preview,initial);
}
