export function values(run,kind){
 const mean=v=>v.reduce((a,x)=>a+x,0)/v.length;
 if(kind==='cost')return run.frames.map(f=>[f.t,f.cost]);
 if(kind==='speed')return run.frames.filter((f,i)=>i%5===0).map(f=>[f.t,Math.hypot(...f.velocity.slice(0,2))]);
 if(kind==='uncertainty')return run.beliefs.map(b=>[b.t,mean(b.std)]);
 if(kind==='error')return run.beliefs.map(b=>[b.t,Math.sqrt(mean(b.mean.map((w,i)=>(w[0]-run.environment.wind[i][0])**2+(w[1]-run.environment.wind[i][1])**2)))]);
 const indices=run.environment.cells.flatMap((p,i)=>(kind==='left'?p[0]<20:p[0]>20)?[i]:[]);
 return run.frames.filter((f,i)=>i%5===0).map(f=>[f.t,mean(indices.map(i=>Math.max(.75-f.q[i],0)))]);
}

export function drawBelief(canvas,run,belief,kind,component,frame){
 const ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height;ctx.clearRect(0,0,W,H);
 ctx.fillStyle='#111b24';ctx.fillRect(0,0,W,H);
 const left=30,top=20,width=W-45,height=H-82;
 const xx=x=>left+x/40*width,yy=y=>top+(28-y)/28*height;
 const color=v=>{
   if(kind==='std'){const u=Math.min(1,v/1.8);return `rgb(${31+u*212},${99+u*103},${116+u*7})`}
   if(component==='speed'){const u=Math.min(1,v/5);return `rgb(${40+u*140},${85+u*119},${123+u*100})`}
   const u=Math.min(1,Math.abs(v)/5);return v>=0?`rgb(${90+u*135},${108-u*22},${121-u*64})`:`rgb(${90-u*45},${108+u*55},${121+u*109})`;
 };
 const vector=kind==='true'?run.environment.wind:belief.mean;
 run.environment.grid.forEach((p,i)=>{const v=kind==='std'?belief.std[i]:component==='speed'?Math.hypot(...vector[i]):vector[i][component==='x'?0:1];ctx.fillStyle=color(v);ctx.fillRect(xx(p[0]-1),yy(p[1]+1),width/20+.5,height/14+.5)});
 ctx.fillStyle='#263139';ctx.strokeStyle='#bac4c9';ctx.lineWidth=1;
 for(const r of run.environment.buildings){ctx.fillRect(xx(r[0]),yy(r[3]),(r[2]-r[0])/40*width,(r[3]-r[1])/28*height);ctx.strokeRect(xx(r[0]),yy(r[3]),(r[2]-r[0])/40*width,(r[3]-r[1])/28*height)}
 ctx.strokeStyle='#d4ba82';for(const r of run.environment.regions)ctx.strokeRect(xx(r[0]),yy(r[3]),(r[2]-r[0])/40*width,(r[3]-r[1])/28*height);
 ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(xx(frame.p[0]),yy(frame.p[1]),3.5,0,2*Math.PI);ctx.fill();
 const barX=95,barW=W-150,barY=H-34;
 const lo=kind==='std'||component==='speed'?0:-5,hi=kind==='std'?1.8:5;
 for(let i=0;i<barW;i++){ctx.fillStyle=color(lo+(hi-lo)*i/(barW-1));ctx.fillRect(barX+i,barY,1,8)}
 ctx.fillStyle='#a6bac8';ctx.font='12px system-ui';ctx.fillText('0 m',left,H-47);ctx.fillText('40 m',W-50,H-47);ctx.fillText('28 m',2,14);if(kind==='mean'){ctx.textAlign='right';ctx.fillText(component==='speed'?'colour scale clipped at 5 m/s':'colour scale clipped at ±5 m/s',W-12,14);ctx.textAlign='left'}
 ctx.textAlign='center';ctx.fillText(`${lo} m/s`,barX,H-9);ctx.fillText(`${hi} m/s`,barX+barW,H-9);
 if(lo<0)ctx.fillText('0',barX+barW/2,H-9);ctx.textAlign='left';
}

export function crossings(run){
 const events=[];let side=0,last=0;
 run.frames.forEach((frame,i)=>{
  const next=frame.p[0]<13?0:frame.p[0]>29?1:null;
  if(next!==null){
   if(next!==side){
    const ys=run.frames.slice(last,i+1).filter(f=>f.p[0]>=14&&f.p[0]<=28).map(f=>f.p[1]).sort((a,b)=>a-b);
    if(ys.length){const y=ys[Math.floor(ys.length/2)];events.push({time:frame.t,direction:next?'right':'left',passage:y<4?'C':y<16?'A':'B'})}
   }
   side=next;last=i;
  }
 });return events;
}
