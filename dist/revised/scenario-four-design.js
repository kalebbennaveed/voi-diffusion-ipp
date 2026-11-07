'use strict';
(async()=>{
 const response=await fetch('data/scenario4-design-20261107/geometry-audit.json');
 if(!response.ok)throw Error('Geometry audit unavailable');
 const data=await response.json(), spec=data.environment, summary=data.summary;
 const $=id=>document.getElementById(id), pct=n=>(100*n).toFixed(1)+'%';
 const colors=['#46738b','#a35e30','#857348','#8665a7','#1d8c75','#bf6272'];
 $('metrics').innerHTML=`<div><b>${pct(summary.minimum_active_fraction)}–${pct(summary.maximum_active_fraction)}</b><span>Proposed domain · one planning call</span></div><div><b>${pct(1-summary.union_active_fraction)}</b><span>Outside the union of 44 tested masks</span></div><div><b>100%</b><span>Traversable grid connected · 0.10 m clearance</span></div><div><b>${summary.geometry_and_mask_ms.median.toFixed(1)} ms</b><span>Geometry + mask only · median of 20 calls</span></div>`;
 const positions=[...new Set(data.cases.map(c=>c.pose))],clarities=[...new Set(data.cases.map(c=>c.clarity))];
 for(const p of positions)$('position').add(new Option(p,p));
 for(const c of clarities)$('clarity').add(new Option(c,c));
 $('checks').innerHTML=`<li>${summary.tested_cases} combinations: ${summary.start_positions} positions × 4 clarity maps.</li><li>Includes positions on the long alternative, not just starts inside B.</li><li>Minimum observed segment clearance: ${summary.minimum_path_clearance_m.toFixed(3)} m.</li><li>All ${summary.connectivity.safe_nodes.toLocaleString()} traversable grid cells are in one connected component.</li><li>Same 64 × 48 grid as the current scenarios; no larger state grid.</li><li>The geometric screen takes ${summary.total_screen_wall_seconds.toFixed(2)} s in total. Full MBD runtime is not measured here.</li>`;
 $('ports').textContent=JSON.stringify(spec.wind_design.boundary_proposal,null,2);
 function frame(canvas){const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);ctx.fillStyle='#fafcfb';ctx.fillRect(0,0,w,h);return{ctx,w,h,point:p=>[p[0]/8*w,(6-p[1])/6*h]};}
 function base(f,layer){const{ctx,w,h,point}=f;
  if(layer){ctx.fillStyle=layer.color;for(const i of layer.ids){const p=data.nodes[i],xy=point([p[0]-.0625,p[1]+.0625]);ctx.fillRect(xy[0],xy[1],.125/8*w+.2,.125/6*h+.2);}}
  for(const r of spec.regions){const [lo,hi]=r.bounds,a=point([lo[0],hi[1]]);ctx.fillStyle=layer?'rgba(248,200,139,.45)':'#fae2bd';ctx.fillRect(...a,(hi[0]-lo[0])/8*w,(hi[1]-lo[1])/6*h);ctx.strokeStyle='#c68740';ctx.lineWidth=2;ctx.strokeRect(...a,(hi[0]-lo[0])/8*w,(hi[1]-lo[1])/6*h);ctx.font='600 29px system-ui';ctx.fillStyle='#6e4b22';ctx.fillText('Monitor '+r.label,a[0]+13,a[1]+29);}
  for(const [x,y,r] of spec.obstacles){const a=point([x,y]);ctx.beginPath();ctx.arc(...a,r/8*w,0,Math.PI*2);ctx.fillStyle='#78898e';ctx.fill();ctx.strokeStyle='#65797d';ctx.lineWidth=1.5;ctx.stroke();}
  ctx.strokeStyle='#a9bdb3';ctx.lineWidth=2;ctx.strokeRect(1,1,w-2,h-2);
 }
 function paths(f,c){const{ctx,point}=f;for(let i=0;i<c.paths.length;i++){const p=c.paths[i];ctx.beginPath();p.forEach((v,j)=>{const a=point(v);j?ctx.lineTo(...a):ctx.moveTo(...a)});ctx.strokeStyle=colors[i%colors.length];ctx.lineWidth=3.5;ctx.stroke();const a=point(p[p.length-1]);ctx.fillStyle=colors[i%colors.length];ctx.beginPath();ctx.arc(...a,4.7,0,Math.PI*2);ctx.fill();}const a=point(c.state);ctx.beginPath();ctx.arc(...a,9,0,Math.PI*2);ctx.fillStyle='#163f33';ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=3;ctx.stroke();}
 function arrow(f,center,direction,label,offset){const{ctx,point}=f;const length=.9,n=Math.hypot(...direction),d=direction.map(v=>v/n),a=point(center.map((v,i)=>v-d[i]*length/2)),b=point(center.map((v,i)=>v+d[i]*length/2));ctx.save();ctx.strokeStyle='#b16c27';ctx.fillStyle='#9a571b';ctx.lineWidth=5;ctx.setLineDash([11,7]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.setLineDash([]);const angle=Math.atan2(b[1]-a[1],b[0]-a[0]);ctx.beginPath();ctx.moveTo(...b);ctx.lineTo(b[0]-17*Math.cos(angle-.45),b[1]-17*Math.sin(angle-.45));ctx.lineTo(b[0]-17*Math.cos(angle+.45),b[1]-17*Math.sin(angle+.45));ctx.closePath();ctx.fill();const t=point(center.map((v,i)=>v+offset[i]));ctx.font='600 28px system-ui';const tw=ctx.measureText(label).width;ctx.fillStyle='rgba(255,252,247,.94)';ctx.fillRect(t[0]-5,t[1]-29,tw+10,39);ctx.fillStyle='#9a571b';ctx.fillText(label,...t);ctx.restore();}
 function render(){const c=data.cases.find(c=>c.pose===$('position').value&&c.clarity===$('clarity').value);const left=frame($('layout')),right=frame($('mask'));base(left);paths(left,c);if($('wind').checked){arrow(left,spec.wind_design.task_passage.center_m,spec.wind_design.task_passage.direction,'Opposes B → A',[-1.65,.63]);arrow(left,spec.wind_design.remote_yard.center_m,spec.wind_design.remote_yard.direction,'Yard flow · intended',[-.8,-.65]);}
  const domain=$('domain').value,ids=domain==='global'?data.free_indices:domain==='union'?data.union_active_indices:c.active_indices;
  base(right,{ids,color:domain==='global'?'#c2dfed':'#c8e7df'});paths(right,c);
  $('domain-title').textContent=domain==='global'?'Baseline 4 · learn across all free space':domain==='union'?'Union of all 44 proposed masks':'Proposed · current candidate-path tubes';
  $('case-summary').textContent=`Shown domain: ${pct(ids.length/summary.free_cells)} of free cells. Current proposed domain: ${pct(c.active_fraction)}. The actual C² weights taper inside each tube; shading shows its nonzero support.`;
  $('routes').innerHTML=c.paths.map((p,i)=>{const length=p.slice(1).reduce((s,v,j)=>s+Math.hypot(v[0]-p[j][0],v[1]-p[j][1]),0);return `<span style="--route:${colors[i]}">Seed ${i+1} · ${length.toFixed(2)} m</span>`}).join('');
 }
 ['position','clarity','domain','wind'].forEach(id=>$(id).addEventListener('change',render));render();
})().catch(error=>{document.getElementById('metrics').textContent='Could not load the geometry audit: '+error.message;console.error(error);});
