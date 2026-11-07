'use strict';
// Playback of actual executed trajectories, measurements and posterior snapshots.
(async()=>{
 const dir='data/waypoint-learning-300',cache=new Map();
 async function read(name){if(cache.has(name))return cache.get(name);const job=(async()=>{const r=await fetch(`${dir}/${name}`);if(!r.ok)throw Error(`${name}: ${r.status}`);if(!name.endsWith('.gz'))return r.json();return new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).json();})();cache.set(name,job);return job;}
 const at=(rows,t,key=r=>r.t)=>{let a=0,b=rows.length-1;while(b-a>1){let c=(a+b)>>1;if(key(rows[c])<=t)a=c;else b=c;}return key(rows[b])<=t?b:a;};
 const ramps={std:['#f7fbff','#deebf7','#9ecae1','#4292c6','#2171b5','#08306b'],mean:['#f7fcfd','#ccece6','#66c2a4','#238b45','#005824'],clarity:['#fff7ec','#fee8c8','#fdbb84','#fc8d59','#d7301f','#7f0000']};
 const layers={std:{name:'Wind uncertainty σ',max:1.8,unit:'m/s',ramp:'std'},mean:{name:'Learned mean wind speed',max:10,unit:'m/s',ramp:'mean'},clarity:{name:'Clarity q',max:1,unit:'',ramp:'clarity'},deficit:{name:'Clarity deficit',max:.7,unit:'',ramp:'clarity'}};
 const rgb=h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16));
 const shade=(v,l)=>{const a=ramps[l.ramp].map(rgb),z=Math.max(0,Math.min(1,v/l.max))*(a.length-1),i=Math.min(a.length-2,Math.floor(z)),f=z-i;return a[i].map((x,k)=>Math.round(x+(a[i+1][k]-x)*f));};
 try{
 const [env,manifest]=await Promise.all([read('environment.json'),read('manifest.json')]);const [nx,ny]=env.grid,W=800,H=600;
 const xy=p=>[100*p[0],H-100*p[1]],line=pp=>pp.map(p=>xy(p).join(',')).join(' '),freeSet=new Set(env.free_indices);
 for(const mode of ['survey','baseline1']){
  const root=document.getElementById(mode==='survey'?'uncertainty-waypoints':'learn-then-mbd');if(!root)continue;
  const host=root.querySelector('.learning-view');const metadata=manifest.runs.filter(r=>r.method===mode).sort((a,b)=>a.scenario-b.scenario);
  const records=await Promise.all(metadata.map(r=>read(r.file)));let time=0,playing=false,last=null;
  host.innerHTML=`<div class="learning-pair"></div><div class="learning-controls"><button data-action="play" aria-pressed="false">▶ Play both scenarios</button><button data-action="reset">Restart</button><input aria-label="${mode} playback time" data-control="time" type="range" min="0" max="300" step=".2" value="0"><output>0 / 300 s</output><label>Speed <select data-control="speed"><option value="1">1×</option><option value="4" selected>4×</option><option value="10">10×</option></select></label><label>Map <select data-control="layer">${Object.entries(layers).map(([k,l])=>`<option value="${k}">${l.name}</option>`).join('')}</select></label><label>Jump <select data-control="jump"><option value="0">Start</option><option value="30">30 s</option><option value="60">60 s · handoff</option><option value="120">120 s</option><option value="180">180 s</option><option value="300">End</option></select></label></div><div class="learning-legend"><b></b><span>0</span><i class="learning-ramp"></i><span class="learning-max"></span></div><div class="learning-results"></div><div class="learning-pair learning-charts"></div>`;
  const find=s=>host.querySelector(s);
  const notice=document.createElement('p');notice.className='learning-note learning-warning';
  notice.textContent=mode==='survey'?'Recorded limitation: Scenario 1 contacts the boundary at 103.3 s. Scenario 2 completes 300 s without contact, but wind-estimation error increases. Reduced model uncertainty does not establish accurate wind learning.':'Baseline 1 is not validated: both runs reach A but later contact the boundary (203.4 s and 186.2 s). These are failed tests, not successful 300 s monitoring results.';
  find('.learning-legend').after(notice);
  const panels=records.map((record,i)=>{
   const arrow=`learning-arrow-${mode}-${i}`;const card=document.createElement('article');card.className='learning-card';
   const shapes=env.obstacles.map(o=>`<circle cx="${100*o[0]}" cy="${H-100*o[1]}" r="${100*o[2]}" fill="#75858c"/>`).join('');
   const regions=env.regions.map(r=>{const[a,b]=r.bounds;return `<rect x="${100*a[0]}" y="${H-100*b[1]}" width="${100*(b[0]-a[0])}" height="${100*(b[1]-a[1])}" fill="none" stroke="${r.label==='A'?'#2563eb':'#b45309'}" stroke-width="2"/><text class="learning-region" x="${100*a[0]+12}" y="${H-100*b[1]+25}">Region ${r.label}</text>`;}).join('');
   card.innerHTML=`<header><h3>Scenario ${record.scenario}</h3><p class="learning-phase"></p></header><div class="learning-map"><svg viewBox="-4 -4 808 608" role="img" aria-label="${mode==='survey'?'Uncertainty waypoints with RPCBF':'Learn then MBD'} scenario ${record.scenario}"><defs><marker id="${arrow}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#163e60"/></marker></defs><rect width="800" height="600" fill="#f7fbff"/><image class="learning-heat" width="800" height="600"/>${regions}${shapes}<g class="learning-vectors"></g><polyline class="learning-route"/><circle class="learning-target" r="6"/><polyline class="learning-trace-halo" fill="none" stroke="white" stroke-width="6"/><polyline class="learning-trace" fill="none" stroke="#086c7c" stroke-width="3"/><circle class="learning-dot" r="6" fill="#086c7c" stroke="white" stroke-width="2"/><rect width="800" height="600" fill="none" stroke="#889eaa"/></svg></div><p class="learning-live"></p>`;
   find('.learning-pair').append(card);
   const c=document.createElement('div');c.className='learning-chart';const f=record.belief_frames;
   c.innerHTML=`<p>Scenario ${record.scenario} · mean posterior variance / initial variance</p><svg viewBox="0 0 570 120"><path d="M30 10V95H550" stroke="#9aaeba" fill="none"/><polyline fill="none" stroke="#1f5c92" stroke-width="2" points="${f.map(b=>`${30+b.t/300*520},${95-80*b.mean_variance/f[0].mean_variance}`).join(' ')}"/>${mode==='baseline1'?'<path d="M134 10V95" stroke="#7c3aed" stroke-dasharray="4 3"/><text x="138" y="23" font-size="10">MBD begins</text>':''}<line class="learning-cursor" y1="10" y2="95" stroke="#173247"/><text x="10" y="18" font-size="11">1</text><text x="10" y="98" font-size="11">0</text><text x="30" y="113" font-size="11">0 s</text><text x="520" y="113" font-size="11">300 s</text></svg>`;find('.learning-charts').append(c);
   return {r:record,card,chart:c,arrow,heatKey:'',vectorKey:''};
  });
  find('.learning-results').innerHTML=`<table><thead><tr><th>Recorded outcome</th><th>Scenario 1</th><th>Scenario 2</th></tr></thead><tbody>${[
   ['Time simulated',s=>`${s.simulated_seconds.toFixed(1)} / 300 s`],['First A visit',s=>s.first_visit_s.A===null?'Not reached':`${s.first_visit_s.A.toFixed(1)} s`],
   ['Contact',s=>s.contact?`Yes · ${s.contact.time_s.toFixed(2)} s`:'None recorded'],['Infeasible RPCBF commands',s=>s.infeasible_execution_steps],
   ['Mean variance reduction',s=>`${(100*s.variance_reduction_fraction).toFixed(1)}%`],['Wind vector RMSE · offline',s=>`${s.initial_rmse_m_s.toFixed(2)} → ${s.final_rmse_m_s.toFixed(2)} m/s`],
   ['Accumulated deficit · until stop',s=>s.integrated_deficit.toFixed(2)]
  ].map(([name,fn])=>`<tr><th>${name}</th>${records.map(r=>`<td>${fn(r.summary)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  {const row=document.createElement('tr');row.innerHTML='<th>Waypoint decisions · warm median / max</th>'+records.map(r=>{const values=r.plans.filter(p=>p.phase==='learning'&&!p.compilation_included&&p.t!==2).map(p=>p.wall_seconds).sort((a,b)=>a-b);return `<td>${(1000*values[Math.floor(values.length/2)]).toFixed(0)} / ${(1000*values[values.length-1]).toFixed(0)} ms</td>`;}).join('');find('.learning-results tbody').append(row);}
  if(mode==='baseline1'){
   const row=document.createElement('tr');row.innerHTML='<th>MBD only · warm median / max</th>'+records.map(r=>{const values=r.plans.filter(p=>p.phase==='monitoring'&&!p.compilation_included).map(p=>p.wall_seconds).sort((a,b)=>a-b);return `<td>${values.length?`${(1000*values[Math.floor(values.length/2)]).toFixed(0)} / ${(1000*values[values.length-1]).toFixed(0)} ms`:'No warm MBD call'}</td>`;}).join('');find('.learning-results tbody').append(row);
  }
  function draw(){
   const layer=find('[data-control=layer]').value,l=layers[layer];find('.learning-legend b').textContent=l.name;find('.learning-max').textContent=`${l.max} ${l.unit}`;find('.learning-ramp').style.background=`linear-gradient(90deg,${ramps[l.ramp].join(',')})`;
   for(const p of panels){const r=p.r,t=Math.min(time,r.summary.simulated_seconds),s=r.samples[at(r.samples,t,z=>z[0])],b=r.belief_frames[at(r.belief_frames,t)],q=r.clarity_frames[at(r.clarity_frames,t)].q,plan=r.plans[at(r.plans,t)],learning=plan.phase==='learning';
    const key=layer+(layer==='clarity'||layer==='deficit'?Math.floor(t):b.t);
    if(key!==p.heatKey){p.heatKey=key;const c=document.createElement('canvas');c.width=nx;c.height=ny;const ctx=c.getContext('2d'),pix=ctx.createImageData(nx,ny);env.free_indices.forEach((cell,n)=>{const v=layer==='clarity'?q[n]:layer==='deficit'?Math.max(0,env.targets[n]-q[n]):layer==='mean'?Math.hypot(...b.mean[cell]):b.std[cell];const off=((ny-1-Math.floor(cell/nx))*nx+cell%nx)*4;shade(v,l).forEach((z,k)=>pix.data[off+k]=z);pix.data[off+3]=255;});ctx.putImageData(pix,0,0);p.card.querySelector('.learning-heat').setAttribute('href',c.toDataURL());}
    if(p.vectorKey!==b.t){p.vectorKey=b.t;let arrows='';for(let y=2;y<ny;y+=4)for(let x=2;x<nx;x+=4){const cell=y*nx+x;if(!freeSet.has(cell))continue;const w=b.mean[cell];if(Math.hypot(...w)<.08)continue;const z=xy([(x+.5)*env.dx_m,(y+.5)*env.dx_m]);arrows+=`<path d="M${z[0]} ${z[1]}l${4*w[0]} ${-4*w[1]}" stroke="#163e60" stroke-width="1.5" marker-end="url(#${p.arrow})"/>`;}p.card.querySelector('.learning-vectors').innerHTML=arrows;}
    p.card.querySelector('.learning-vectors').style.display=layer==='mean'?'':'none';
    const trace=line(r.samples.filter((z,k)=>k%4===0&&z[0]<=t).map(z=>z.slice(1,3)));for(const cl of ['learning-trace','learning-trace-halo'])p.card.querySelector('.'+cl).setAttribute('points',trace);
    const dot=p.card.querySelector('.learning-dot'),z=xy(s.slice(1,3));dot.setAttribute('cx',z[0]);dot.setAttribute('cy',z[1]);
    p.card.querySelector('.learning-route').setAttribute('points',line(learning?plan.path:plan.predicted_states[0].map(z=>z.slice(0,2))));
    const target=p.card.querySelector('.learning-target');target.style.display=learning?'':'none';if(learning){const z=xy(plan.target);target.setAttribute('cx',z[0]);target.setAttribute('cy',z[1]);}
    p.card.querySelector('.learning-phase').textContent=learning?'Uncertainty waypoint → tracking → RPCBF':'Learned POAM → disturbance-aware MBD → RPCBF';
    p.card.querySelector('.learning-live').textContent=`${b.observations} actual samples · mean σ ${b.mean_std.toFixed(2)} m/s · clarity A ${s[15].toFixed(2)} / B ${s[16].toFixed(2)} · ${s[11]?'QP feasible':'QP infeasible: fallback'}${r.summary.contact&&time>=r.summary.contact.time_s?' · stopped at contact':''}${!r.summary.completed&&!r.summary.contact&&time>=r.summary.simulated_seconds?' · run stopped: '+r.summary.stop_reason:''}`;
    const cursor=p.chart.querySelector('.learning-cursor');cursor.setAttribute('x1',30+t/300*520);cursor.setAttribute('x2',30+t/300*520);
   }
   find('[data-control=time]').value=time;find('output').textContent=`${time.toFixed(1)} / 300 s`;
  }
  function play(v){playing=v;find('[data-action=play]').textContent=v?'Ⅱ Pause':'▶ Play both scenarios';find('[data-action=play]').setAttribute('aria-pressed',String(v));}
  find('[data-action=play]').addEventListener('click',()=>{if(time>=300)time=0;play(!playing);draw();});find('[data-action=reset]').addEventListener('click',()=>{time=0;play(false);draw();});find('[data-control=time]').addEventListener('input',e=>{time=+e.target.value;play(false);draw();});find('[data-control=jump]').addEventListener('change',e=>{time=+e.target.value;play(false);draw();});find('[data-control=layer]').addEventListener('change',draw);
  draw();function tick(now){const dt=last===null?0:Math.min(.15,(now-last)/1000);last=now;if(playing&&!document.hidden){time=Math.min(300,time+dt*+find('[data-control=speed]').value);draw();if(time>=300)play(false);}requestAnimationFrame(tick);}requestAnimationFrame(tick);
 }
 }catch(e){document.querySelectorAll('.learning-view').forEach(el=>el.innerHTML=`<p role="alert">Could not load the new learning recordings: ${e.message}</p>`);console.error(e);}
})();
