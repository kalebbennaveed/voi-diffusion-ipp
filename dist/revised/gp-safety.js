'use strict';
(async()=>{
 const root=document.getElementById('gp-safety-validation');if(!root)return;
 const dir='data/full-learning-20261104';
 const fileCache=new Map();
 const fetchFile=async file=>{const r=await fetch(`${dir}/${file}`,{cache:'no-store'});if(!r.ok)throw Error(`${file}: ${r.status}`);return file.endsWith('.gz')?new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).json():r.json();};
 const load=file=>{if(!fileCache.has(file))fileCache.set(file,fetchFile(file).catch(e=>{fileCache.delete(file);throw e;}));return fileCache.get(file);};
 const fmt=(x,d=2)=>Number.isFinite(x)?x.toFixed(d):'—';
 const median=values=>values.length%2?values[Math.floor(values.length/2)]:(values[values.length/2-1]+values[values.length/2])/2;
 const at=(rows,t,key=z=>z.t)=>{let lo=0,hi=rows.length-1;if(hi<0)return null;while(lo<hi){const m=Math.ceil((lo+hi)/2);if(key(rows[m])<=t)lo=m;else hi=m-1;}return rows[lo];};
 const blues=['#f7fbff','#d0e1f2','#73b3d8','#2879b9','#08306b'],warm=['#fff7ec','#fee8c8','#fdbb84','#ef6548','#b30000','#7f0000'];
 const color=(x,max,ramp)=>{const p=Math.max(0,Math.min(1,x/max))*(ramp.length-1),i=Math.min(ramp.length-2,Math.floor(p)),f=p-i;return `rgb(${[1,3,5].map(k=>Math.round(parseInt(ramp[i].slice(k,k+2),16)*(1-f)+parseInt(ramp[i+1].slice(k,k+2),16)*f)).join(',')})`;};
 try{
 const [env,manifest]=await Promise.all([load('environment.json'),load('manifest.json')]);
 const focused=root.dataset.focus==='true'?new URLSearchParams(location.search).get('group'):null;
 root.querySelector('.gpv-overview').innerHTML=focused?'<p><a href="index.html#'+focused+'">Open this comparison in the full demo</a></p><p role="status">'+(manifest.notice||'')+'</p>':manifest.overview;
 root.querySelector('.gpv-evidence').innerHTML=manifest.evidence;
 root.querySelector('.gpv-experiments').innerHTML='';
 const free=new Set(env.free_indices),[nx,ny]=env.grid;
 const groupLoaders=new Map();
 let archiveHost=null;
 if(!focused&&manifest.groups.some(g=>g.primary)){
  const archive=document.createElement('details');archive.id='gpv-archive';archive.className='gpv-archive';
  archive.innerHTML='<summary>Earlier simulations and diagnostics — preserved recordings</summary><div class="gpv-archive-groups"></div>';
  root.querySelector('.gpv-experiments').append(archive);archiveHost=archive.querySelector('div');
 }
 for(const group of manifest.groups.filter(g=>!focused||g.id===focused)){
  const host=document.createElement('div');host.className='gpv-group';host.id=group.id;
  const introduction=`<h3>${group.title}</h3><p>${group.description}</p>`;
  host.innerHTML=introduction+'<p role="status">Recordings load when this comparison comes into view.</p><button data-load>Load this comparison</button>';
  if(archiveHost&&!group.primary)archiveHost.append(host);
  else if(archiveHost)root.querySelector('.gpv-experiments').insertBefore(host,archiveHost.parentElement);
  else root.querySelector('.gpv-experiments').append(host);
  let loaded=false;
  async function renderGroup(){
  const records=await Promise.all(group.runs.map(load));
  if(!records.length){host.innerHTML=introduction+`<p role="status">${group.result}</p>`;return;}
  host.innerHTML=`<h3>${group.title}</h3><p>${group.description}</p><div class="gpv-pair"></div><div class="gpv-controls"><button data-play aria-pressed="false">▶ Play recordings</button><button data-reset>Restart</button><input data-time type="range" min="0" max="300" step=".05" value="0" aria-label="${group.title} time"><output></output><button data-end>End</button><label>Speed <select data-speed><option>1</option><option selected>4</option><option>10</option></select>×</label><label>Map <select data-map><option value="mean">Learned wind</option><option value="std">Wind uncertainty</option><option value="error">Wind error</option><option value="truth">True wind</option><option value="clarity">Clarity</option><option value="task">Task mask</option></select></label></div><div class="gpv-legend"></div><div class="gpv-chart-host gpv-pair"></div><div class="gpv-result"></div><div class="gpv-scroll gpv-outcomes"></div>`;
  const $=q=>host.querySelector(q),duration=Math.max(...records.map(r=>r.summary.simulated_seconds));let t=0,playing=false,last=null;
  $('[data-time]').max=duration;
  const runTitle=(r,i)=>group.comparison?group.labels[i]:`Scenario ${r.scenario}`;
  const cards=records.map((r,i)=>{const article=document.createElement('article');article.innerHTML=`<header><h3>${runTitle(r,i)}</h3><p></p></header><canvas width="960" height="720" role="img" aria-label="Scenario ${r.scenario}: recorded robot and wind belief"></canvas>`;$('.gpv-pair').append(article);return article;});
  for(const scenario of (group.comparison?[]:[1,2]))if(!records.some(r=>r.scenario===scenario)){const pending=document.createElement('article');pending.innerHTML=`<header><h3>Scenario ${scenario}</h3><p>${group.absent_run_label||'Queued · no recording yet'}</p></header><div class="gpv-pending-map" role="status">${group.absent_run_label?.startsWith('Not run')?'No trajectory is claimed for this scenario.':'The matching run will appear here after execution begins.'}</div>`;$('.gpv-pair').append(pending);}
  const chartMax=Math.ceil(Math.max(1,...records.flatMap(r=>r.belief_frames.flatMap(b=>[b.rmse_m_s,b.task_rmse_m_s||0,b.task_vector_std_m_s||0,b.route_rmse_m_s||0,b.route_vector_std_m_s||0]))));
  const charts=records.map((r,i)=>{const c=document.createElement('div');const threshold=r.summary.gate_kind==='whole-map RMSE'?r.summary.rmse_threshold_m_s:(r.summary.task_std_threshold_m_s??.2),handoff=r.summary.gate_time_s;c.innerHTML=`<svg class="gpv-chart" viewBox="0 0 600 165" role="img" aria-label="Scenario ${r.scenario} wind RMSE over time"><text x="42" y="16" font-size="12" fill="#20394a">Wind vector RMSE (m/s)</text><rect x="42" y="26" width="${538*(handoff??r.summary.simulated_seconds)/Math.max(1,duration)}" height="106" fill="#fff0d6"/><path d="M42 26V132H580" stroke="#9aabb9" fill="none"/><path d="M42 ${132-threshold/chartMax*100}H580" stroke="#536b84" stroke-dasharray="4 4"/><polyline points="${r.belief_frames.map(b=>`${42+b.t/Math.max(1,duration)*538},${132-b.rmse_m_s/chartMax*100}`).join(' ')}" stroke="#b65c19" fill="none" stroke-width="2"/><line class="gpv-cursor" y1="25" y2="132" stroke="#174d73"/><text x="4" y="36" font-size="12">${chartMax}</text><text x="24" y="136" font-size="12">0</text><text x="42" y="154" font-size="12">0 s</text><text x="530" y="154" font-size="12">${fmt(duration,0)} s</text></svg>`;$('.gpv-chart-host').append(c);return c;});
  charts.forEach((c,i)=>{const r=records[i],p=document.createElement('p');p.className='gpv-series-legend';p.textContent=`Shaded interval: wind learning · ${r.summary.task_std_threshold_m_s===0?'MBD handoff disabled for this calibration':'MBD starts '+(r.summary.gate_time_s===null?'after the gate is reached':fmt(r.summary.gate_time_s,1)+' s')}${r.summary.gate_kind==='whole-map RMSE'?' · RMSE threshold '+fmt(r.summary.rmse_threshold_m_s,2)+' m/s':''}${r.summary.validated_handoff?' · forecast validation required':''}`;c.append(p);});
  const returnToB=r=>{const first=r.summary.first_visit_s.A;if(first===null)return 'A not reached';const [lo,hi]=env.regions.find(z=>z.label==='B').bounds;const row=r.samples.find(z=>z[0]>first&&z[1]>=lo[0]&&z[1]<=hi[0]&&z[2]>=lo[1]&&z[2]<=hi[1]);return row?`${fmt(row[0],1)} s`:'No return recorded';};
  const monitoringVisit=(r,label)=>{const gate=r.summary.gate_time_s;if(gate===null)return 'MBD not started';const [lo,hi]=env.regions.find(z=>z.label===label).bounds;const row=r.samples.find(z=>z[0]>=gate&&z[1]>=lo[0]&&z[1]<=hi[0]&&z[2]>=lo[1]&&z[2]<=hi[1]);return row?`${fmt(row[0],1)} s`:'Not reached during MBD';};
  const executedMbdSeconds=(r,acceptedOnly=false)=>r.samples.slice(1).reduce((seconds,row,i)=>{
   const previous=r.samples[i],plan=at(r.plans,(row[0]+previous[0])/2);
   return seconds+(plan?.phase==='monitoring'&&(!acceptedOnly||(!plan.execution_fallback&&!plan.validation?.no_feasible))?row[0]-previous[0]:0);
  },0);
  records.forEach((r,i)=>{
   if(!r.belief_frames.some(b=>Number.isFinite(b.task_vector_std_m_s)))return;
   const svg=charts[i].querySelector('svg');
   const isRoute=r.summary.gate_kind==='route uncertainty';
   for(const [key,color,dash] of (isRoute?[['route_rmse_m_s','#8c2d04',''],['route_vector_std_m_s','#175e91','5 3']]:[['task_rmse_m_s','#8c2d04',''],['task_vector_std_m_s','#175e91','5 3']])){
    const line=document.createElementNS('http://www.w3.org/2000/svg','polyline');
    line.setAttribute('points',r.belief_frames.filter(b=>Number.isFinite(b[key])).map(b=>`${42+b.t/Math.max(1,duration)*538},${132-b[key]/chartMax*100}`).join(' '));
    line.setAttribute('stroke',color);line.setAttribute('fill','none');line.setAttribute('stroke-width','2');line.setAttribute('stroke-dasharray',dash);svg.append(line);
   }
   svg.querySelector('text').textContent='Wind error and uncertainty (m/s)';
   const legend=document.createElement('p');legend.className='gpv-series-legend';legend.textContent=`Orange: whole-map RMSE · Brown: ${isRoute?'route':'task'} RMSE · Dashed blue: ${isRoute?'route':'task'} posterior uncertainty · ${r.summary.task_std_threshold_m_s===0?'Handoff disabled for calibration':'Horizontal: '+fmt(r.summary.task_std_threshold_m_s??.2,2)+' m/s'}`;charts[i].append(legend);
  });
  const warmDecisionCycles=r=>{
   const cycles=new Map();
   for(const p of r.plans){if(!cycles.has(p.t))cycles.set(p.t,[]);cycles.get(p.t).push(p);}
   return [...cycles.values()].filter(group=>!group.some(p=>p.compilation_included)).map(group=>group.at(-1));
  };
  const cycleTime=r=>{
   const values=warmDecisionCycles(r).map(p=>p.wall_seconds).sort((a,b)=>a-b);
   return values.length?`${fmt(median(values),3)} / ${fmt(values.at(-1),3)} s`:'No warm cycle';
  };
  const requestTime=(r,phase='monitoring',full=false)=>{
   const values=r.plans.filter(p=>p.phase===phase&&!p.compilation_included&&p.timing).map(p=>full?p.wall_seconds:Object.entries(p.timing).filter(([k])=>!['posterior','safety_belief_refresh'].includes(k)).reduce((a,[,v])=>a+v,0)).sort((a,b)=>a-b);
   return values.length?`${fmt(median(values),3)} / ${fmt(values.at(-1),3)} s`:phase==='monitoring'?'MBD not started / no warm call':'Not recorded separately';
  };
  const rows=[['Recorded duration',r=>`${fmt(r.summary.simulated_seconds,1)} / ${r.summary.planned_seconds} s`],['Survey duration',r=>`${fmt(r.summary.survey_duration_s??r.summary.gate_time_s??r.summary.simulated_seconds,1)} s`],['Requested MBD duration after survey',r=>r.summary.requested_monitoring_duration_s?`${r.summary.requested_monitoring_duration_s} s`:'Within the fixed total duration'],['Contact',r=>r.summary.contact?`${fmt(r.summary.contact.time_s)} s`:'None recorded'],['First visit to region A',r=>r.summary.first_visit_s.A===null?'Not reached':`${fmt(r.summary.first_visit_s.A,1)} s`],['Return to B after A',returnToB],['Final wind RMSE',r=>`${fmt(r.summary.final_rmse_m_s,3)} m/s`],['Task-weighted wind error',r=>`${fmt(r.summary.final_task_rmse_m_s,3)} m/s`],['Task-weighted posterior uncertainty',r=>`${fmt(r.summary.final_task_vector_std_m_s,3)} m/s`],['Route wind error',r=>`${fmt(r.summary.final_route_rmse_m_s,3)} m/s`],['Route posterior uncertainty',r=>`${fmt(r.summary.final_route_vector_std_m_s,3)} m/s`],['Handoff criterion',r=>(r.summary.gate_kind||'whole-map RMSE')+(r.summary.validated_handoff?' + forecast validation':'')],['Accumulated monitoring cost',r=>`${fmt(r.summary.integrated_deficit,3)}${r.summary.completed?'':' · partial recording'}`],['Final clarity · A / B',r=>`${fmt(r.samples.at(-1)[15],3)} / ${fmt(r.samples.at(-1)[16],3)}`],['Minimum clearance',r=>`${fmt(r.summary.min_clearance_m,3)} m`],['Infeasible commands',r=>r.summary.infeasible_execution_steps],['MBD handoff eligible',r=>r.summary.gate_time_s===null?'Accuracy threshold not reached':`${fmt(r.summary.gate_time_s,1)} s`],['Monitoring phase (including hold)',r=>`${fmt(executedMbdSeconds(r),1)} s`],['Accepted MBD execution',r=>`${fmt(executedMbdSeconds(r,true),1)} s`],['Targeted learning requests after handoff',r=>r.plans.filter(p=>p.execution_mode==='targeted wind learning').length],[`Decision cycles over ${group.planning_budget_s||1} s target`,r=>warmDecisionCycles(r).filter(p=>p.wall_seconds>(group.planning_budget_s||1)).length],['15 s MBD request · median / max',r=>requestTime(r)],['MBD update + request · median / max',r=>requestTime(r,'monitoring',true)],['Learning decision · median / max',r=>requestTime(r,'learning')],['Actual update + decision · median / max',cycleTime],['Stop reason',r=>r.summary.stop_reason||(r.summary.completed?'Completed':'Partial checkpoint · see experiment status')]];
  rows.splice(6,0,['First MBD-phase visit to A',r=>monitoringVisit(r,'A')],['First MBD-phase visit to B',r=>monitoringVisit(r,'B')]);
  $('.gpv-outcomes').innerHTML=`<table><thead><tr><th>Measured outcome</th>${records.map((r,i)=>`<th>${runTitle(r,i)}</th>`).join('')}</tr></thead><tbody>${rows.map(([label,fn])=>`<tr><th>${label}</th>${records.map(r=>`<td>${fn(r)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  function draw(){
   const layer=$('[data-map]').value,ramp=['error','clarity','task'].includes(layer)?warm:blues,max=['clarity','task'].includes(layer)?1:layer==='std'?1.8:layer==='error'?3:18;
   $('.gpv-legend').innerHTML=`<strong>${{mean:'Learned wind speed + direction',truth:'True stationary wind + direction',std:'Posterior standard deviation σ',error:'Wind vector error',clarity:'Clarity q',task:'Latest acquisition mask · normalized by its maximum'}[layer]}</strong><span>0</span><i class="gpv-ramp" style="background:linear-gradient(90deg,${ramp.join(',')})"></i><span>${max} ${['clarity','task'].includes(layer)?'':'m/s'} ${layer==='task'?'· updated at recorded learning decisions':'· fixed scale'}${layer==='error'?' (values above 3 saturate)':layer==='std'?' (values above 1.8 saturate)':''}</span>`;
   records.forEach((r,i)=>{
    const time=Math.min(t,r.summary.simulated_seconds),b=at(r.belief_frames,time),s=at(r.samples,time,z=>z[0]),q=at(r.clarity_frames,time).q,plan=at(r.plans,time),truth=manifest.truth[r.scenario];
    const tf=at(r.task_frames||[],time),task=tf?.weights||(r.summary.learning_policy==='all'?Array.from({length:nx*ny},(_,k)=>free.has(k)?1:0):[]),taskMax=Math.max(1e-12,...task);
    const canvas=cards[i].querySelector('canvas'),ctx=canvas.getContext('2d');ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,960,720);ctx.scale(120,120);
    const pos=p=>[p[0],6-p[1]];
    env.free_indices.forEach((idx,j)=>{let v=layer==='task'?(task[idx]||0)/taskMax:layer==='clarity'?q[j]:layer==='std'?b.std[idx]:layer==='error'?Math.hypot(b.mean[idx][0]-truth[idx][0],b.mean[idx][1]-truth[idx][1]):Math.hypot(...(layer==='truth'?truth[idx]:b.mean[idx]));ctx.fillStyle=color(v,max,ramp);ctx.fillRect((idx%nx)*env.dx_m,6-(Math.floor(idx/nx)+1)*env.dx_m,env.dx_m+.003,env.dx_m+.003);});
    ctx.lineWidth=.015;ctx.strokeStyle='#46718a';ctx.font='.16px system-ui';env.regions.forEach(r=>{const[a,b]=r.bounds;ctx.strokeRect(a[0],6-b[1],b[0]-a[0],b[1]-a[1]);ctx.fillStyle='#24475d';ctx.fillText('Region '+r.label,a[0]+.1,6-b[1]+.22);});
    ctx.fillStyle='#788791';env.obstacles.forEach(o=>{ctx.beginPath();ctx.arc(o[0],6-o[1],o[2],0,2*Math.PI);ctx.fill();});
    if(layer==='mean'||layer==='truth'){ctx.strokeStyle='#153e5b';ctx.lineWidth=.012;for(let y=2;y<ny;y+=4)for(let x=2;x<nx;x+=4){const k=y*nx+x;if(!free.has(k))continue;const w=layer==='truth'?truth[k]:b.mean[k],norm=Math.hypot(...w);if(norm<.1)continue;const px=(x+.5)*env.dx_m,py=6-(y+.5)*env.dx_m,dx=w[0]*.035,dy=-w[1]*.035;ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(px+dx,py+dy);ctx.lineTo(px+dx-.06*dx/norm/.035+.04*dy/norm/.035,py+dy-.06*dy/norm/.035-.04*dx/norm/.035);ctx.stroke();}}
    const line=(path,color,width,dash=[])=>{if(!path?.length)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();path.forEach((p,k)=>{const[x,y]=pos(p);k?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.setLineDash([]);};
    if(layer==='task'&&tf?.geometric_paths)for(const route of tf.geometric_paths)line(route,'#5a3a8b',.018,[.06,.035]);
    if(plan?.global_geometric_path)line(plan.global_geometric_path,'#9ca9bb',.012,[.03,.07]);
    const path=plan?.phase==='learning'?plan.path:plan?.predicted_states?.[0]?.map(z=>z.slice(0,2));line(path,'#586e91',.016,[.06,.04]);
    const trail=r.samples.filter((z,k)=>k%4===0&&z[0]<=time).map(z=>z.slice(1,3));line(trail,'white',.055);line(trail,'#005f73',.025);
    const [px,py]=pos(s.slice(1,3));ctx.fillStyle=s[18]?'#005f73':'#b13a1b';ctx.strokeStyle='white';ctx.lineWidth=.025;ctx.beginPath();ctx.arc(px,py,.065,0,2*Math.PI);ctx.fill();ctx.stroke();ctx.strokeStyle='#8296a5';ctx.lineWidth=.012;ctx.strokeRect(0,0,8,6);
    cards[i].querySelector('header p').textContent=`${fmt(time,1)} s · ${plan?.phase==='monitoring'?'MBD':'Wind learning'} · RMSE ${fmt(b.rmse_m_s,3)} m/s · ${b.observations} readings · ${!r.summary.simulated_seconds?'no execution recorded':s[18]?(s[11]?'QP feasible':'QP infeasible'):'stopped at contact'}`;
    const cursor=charts[i].querySelector('.gpv-cursor'),cx=42+time/Math.max(1,duration)*538;cursor.setAttribute('x1',cx);cursor.setAttribute('x2',cx);
   });
   $('[data-time]').value=t;$('output').textContent=`${fmt(t,1)} / ${fmt(duration,1)} s`;
  }
  const play=value=>{playing=value;$('[data-play]').textContent=value?'Ⅱ Pause':'▶ Play recordings';$('[data-play]').setAttribute('aria-pressed',String(value));};
  $('[data-play]').onclick=()=>{if(t>=duration)t=0;play(!playing);draw();};$('[data-reset]').onclick=()=>{t=0;play(false);draw();};$('[data-end]').onclick=()=>{t=duration;play(false);draw();};$('[data-time]').oninput=e=>{t=+e.target.value;play(false);draw();};$('[data-map]').onchange=draw;
  $('.gpv-result').textContent=group.result;draw();
  const observer=new IntersectionObserver(entries=>host.dataset.visible=String(entries[0].isIntersecting));observer.observe(host);
  function tick(now){const dt=last===null?0:Math.min(.2,(now-last)/1000);last=now;if(playing&&!document.hidden&&host.dataset.visible==='true'){t=Math.min(duration,t+dt*+$('[data-speed]').value);draw();if(t>=duration)play(false);}requestAnimationFrame(tick);}requestAnimationFrame(tick);
  }
  const begin=()=>{
   if(loaded)return;loaded=true;
   host.querySelector('[role="status"]').textContent='Loading recorded simulation data…';
   renderGroup().catch(error=>{
    loaded=false;host.innerHTML=introduction+`<p role="alert">Could not load this comparison: ${error.message}</p><button data-load>Retry loading</button>`;
    host.querySelector('[data-load]').onclick=begin;console.error(error);
   });
  };
  host.querySelector('[data-load]').onclick=begin;
  let visibleTimer;
  const loader=new IntersectionObserver(entries=>{
   clearTimeout(visibleTimer);
   if(entries.some(e=>e.isIntersecting))visibleTimer=setTimeout(()=>{loader.disconnect();begin();},400);
  },{rootMargin:'0px'});
  loader.observe(host);groupLoaders.set(group.id,begin);
  if(location.hash==='#'+group.id)begin();
 }
 const navigate=id=>{const target=document.getElementById(id);const archive=target?.closest('details');if(archive)archive.open=true;groupLoaders.get(id)?.();target?.scrollIntoView({behavior:'instant',block:'start'});};
 root.querySelector('.gpv-overview').addEventListener('click',event=>{const a=event.target.closest('a[href^="#gpv-"]');if(!a)return;event.preventDefault();history.replaceState(null,'',a.hash);navigate(a.hash.slice(1));});
 window.addEventListener('hashchange',()=>navigate(location.hash.slice(1)));
 if(location.hash.startsWith('#gpv-'))navigate(location.hash.slice(1));
 }catch(e){root.querySelector('.gpv-experiments').innerHTML=`<p role="alert">Could not load the validation recordings: ${e.message}</p>`;console.error(e);}
})();
