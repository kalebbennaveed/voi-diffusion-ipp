'use strict';
(async()=>{
 const dir='data/rmse-learning-300';
 async function read(n){const r=await fetch(`${dir}/${n}`);if(!r.ok)throw Error(`${n}: ${r.status}`);return n.endsWith('.gz')?new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).json():r.json();}
 const at=(a,t,f=z=>z.t)=>{let l=0,h=a.length-1;while(h-l>1){const m=(l+h)>>1;if(f(a[m])<=t)l=m;else h=m;}return a[f(a[h])<=t?h:l];};
 const n=(v,d=2)=>Number.isFinite(v)?v.toFixed(d):'—';
 const xy=p=>[100*p[0],600-100*p[1]],line=a=>a.map(p=>xy(p).join(',')).join(' ');
 const ramp=['#f7fbff','#deebf7','#9ecae1','#4292c6','#2171b5','#08306b'];
 const warm=['#fff7ec','#fee8c8','#fdbb84','#fc8d59','#d7301f','#7f0000'];
 const colors=(v,max,colors)=>{const z=Math.max(0,Math.min(1,v/max))*(colors.length-1),i=Math.min(colors.length-2,Math.floor(z)),f=z-i;return [1,3,5].map(k=>Math.round(parseInt(colors[i].slice(k,k+2),16)*(1-f)+parseInt(colors[i+1].slice(k,k+2),16)*f));};
 try{
 const [env,manifest]=await Promise.all([read('environment.json'),read('manifest.json')]);
 const records=await Promise.all(manifest.runs.sort((a,b)=>a.scenario-b.scenario).map(r=>read(r.file)));
 const [nx,ny]=env.grid,free=new Set(env.free_indices);
 for(const stage of ['learning','mbd']){
  const root=document.getElementById(stage==='learning'?'wind-learning-rmse':'mbd-rmse');if(!root)continue;
  const host=root.querySelector('.rmse-view'),isLearning=stage==='learning';let time=0,playing=false,last=null;
  const windows=records.map(r=>isLearning?[0,r.summary.gate_time_s??r.summary.simulated_seconds]:[r.summary.gate_time_s??r.summary.simulated_seconds,r.summary.simulated_seconds]);
  const duration=Math.max(0,...windows.map(w=>w[1]-w[0]));
  const chartMax=Math.ceil(Math.max(1,...records.flatMap(r=>r.belief_frames.map(b=>b.rmse_m_s))));
  host.innerHTML='<div class="learning-pair rmse-pair"></div><div class="learning-controls"><button data-rmse="play">▶ Play both scenarios</button><button data-rmse="reset">Restart</button><input type="range" data-rmse="time" aria-label="'+stage+' elapsed time" min="0" max="'+duration+'" step=".2" value="0"><output></output><label>Speed <select data-rmse="speed"><option>1</option><option selected>4</option><option>10</option></select>×</label><label>Map <select data-rmse="map"><option value="std">Wind uncertainty σ</option><option value="mean">Learned wind + vectors</option><option value="error">Wind error against truth</option><option value="clarity">Clarity</option></select></label><button data-rmse="end">End</button></div><div class="rmse-legend learning-legend"></div><div class="learning-pair rmse-charts"></div><div class="learning-results rmse-results"></div>';
  const $=s=>host.querySelector(s);
  const panels=records.map((r,i)=>{
   const card=document.createElement('article');card.className='learning-card';const blocked=!isLearning&&r.summary.gate_time_s===null;
   const shapes=env.obstacles.map(o=>`<circle cx="${100*o[0]}" cy="${600-100*o[1]}" r="${100*o[2]}" fill="#75858c"/>`).join('');
   const regions=env.regions.map(r=>{const[a,b]=r.bounds;return `<rect x="${100*a[0]}" y="${600-100*b[1]}" width="${100*(b[0]-a[0])}" height="${100*(b[1]-a[1])}" fill="none" stroke="#587c91" stroke-width="2"/><text class="learning-region" x="${100*a[0]+12}" y="${600-100*b[1]+25}">Region ${r.label}</text>`;}).join('');
   const aid=`rmse-arrow-${stage}-${i}`;
   card.innerHTML=`<header><h3>Scenario ${r.scenario} · ${isLearning?'Wind learning':'MBD monitoring'}</h3><p class="rmse-status"></p></header><div class="learning-map"><svg viewBox="-4 -4 808 608" role="img" aria-label="${stage} scenario ${r.scenario}"><defs><marker id="${aid}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#163e60"/></marker></defs><rect width="800" height="600" fill="#f7fbff"/><image class="rmse-heat" width="800" height="600"/>${regions}${shapes}<g class="rmse-vectors"></g><polyline class="learning-route"/><polyline class="rmse-trail-halo" stroke="white" stroke-width="6" fill="none"/><polyline class="rmse-trail" stroke="#086c7c" stroke-width="3" fill="none"/><circle class="rmse-dot" r="6" fill="#086c7c" stroke="white" stroke-width="2"/><rect width="800" height="600" fill="none" stroke="#889eaa"/></svg></div><p class="learning-live rmse-live"></p>${blocked?'<p class="learning-note learning-warning">MBD has not started. The learning stage did not meet the 0.20 m/s accuracy requirement. This map shows its final state, not an MBD trajectory.</p>':''}`;
   $('.rmse-pair').append(card);
   const chart=document.createElement('div');chart.className='learning-chart';const f=r.belief_frames.filter(b=>b.t<=windows[i][1]&&b.t>=windows[i][0]);
   const max=chartMax,span=Math.max(1,duration);
   chart.innerHTML=`<p>Scenario ${r.scenario} · whole-map wind vector RMSE (m/s)</p><svg viewBox="0 0 570 140"><path d="M35 8V110H550" fill="none" stroke="#8fa4b3"/><path d="M35 ${110-.2/max*100}H550" stroke="#047857" stroke-dasharray="5 3"/><text x="400" y="${104-.2/max*100}" fill="#047857" font-size="11">Start MBD at ≤ 0.20</text><polyline points="${f.map(b=>`${35+(b.t-windows[i][0])/span*515},${110-b.rmse_m_s/max*100}`).join(' ')}" fill="none" stroke="#b45309" stroke-width="2"/><line class="rmse-cursor" y1="8" y2="110" stroke="#173247"/><text x="5" y="18" font-size="11">${max.toFixed(1)}</text><text x="17" y="113" font-size="11">0</text><text x="35" y="132" font-size="11">0 s</text><text x="505" y="132" font-size="11">${span.toFixed(0)} s</text></svg>`;
   if(!blocked) $('.rmse-charts').append(chart);
   return{r,card,chart,blocked,heatKey:'',aid,span};
  });
  const fields=[['Accuracy threshold',r=>'0.20 m/s'],['Best whole-map RMSE',r=>`${n(Math.min(...r.belief_frames.map(b=>b.rmse_m_s)))} m/s`],['Final whole-map RMSE',r=>`${n(r.summary.final_rmse_m_s)} m/s`],['MBD switch time',r=>r.summary.gate_time_s===null?'Not reached':`${n(r.summary.gate_time_s,1)} s`],['Contact',r=>r.summary.contact?`Yes · ${n(r.summary.contact.time_s,2)} s`:'None recorded'],['Infeasible filter commands',r=>r.summary.infeasible_execution_steps],['Recorded duration',r=>`${n(r.summary.simulated_seconds,1)} / ${r.summary.planned_seconds} s`]];
  $('.rmse-results').innerHTML=`<table><thead><tr><th>Measured outcome</th><th>Scenario 1</th><th>Scenario 2</th></tr></thead><tbody>${fields.map(([label,fn])=>`<tr><th>${label}</th>${records.map(r=>`<td>${fn(r)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  function draw(){
   const map=$('[data-rmse=map]').value,max=map==='std'?1.8:map==='clarity'?1:10,colorset=map==='error'||map==='clarity'?warm:ramp;
   $('.rmse-legend').innerHTML=`<b>${{std:'POAM uncertainty σ',mean:'Learned wind speed',error:'Vector wind error',clarity:'Clarity q'}[map]}</b><span>0</span><i class="learning-ramp" style="background:linear-gradient(90deg,${colorset.join(',')})"></i><span>${max}${map==='clarity'?'':' m/s'}${map==='error'?' (saturates at 10)':''}</span>`;
   panels.forEach((p,i)=>{
    const [start,end]=windows[i],t=Math.min(end,start+time),r=p.r,s=at(r.samples,t,z=>z[0]),b=at(r.belief_frames,t),q=at(r.clarity_frames,t).q;
    const plan=at(r.plans,t),key=map+':'+(map==='clarity'?Math.floor(t):b.t);
    if(p.heatKey!==key){p.heatKey=key;const c=document.createElement('canvas');c.width=nx;c.height=ny;const ctx=c.getContext('2d'),pixels=ctx.createImageData(nx,ny);env.free_indices.forEach((idx,j)=>{let value=map==='clarity'?q[j]:map==='std'?b.std[idx]:map==='mean'?Math.hypot(...b.mean[idx]):Math.hypot(b.mean[idx][0]-manifest.truth[r.scenario][idx][0],b.mean[idx][1]-manifest.truth[r.scenario][idx][1]);const off=((ny-1-Math.floor(idx/nx))*nx+idx%nx)*4;colors(value,max,colorset).forEach((v,k)=>pixels.data[off+k]=v);pixels.data[off+3]=255;});ctx.putImageData(pixels,0,0);p.card.querySelector('.rmse-heat').setAttribute('href',c.toDataURL());}
    let arrows='';if(map==='mean')for(let y=2;y<ny;y+=4)for(let x=2;x<nx;x+=4){const idx=y*nx+x;if(!free.has(idx))continue;const w=b.mean[idx],z=xy([(x+.5)*env.dx_m,(y+.5)*env.dx_m]);if(Math.hypot(...w)<.08)continue;arrows+=`<path d="M${z[0]} ${z[1]}l${4*w[0]} ${-4*w[1]}" stroke="#163e60" stroke-width="1.5" marker-end="url(#${p.aid})"/>`;}
    p.card.querySelector('.rmse-vectors').innerHTML=arrows;
    const trail=p.blocked?'':line(r.samples.filter((z,k)=>k%4===0&&z[0]>=start&&z[0]<=t).map(z=>z.slice(1,3)));
    for(const cls of ['rmse-trail','rmse-trail-halo'])p.card.querySelector('.'+cls).setAttribute('points',trail);
    p.card.querySelector('.learning-route').setAttribute('points',p.blocked?'':line(plan.phase==='learning'?plan.path:plan.predicted_states[0].map(z=>z.slice(0,2))));
    const pos=xy(s.slice(1,3));p.card.querySelector('.rmse-dot').setAttribute('cx',pos[0]);p.card.querySelector('.rmse-dot').setAttribute('cy',pos[1]);
    p.card.querySelector('.rmse-status').textContent=`Wind RMSE ${n(b.rmse_m_s)} m/s · required ≤ 0.20 m/s · ${p.blocked?'MBD not started':isLearning?'uncertainty waypoints + RPCBF':'MBD + RPCBF'}`;
    p.card.querySelector('.rmse-live').textContent=`${b.observations} actual measurements · mean σ ${n(b.mean_std)} m/s · ${s[18]?(s[11]?'QP feasible':'QP infeasible: fallback'):'integration stopped'}${r.summary.contact&&t>=r.summary.contact.time_s?' · stopped at contact':''}${!isLearning&&!p.blocked?` · ${plan.evaluations} MBD evaluations · initial / optimized cost ${n(plan.initial_cost)} / ${n(plan.optimized_training_cost)}`:''}`;
    const cursor=p.chart.querySelector('.rmse-cursor'),cx=35+(t-start)/p.span*515;cursor.setAttribute('x1',cx);cursor.setAttribute('x2',cx);
   });
   $('[data-rmse=time]').value=time;$('output').textContent=duration===0?'No MBD execution recorded':`${n(time,1)} / ${n(duration,1)} s ${isLearning?'learning':'MBD'}`;
  }
  function play(value){playing=value;$('[data-rmse=play]').textContent=value?'Ⅱ Pause':'▶ Play both scenarios';}
  for(const sel of ['play','reset','end','time'])$(`[data-rmse=${sel}]`).disabled=duration===0;
  $('[data-rmse=play]').onclick=()=>{if(time>=duration)time=0;play(!playing);draw();};
  $('[data-rmse=reset]').onclick=()=>{time=0;play(false);draw();};$('[data-rmse=end]').onclick=()=>{time=duration;play(false);draw();};
  $('[data-rmse=time]').oninput=e=>{time=+e.target.value;play(false);draw();};$('[data-rmse=map]').onchange=draw;
  if(duration===0){$('[data-rmse=play]').textContent='Accuracy requirement unmet';for(const name of ['reset','end','time']) $(`[data-rmse=${name}]`).style.display='none';$('[data-rmse=speed]').parentElement.style.display='none';}
  draw();function tick(now){const dt=last===null?0:Math.min(.1,(now-last)/1000);last=now;if(playing&&!document.hidden){time=Math.min(duration,time+dt*+$('[data-rmse=speed]').value);draw();if(time>=duration)play(false);}requestAnimationFrame(tick);}requestAnimationFrame(tick);
 }
 }catch(e){document.querySelectorAll('.rmse-view').forEach(x=>x.innerHTML=`<p role="alert">Could not load RMSE experiment: ${e.message}</p>`);console.error(e);}
})();
