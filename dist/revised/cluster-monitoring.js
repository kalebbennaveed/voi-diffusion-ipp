'use strict';
// Deterministic recorded simulations; the browser only replays saved states.
async function knownComparison(root){
  const scenario=Number(root.dataset.knownScenario),directory='data/monitoring-cluster-300';
  const find=selector=>root.querySelector(selector),colors=['#147861','#c46a35'];
  const names=['Disturbance-aware MBD','MBD + execution RPCBF'];
  const descriptions=['Known wind and relaxed RPCBF inside planning.','Still-air planning; known-wind RPCBF during execution.'];
  let records=[],panels=[],env,manifest,time=0,playing=false,last=null,loadVersion=0;
  let height=600,nx=64,ny=48;
  const duration=300;
  const screen=p=>[p[0]*100,height-p[1]*100];
  const line=points=>points.map(p=>screen(p).map(v=>v.toFixed(2)).join(',')).join(' ');
  function at(rows,t,key){let a=0,b=rows.length-1;while(b-a>1){const c=(a+b)>>1;if(key(rows[c])<=t)a=c;else b=c;}return key(rows[b])<=t?b:a;}
  function sample(rows,t){const i=at(rows,t,r=>r[0]),a=rows[i],b=rows[Math.min(i+1,rows.length-1)],f=b[0]>a[0]?(t-a[0])/(b[0]-a[0]):0;return a.map((v,k)=>[11,18].includes(k)?v:v+f*(b[k]-v));}
  const stops=['fff7ec','fee8c8','fdbb84','fc8d59','e34a33','b30000','7f0000'].map(h=>[0,2,4].map(k=>parseInt(h.slice(k,k+2),16)));
  function shade(value,max){const u=Math.min(1,Math.max(0,value/max))*6,i=Math.min(5,Math.floor(u)),f=u-i;return stops[i].map((v,k)=>Math.round(v+f*(stops[i+1][k]-v)));}
  function play(value){playing=value;find('[data-action="play"]').textContent=value?'Ⅱ Pause':'▶ Play comparison';find('[data-action="play"]').setAttribute('aria-pressed',String(value));}
  async function read(name){const r=await fetch(`${directory}/${name}`,{cache:'no-store'});if(!r.ok)throw Error(`Could not load ${name} (${r.status})`);if(!name.endsWith('.gz'))return r.json();const bytes=new Uint8Array(await r.arrayBuffer());return bytes[0]===31?new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).json():JSON.parse(new TextDecoder().decode(bytes));}
  function makePanel(record,index){
    const card=document.createElement('article');card.className='known-map-card';
    const clip=`known-${scenario}-${index}`,s=record.summary;
    const regions=env.regions.map(r=>{const [lo,hi]=r.bounds;return `<rect x="${lo[0]*100}" y="${height-hi[1]*100}" width="${(hi[0]-lo[0])*100}" height="${(hi[1]-lo[1])*100}" fill="none" stroke="${r.label==='A'?'#2563eb':'#b45309'}" stroke-width="2"/><text x="${lo[0]*100+16}" y="${height-hi[1]*100+27}" class="region-label monitor-region-label">Region ${r.label}</text>`;}).join('');
    const obstacles=env.obstacles.map(o=>`<circle cx="${o[0]*100}" cy="${height-o[1]*100}" r="${o[2]*100}" fill="#73828a"/>`).join('');
    card.innerHTML=`<div class="robot-card-heading"><h4>${names[index]}</h4><span class="outcome ${s.contact?'collision':s.infeasible_execution_steps?'stopped':''}">${s.contact?'Contact recorded':'300 s · no contact'}</span></div><p class="monitor-method">${descriptions[index]}</p><div class="robot-map"><svg viewBox="-5 -5 810 ${height+10}" role="img" aria-label="Scenario ${scenario}, ${names[index]}, executed trajectory and clarity"><defs><clipPath id="${clip}"><rect width="800" height="${height}"/></clipPath></defs><rect width="800" height="${height}" fill="#fff7ec"/><image class="known-heat" width="800" height="${height}"/>${regions}${obstacles}<g clip-path="url(#${clip})"><g class="known-seeds"></g><polyline class="known-forecast" fill="none" stroke="#475569" stroke-width="1.5" stroke-dasharray="5 5"/><polyline class="known-halo" fill="none" stroke="white" stroke-width="6"/><polyline class="known-trace" fill="none" stroke="${colors[index]}" stroke-width="3"/><circle class="known-dot" r="6.5" fill="${colors[index]}" stroke="white" stroke-width="2"/><path class="known-contact" d="M-8 -8L8 8M-8 8L8 -8" stroke="#991b1b" stroke-width="3" display="none"/></g><rect width="800" height="${height}" fill="none" stroke="#85938d"/></svg></div>`;
    const details=document.createElement('div');details.className='known-details';
    details.innerHTML=`<div class="live-metrics"><div><span>Clarity · A</span><b data-metric="a"></b></div><div><span>Clarity · B</span><b data-metric="b"></b></div><div><span>Mean deficit</span><b data-metric="deficit"></b></div><div><span>Accumulated deficit</span><b data-metric="cost"></b></div></div><p class="known-live"></p><p class="known-summary">First visit to A: <strong>${s.first_visit_s.A===null?'not reached':s.first_visit_s.A.toFixed(2)+' s'}</strong> · first route <strong>${s.first_transfer?.route||'not completed'}</strong> · minimum clearance <strong>${s.min_clearance_m.toFixed(3)} m</strong> · infeasible filter steps <strong>${s.infeasible_execution_steps}</strong>${s.contact?` · contact at ${s.contact.time_s.toFixed(2)} s`:''}</p><details><summary>Current planning decision and runtime</summary><div class="known-decision"></div></details><div class="known-chart"></div>`;
    const rows=record.samples.filter((_,i)=>i%10===0),points=k=>rows.map(s=>`${35+s[0]/duration*530},${125-s[k]*105}`).join(' ');
    details.querySelector('.known-chart').innerHTML=`<p><span style="color:#2563eb">━ Region A</span> &nbsp; <span style="color:#b45309">━ Region B</span> · mean clarity · dashed line: 0.70 target</p><svg viewBox="0 0 590 155" role="img" aria-label="Region clarity over the complete 300 second mission"><path d="M35 20V125H565" fill="none" stroke="#94a3b8"/><path d="M35 51.5H565" stroke="#64748b" stroke-dasharray="5 4"/><polyline points="${points(15)}" fill="none" stroke="#2563eb" stroke-width="1.7"/><polyline points="${points(16)}" fill="none" stroke="#b45309" stroke-width="1.7"/><line class="known-chart-time" y1="20" y2="125" stroke="#0f172a"/><text x="13" y="24">1</text><text x="13" y="129">0</text><text x="32" y="145">0 s</text><text x="532" y="145">300 s</text></svg>`;
    return {record,card,details,heatKey:'',planIndex:-1};
  }
  function draw(){
    const layer=find('[data-control="layer"]').value;
    for(const p of panels){
      const s=sample(p.record.samples,time),pos=screen(s.slice(1,3));
      for(const name of ['known-trace','known-halo'])p.card.querySelector('.'+name).setAttribute('points',line([...p.record.samples.filter(r=>r[0]<time).map(r=>r.slice(1,3)),s.slice(1,3)]));
      const dot=p.card.querySelector('.known-dot');dot.setAttribute('cx',pos[0]);dot.setAttribute('cy',pos[1]);
      const contact=p.card.querySelector('.known-contact');contact.setAttribute('transform',`translate(${pos.join(' ')})`);contact.setAttribute('display',s[18]?'none':'');
      const fi=at(p.record.clarity_frames,time,r=>r.t),heatKey=layer+fi;
      if(p.heatKey!==heatKey){p.heatKey=heatKey;const c=document.createElement('canvas');c.width=nx;c.height=ny;const ctx=c.getContext('2d'),pixels=ctx.createImageData(nx,ny),q=p.record.clarity_frames[fi].q;
        env.free_indices.forEach((cell,n)=>{const i=((ny-1-Math.floor(cell/nx))*nx+cell%nx)*4,v=layer==='deficit'?Math.max(0,env.targets[n]-q[n]):q[n],rgb=shade(v,layer==='deficit'?.7:1);for(let k=0;k<3;k++)pixels.data[i+k]=rgb[k];pixels.data[i+3]=255;});ctx.putImageData(pixels,0,0);p.card.querySelector('.known-heat').setAttribute('href',c.toDataURL());}
      const values={a:s[15],b:s[16],deficit:s[13],cost:s[14]};for(const [key,v] of Object.entries(values))p.details.querySelector(`[data-metric="${key}"]`).textContent=v.toFixed(3);
      p.details.querySelector('.known-live').textContent=`Speed ${Math.hypot(s[3],s[4]).toFixed(2)} m/s · clearance ${Math.max(0,s[12]).toFixed(3)} m · ${s[18]?'executing':'stopped after contact'} · known stationary wind`;
      const cursor=p.details.querySelector('.known-chart-time');cursor.setAttribute('x1',35+time/duration*530);cursor.setAttribute('x2',35+time/duration*530);
      const pi=at(p.record.plans,time,r=>r.t),plan=p.record.plans[pi];
      if(p.planIndex!==pi){p.planIndex=pi;p.card.querySelector('.known-forecast').setAttribute('points',line(plan.predicted_states[0].map(r=>r.slice(0,2))));p.card.querySelector('.known-seeds').innerHTML=(plan.initialization.coarse_paths||[]).map(path=>`<polyline points="${line(path)}" fill="none" stroke="#86198f" stroke-width="1.6" stroke-dasharray="3 6"/>`).join('');
        p.details.querySelector('.known-decision').innerHTML=`<p>Plan at ${plan.t.toFixed(0)} s · ${manifest.planning_horizon_s} s horizon · ${plan.selected_origin==='diffusion'?'diffusion proposal selected':'initialized trajectory retained'}.</p><p>Total ${plan.wall_seconds.toFixed(3)} s${plan.timing.compilation_included?' including compilation':''}. Geometry ${(plan.timing.initialization*1000).toFixed(1)} ms · MBD + selected forecast ${(plan.timing.optimization_validation*1000).toFixed(1)} ms. ${plan.evaluations} search evaluations; one selected forecast for display.</p><p>Expected objective ${plan.expected_cost.toFixed(3)}. ${plan.predicted_contacts[0]?'Forecast includes contact.':'Forecast has no contact.'} All ${manifest.control_coordinates} acceleration coordinates are optimized. Wind uncertainty is zero; no posterior update or wind-sampling cost.</p>`;}
      p.card.querySelector('.known-forecast').style.display=find('[data-control="forecast"]').checked?'':'none';p.card.querySelector('.known-seeds').style.display=find('[data-control="seeds"]').checked?'':'none';
    }
    find('[data-control="time"]').value=time;find('output').textContent=`${time.toFixed(1)} / 300.0 s`;
    find('.known-scale-label').textContent=layer==='deficit'?'Deficit below target · 0 to 0.70':'Clarity q · 0 to 1';find('.known-scale-max').textContent=layer==='deficit'?'0.70 · larger deficit':'1.00 · higher clarity';
  }
  async function loadCase(){
    const ticket=++loadVersion;play(false);time=0;
    const id=find('[data-control="case"]').value;
    find('.known-status').textContent='Loading the selected initial condition…';
    for(const b of root.querySelectorAll('.robot-transport button'))b.disabled=true;
    const loaded=await Promise.all(['aware','baseline'].map(method=>read(`s${scenario}-${id}-${method}.json.gz`)));
    if(ticket!==loadVersion)return;
    records=loaded;for(const p of panels){p.card.remove();p.details.remove();}
    panels=records.map(makePanel);const pair=find('.known-pair');pair.before(find('.known-legend'));pair.prepend(...panels.map(p=>p.card));pair.append(...panels.map(p=>p.details));
    for(const b of root.querySelectorAll('button,input,select'))b.disabled=false;
    const selected=manifest.runs.filter(r=>r.scenario===scenario&&r.case===id);
    const timing=selected.map((r,i)=>`${names[i]}: ${Math.round(r.timing.median_s*1000)} ms median, ${Math.round(r.timing.max_s*1000)} ms max`);
    find('.known-runtime').textContent=`Measured warm planning — ${timing.join(' · ')}. Limit: 1 s.`;
    const condition=manifest.conditions.find(c=>c.id===id);
    find('.known-status').textContent=`${condition.label}. Start p=(${condition.state.slice(0,2).join(', ')}) m, v=(${condition.state.slice(2).join(', ')}) m/s. Two completed 300 s simulations.`;
    draw();
  }
  try{
    [manifest,env]=await Promise.all([read('manifest.json'),read('environment.json')]);
    height=100*env.bounds[1][1];[nx,ny]=env.grid;
    find('[data-control="case"]').innerHTML=manifest.conditions.map(c=>`<option value="${c.id}">${c.label}</option>`).join('');
    find('[data-control="case"]').addEventListener('change',()=>loadCase().catch(e=>find('.known-status').textContent=e.message));
    find('[data-action="play"]').addEventListener('click',()=>{if(time>=duration)time=0;play(!playing);draw();});
    find('[data-action="restart"]').addEventListener('click',()=>{time=0;play(true);draw();});
    find('[data-control="time"]').addEventListener('input',e=>{time=Number(e.target.value);play(false);draw();});
    for(const control of ['layer','forecast','seeds'])find(`[data-control="${control}"]`).addEventListener('change',draw);
    await loadCase();
    const rows=manifest.conditions.map(c=>{
      const a=manifest.runs.find(r=>r.scenario===scenario&&r.case===c.id&&r.method==='aware');
      const b=manifest.runs.find(r=>r.scenario===scenario&&r.case===c.id&&r.method==='baseline');
      return `<tr><th>${c.label}</th><td>${b.summary.first_planned_route_to_A||'none'}</td><td>${a.summary.first_transfer.route||'none'} / ${b.summary.first_transfer.route||'none'}</td><td>${a.summary.first_visit_s.A?.toFixed(1)||'—'} / ${b.summary.first_visit_s.A?.toFixed(1)||'—'}</td><td>${a.summary.integrated_deficit.toFixed(2)} / ${b.summary.integrated_deficit.toFixed(2)}</td><td>${a.summary.contact?'yes':'no'} / ${b.summary.contact?'yes':'no'}</td></tr>`;
    }).join('');
    find('.cluster-results').innerHTML=`<table class="monitor-all-results"><caption>All initial conditions · proposed / baseline</caption><thead><tr><th>Start</th><th>Baseline’s first plan to A</th><th>First completed B → A route</th><th>First A visit (s)</th><th>Deficit cost ↓</th><th>Contact</th></tr></thead><tbody>${rows}</tbody></table>`;
    function frame(now){const dt=last===null?0:Math.min(.1,(now-last)/1000);last=now;if(playing&&!document.hidden){time=Math.min(duration,time+dt*Number(find('[data-control="speed"]').value));draw();if(time===duration)play(false);}requestAnimationFrame(frame);}requestAnimationFrame(frame);
  }catch(error){find('.known-status').textContent=error.message;}
}
document.querySelectorAll('[data-known-scenario]').forEach(knownComparison);
