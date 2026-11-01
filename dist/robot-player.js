'use strict';
// Replay of computed four-state trajectories. Animation never changes physics.
(async function robotReview(){
  const root=document.querySelector('#robot-review'),status=root.querySelector('#robot-status');
  const $=id=>root.querySelector('#'+id),colors={unfiltered:'#c46a35',rpcbf:'#147861',still:'#82939e',reference:'#43545d'};
  const screen=p=>[p[0]*100,462.5-p[1]*100];
  const poly=(rows,x=1,y=2)=>rows.map(s=>screen([s[x],s[y]]).map(v=>v.toFixed(2)).join(',')).join(' ');
  let manifest,records=[],panels=[],fields={},time=0,horizon=1,playing=false,version=0,previous=null;
  const cache=new Map();
  function sample(rows,t){
    if(t<=rows[0][0])return rows[0];if(t>=rows.at(-1)[0])return rows.at(-1);
    let lo=0,hi=rows.length-1;while(hi-lo>1){const m=(lo+hi)>>1;if(rows[m][0]<=t)lo=m;else hi=m;}
    const a=rows[lo],b=rows[hi],f=(t-a[0])/(b[0]-a[0]);
    return a.map((v,i)=>i===14?v:v+(b[i]-v)*f);
  }
  function setPlaying(value){playing=value;$('robot-play').textContent=value?'Ⅱ Pause robots':'▶ Play robots';$('robot-play').setAttribute('aria-pressed',String(value));}
  function detour(summary,record){const crossing=(summary.passage_crossings||[]).filter(e=>e.direction===record?.direction).at(-1);return summary.target_region_reached&&crossing&&crossing.route!==record.route?crossing.route:null;}
  function tag(summary,record){return summary.status==='collision'?'collision':summary.min_clearance_m<.06||summary.infeasible_steps>0||detour(summary,record)?'caution':summary.target_region_reached?'reached':'stopped';}
  function label(summary,record){const via=detour(summary,record),finish=summary.target_region_reached?(via?`Target via ${via} detour`:'Target reached'):'Outside target';return summary.status==='collision'?'Contact · run stopped':summary.min_clearance_m<.06?`${finish} · margin breached`:summary.infeasible_steps>0?`${finish} · QP infeasibility`:via?finish:summary.target_region_reached?'Finished in target region':'Finished outside target region';}
  function panel(record,method){
    const id=record.id+'-'+method,result=record[method],summary=result.summary;
    const article=document.createElement('article');article.className='robot-card';article.dataset.case=record.id;
    const regions=design.regions.map(r=>`<rect x="${r.x}" y="${r.y}" width="${r.width}" height="${r.height}" fill="#edf3e850" stroke="#477763" stroke-width="1.8"/><text x="${r.x+18}" y="${r.y+30}" class="region-label">Region ${r.label}</text>`).join('');
    const circles=design.obstacles.map(o=>`<circle cx="${o.center[0]}" cy="${o.center[1]}" r="${o.radius}" fill="#73828a"/><circle cx="${o.center[0]}" cy="${o.center[1]}" r="${o.radius+6}" fill="none" stroke="#73828a" stroke-dasharray="3 4"/>`).join('');
    const start=screen(record.reference.samples[0].slice(1,3)),goal=screen(record.reference.samples.at(-1).slice(1,3));
    const crossing=(summary.passage_crossings||[]).find(e=>e.direction===record.direction&&e.route===record.route),baseline=(record.still_air.summary.passage_crossings||[]).find(e=>e.direction===record.direction&&e.route===record.route);
    const crossingText=crossing&&baseline?`<br>${record.route[0].toUpperCase()+record.route.slice(1)} crossing: <b>${crossing.speed_m_s.toFixed(2)} m/s</b> at <b>${crossing.time_s.toFixed(2)} s</b> · still air: ${baseline.speed_m_s.toFixed(2)} m/s at ${baseline.time_s.toFixed(2)} s`:'';
    article.innerHTML=`<div class="robot-card-heading"><h4>Scenario ${record.scenario}</h4><span class="outcome ${tag(summary,record)}">${label(summary,record)}</span></div>
    <div class="robot-map"><svg viewBox="-5 -5 810 472.5" role="img" aria-label="Scenario ${record.scenario}, ${record.route} route, ${record.direction.replace('-',' to ')}, ${method==='rpcbf'?'RPCBF':'without safety filter'}">
    <defs><clipPath id="${id}-clip"><rect width="800" height="462.5"/></clipPath><marker id="${id}-air" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#b14076"/></marker><marker id="${id}-control" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#476fc0"/></marker></defs>
    <rect width="800" height="462.5" fill="#fafdfe"/><image class="robot-heat" width="800" height="462.5"/>${regions}${circles}
    <g clip-path="url(#${id}-clip)"><polyline class="reference-trace" points="${poly(record.reference.samples)}" fill="none" stroke="${colors.reference}" stroke-width="2" stroke-dasharray="6 5"/>
    <polyline class="still-trace" points="${poly(record.still_air.samples)}" fill="none" stroke="${colors.still}" stroke-width="2.4" stroke-dasharray="2 4"/>
    <polyline class="actual-trace" fill="none" stroke="${colors[method]}" stroke-width="3.4"/>
    <circle class="ref-dot" r="5" fill="white" stroke="${colors.reference}" stroke-width="2"/>
    <line class="air-arrow" stroke="#b14076" stroke-width="2.7" marker-end="url(#${id}-air)"/>
    <line class="control-arrow" stroke="#476fc0" stroke-width="2.7" marker-end="url(#${id}-control)"/>
    <circle class="robot-dot" r="6.5" fill="${colors[method]}" stroke="white" stroke-width="2.4"/>
    <path class="contact-marker" d="M-8 -8L8 8M-8 8L8 -8" stroke="#a93136" stroke-width="3" display="none"/>
    </g><circle cx="${start[0]}" cy="${start[1]}" r="10" fill="none" stroke="#66766d" stroke-width="1.2"/><circle cx="${goal[0]}" cy="${goal[1]}" r="11" fill="none" stroke="#66766d" stroke-width="1.2"/><text x="${goal[0]+16}" y="${goal[1]+5}" fill="#43594c" font-size="15">goal</text><rect width="800" height="462.5" fill="none" stroke="#85938d"/>${renderWindOpenings(record.scenario)}</svg></div>
    <div class="live-metrics"><div><span>Tracking error</span><b data-live="error">—</b></div><div><span>Robot speed</span><b data-live="speed">—</b></div><div><span>Wind acceleration</span><b data-live="air">—</b></div><div><span>Clearance</span><b data-live="clearance">—</b></div></div>
    <div class="robot-readout" role="status"></div>
    <div class="run-summary">Full run: minimum clearance <b>${summary.min_clearance_m.toFixed(3)} m</b> · final goal error <b>${summary.endpoint_error_m.toFixed(2)} m</b> · maximum tracking error <b>${summary.max_tracking_error_m.toFixed(2)} m</b> · actual passage: <b>${summary.passage_crossings?(summary.passage_crossings.map(e=>e.route).filter((r,i,a)=>a.indexOf(r)===i).join(", ")||"none crossed"):"not recorded"}</b>${method==='rpcbf'?` · <b>${summary.infeasible_steps}</b> infeasible QP steps`:''}${crossingText}</div>`;
    fields[record.scenario].paint(article.querySelector('.robot-heat'));
    return {article,record,method,result};
  }
  function drawPanel(panel){
    const {article,record,method,result}=panel,s=sample(result.samples,time),r=sample(record.reference.samples,time),p=screen(s.slice(1,3)),rp=screen(r.slice(1,3));
    const stopped=!!result.summary.contact&&time>=result.summary.contact.time_s;
    const set=(sel,attrs)=>{const e=article.querySelector(sel);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);};
    const past=result.samples.filter(row=>row[0]<time);past.push(s);
    set('.actual-trace',{points:poly(past)});set('.robot-dot',{cx:p[0],cy:p[1]});set('.ref-dot',{cx:rp[0],cy:rp[1]});
    for(const [sel,i] of [['.air-arrow',7],['.control-arrow',9]])set(sel,{x1:p[0],y1:p[1],x2:p[0]+20*s[i],y2:p[1]-20*s[i+1],display:stopped?'none':''});
    set('.contact-marker',{transform:`translate(${p[0]} ${p[1]})`,display:stopped?'':'none'});
    article.querySelector('.still-trace').style.display=$('robot-still').checked?'':'none';
    const vals={error:s[11].toFixed(2)+' m',speed:Math.hypot(s[3],s[4]).toFixed(2)+' m/s',air:Math.hypot(s[7],s[8]).toFixed(2)+' m/s²',clearance:s[12].toFixed(3)+' m'};
    for(const [k,v] of Object.entries(vals))article.querySelector(`[data-live=${k}]`).textContent=v;
    const speed=Math.hypot(r[3],r[4]),along=speed>1e-4?(s[7]*r[3]+s[8]*r[4])/speed:null;
    const windText=along===null?'Reference at rest':`Air force ${along<0?'opposes':'assists'} reference motion: ${Math.abs(along).toFixed(2)} m/s² along route`;
    article.querySelector('.robot-readout').textContent=stopped?`Contact at ${result.summary.contact.time_s.toFixed(2)} s · integration stopped`:method==='rpcbf'?`${windText} · QP ${s[14]?'feasible':'INFEASIBLE'} · correction ${s[13].toFixed(2)} m/s²`:`${windText} · control (${s[9].toFixed(2)}, ${s[10].toFixed(2)}) m/s²`;
    article.querySelector('.robot-readout').classList.toggle('run-warning',stopped||!s[14]);
  }
  function charts(){
    $('robot-charts').innerHTML=records.map(rec=>{
      const ymax=Math.max(.25,...['unfiltered','rpcbf'].map(k=>rec[k].summary.max_tracking_error_m))*1.1;
      const line=(method)=>rec[method].samples.map(s=>`${(42+540*s[0]/horizon).toFixed(2)},${(143-112*s[11]/ymax).toFixed(2)}`).join(' ');
      return `<div class="error-chart"><h4>Scenario ${rec.scenario} · tracking error</h4><svg viewBox="0 0 610 170" role="img" aria-label="Tracking error versus time, scenario ${rec.scenario}"><path d="M42 24V143H584" fill="none" stroke="#becac2"/><text x="38" y="22" text-anchor="end">${ymax.toFixed(1)} m</text><text x="34" y="147" text-anchor="end">0</text><text x="42" y="162">0 s</text><text x="582" y="162" text-anchor="end">${horizon.toFixed(1)} s</text>${['still_air','unfiltered','rpcbf'].map(k=>`<polyline points="${line(k)}" fill="none" stroke="${k==='still_air'?colors.still:colors[k]}" stroke-width="${k==='still_air'?1.3:2}"/>`).join('')}<line class="chart-time" y1="25" y2="143" stroke="#43594c" stroke-dasharray="3 3"/></svg></div>`;
    }).join('');
  }
  function draw(){
    $('robot-time').value=String(time);$('robot-clock').textContent=`${time.toFixed(1)} / ${horizon.toFixed(1)} s`;
    panels.forEach(drawPanel);root.querySelectorAll('.chart-time').forEach(e=>{e.setAttribute('x1',42+540*time/horizon);e.setAttribute('x2',42+540*time/horizon);});
  }
  async function loadSelection(){
    const generation=++version;setPlaying(false);status.textContent='Loading computed robot trajectories…';
    const route=$('robot-route').value,direction=$('robot-direction').value,scenarios=$('robot-scenario').value==='both'?[1,2]:[Number($('robot-scenario').value)];
    try{
      const loaded=await Promise.all(scenarios.map(async s=>{
        const id=`s${s}-${route}-${direction}`;
        if(!cache.has(id)){const response=await fetch(`data/robot/${id}.json`);if(!response.ok)throw Error(`Missing run ${id}`);cache.set(id,await response.json());}
        return cache.get(id);
      }));
      if(generation!==version)return;records=loaded;panels=[];time=0;horizon=records[0].reference.samples.at(-1)[0];$('robot-time').max=horizon;
      for(const method of ['unfiltered','rpcbf']){
        const container=$('robot-'+method);container.innerHTML='';container.classList.toggle('single-case',records.length===1);
        for(const record of records){const item=panel(record,method);panels.push(item);container.append(item.article);}
      }
      charts();draw();status.textContent=`${route[0].toUpperCase()+route.slice(1)} route · ${direction.replace('-',' → ')} · reference ${records[0].reference.checks.duration_s.toFixed(2)} s, then 10 s goal hold · same field, reference, tracker, and actuator limits in both rows.`;
      if(records.every(r=>r.directional_authority))$('robot-authority').textContent=records.map(r=>{const a=r.directional_authority;return `Scenario ${r.scenario}: exact reference tracking requires up to ${a.max_required_control_component_m_s2.toFixed(2)} m/s² per axis (available: 1.60); opposing air acceleration peaks at ${a.max_opposing_air_acceleration_m_s2.toFixed(2)} m/s².`;}).join(' ')+ ' This checks the prescribed route, not every possible detour.';
      root.querySelectorAll('[data-case-select]').forEach(b=>{const selected=records.some(r=>r.id===b.dataset.caseSelect);b.setAttribute('aria-pressed',String(selected));b.closest('tr').classList.toggle('selected-case',selected);});
      $('robot-play').disabled=false;$('robot-restart').disabled=false;setPlaying(true);
    }catch(e){status.textContent=`Robot review could not load: ${e.message}`;status.classList.add('run-warning');}
  }
  try{
    const response=await fetch('data/robot/manifest.json');if(!response.ok)throw Error('Simulation results are not available yet');manifest=await response.json();
    for(const s of [1,2]){const meta=await (await fetch(`data/scenario${s}/metadata.json`)).json();fields[s]=new StationaryFlow(meta,await(await fetch(`data/scenario${s}/${meta.data_file}`)).arrayBuffer());}
    $('robot-cases').innerHTML=manifest.cases.map(c=>`<tr><td><button data-case-select="${c.id}" aria-pressed="false">Scenario ${c.scenario} · ${c.route} · ${c.direction.replace('-',' → ')}</button></td><td><span class="outcome ${tag(c.unfiltered,c)}">${label(c.unfiltered,c)}</span></td><td><span class="outcome ${tag(c.rpcbf,c)}">${label(c.rpcbf,c)}</span></td><td>${c.rpcbf.infeasible_steps}</td></tr>`).join('');
    root.querySelectorAll('[data-case-select]').forEach(button=>button.addEventListener('click',()=>{const c=manifest.cases.find(c=>c.id===button.dataset.caseSelect);$('robot-scenario').value=String(c.scenario);$('robot-route').value=c.route;$('robot-direction').value=c.direction;loadSelection();root.scrollIntoView({behavior:'smooth',block:'start'});}));
    for(const id of ['robot-scenario','robot-route','robot-direction'])$(id).addEventListener('change',loadSelection);
    $('robot-play').addEventListener('click',()=>{if(time>=horizon){time=0;draw();}setPlaying(!playing);});
    $('robot-restart').addEventListener('click',()=>{time=0;draw();setPlaying(true);});
    $('robot-time').addEventListener('input',e=>{time=Number(e.target.value);setPlaying(false);draw();});
    $('robot-still').addEventListener('change',draw);
    const maxDifference=Math.max(...manifest.timestep_refinement.map(c=>c.max_common_time_position_difference_m));
    const upperDifference=Math.max(...manifest.timestep_refinement.filter(c=>c.scenario===2&&c.route==='upper').map(c=>c.max_common_time_position_difference_m));
    document.querySelector('.robot-numerical-note').textContent=`Numerical check: halving the integration step changes the Scenario 2 upper-bypass paths by at most ${(upperDifference*1000).toFixed(3)} mm. Maximum change across the checked runs: ${(maxDifference*100).toFixed(1)} cm; outcomes ${manifest.timestep_refinement.every(c=>c.same_outcome)?'agree':'differ'}. CFD grid convergence remains unestablished.`;
    $('robot-verification').textContent=`All six directed reference trajectories pass the 0.06 m clearance and acceleration checks. Halving the integration step from 0.01 to 0.005 s in all 12 directed cases, with and without RPCBF, gives a maximum common-time position difference of ${(1000*maxDifference).toFixed(2)} mm. Outcomes ${manifest.timestep_refinement.every(c=>c.same_outcome)?'agree':'differ; inspect the downloadable checks'}.`;
    await loadSelection();
    function frame(stamp){const dt=previous===null?0:Math.min(.1,(stamp-previous)/1000);previous=stamp;if(playing&&!document.hidden){time=Math.min(horizon,time+dt*Number($('robot-speed').value));draw();if(time>=horizon)setPlaying(false);}requestAnimationFrame(frame);}requestAnimationFrame(frame);
  }catch(e){status.textContent=`Robot review could not load: ${e.message}`;status.classList.add('run-warning');}
})();
