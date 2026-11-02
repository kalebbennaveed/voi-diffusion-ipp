'use strict';
// Recorded MBD missions only. Playback never integrates robot or clarity models.
async function monitoringComparison(root){
  const prefix=root.dataset.controlPrefix||'monitor-';
  const $=id=>root.querySelector('#'+id.replace(/^monitor-/,prefix)),colors={aware:'#147861',baseline:'#c46a35'};
  let env,manifest,records=[],panels=[],time=0,duration=60,playing=false,previous=null,version=0;
  const dataDirectory=root.dataset.monitoringDirectory||'data/monitoring';
  const cache=new Map();
  // The same fixed scales apply to both methods and both scenarios.
  const scales={
    clarity:{title:'Clarity q',unit:'dimensionless',max:1,stops:['#fff7ec','#fee8c8','#fdbb84','#fc8d59','#e34a33','#b30000','#7f0000'],low:'Low clarity',high:'High clarity',note:'Local target: 0.70 in A/B · 0.35 elsewhere'},
    deficit:{title:'Deficit below local target',unit:'dimensionless',max:.7,stops:['#fff7ec','#fee8c8','#fdbb84','#fc8d59','#e34a33','#b30000','#7f0000'],low:'Target met',high:'Large deficit',note:'max(target − q, 0) · targets are 0.70 in A/B and 0.35 elsewhere'},
    uncertainty:{title:'Wind-belief standard deviation σ',unit:'m/s',max:3,stops:['#f7fbff','#d6e6f4','#94c4df','#4a98c9','#1764ab','#08306b'],low:'Less uncertain',high:'More uncertain',note:'Posterior standard deviation per wind component; ≥3 m/s uses the darkest color'},
    wind:{title:'Posterior mean wind speed |μw|',unit:'m/s',max:8,stops:['#f7fbff','#c6dbef','#6baed6','#2171b5','#08306b'],low:'0 m/s',high:'8 m/s',note:'Estimated air velocity, not the true field. Arrows show direction; color shows speed. Values ≥8 m/s use the darkest color.'},
    decay:{title:'Clarity process noise Q',unit:'s⁻¹',max:.16,stops:['#fcfbfd','#dadaeb','#bcbddc','#9e9ac8','#756bb1','#54278f'],low:'Slower decay',high:'Faster decay',note:'Q = 0.02 s⁻¹ in the background · 0.16 s⁻¹ in A/B'}
  };
  function colorAt(scale,value){
    const u=Math.max(0,Math.min(1,value/scale.max))*(scale.stops.length-1),i=Math.min(scale.stops.length-2,Math.floor(u)),f=u-i;
    const rgb=h=>[1,3,5].map(k=>parseInt(h.slice(k,k+2),16));
    const a=rgb(scale.stops[i]),b=rgb(scale.stops[i+1]);return a.map((v,k)=>Math.round(v+f*(b[k]-v)));
  }
  function scaleMarkup(layer){
    const z=scales[layer],ticks=Array.from({length:5},(_,i)=>{const v=z.max*i/4;return `<span>${layer==='decay'?v.toFixed(2):v.toFixed(layer==='clarity'||layer==='uncertainty'?2:3)}</span>`;}).join('');
    return `<div class="monitor-scale-heading"><strong>${z.title}</strong><span>${z.unit} · fixed scale</span></div><div class="monitor-colorbar" style="background:linear-gradient(to right,${z.stops.join(',')})"></div><div class="monitor-scale-ticks">${ticks}</div><div class="monitor-scale-ends"><span>${z.low}</span><span>${z.high}</span></div><p>${z.note}</p>`;
  }
  const screen=p=>[100*p[0],462.5-100*p[1]];
  const line=rows=>rows.map(r=>screen(r).map(v=>v.toFixed(2)).join(',')).join(' ');
  function nearest(rows,t){let lo=0,hi=rows.length-1;while(hi-lo>1){const n=(lo+hi)>>1;if(rows[n].t<=t)lo=n;else hi=n;}return rows[hi]?.t<=t?hi:lo;}
  function sample(rows,t){
    if(t<=rows[0][0])return rows[0];if(t>=rows.at(-1)[0])return rows.at(-1);
    let lo=0,hi=rows.length-1;while(hi-lo>1){const n=(lo+hi)>>1;if(rows[n][0]<=t)lo=n;else hi=n;}
    const a=rows[lo],b=rows[hi],f=(t-a[0])/(b[0]-a[0]);
    return a.map((v,i)=>[11,18].includes(i)?v:v+f*(b[i]-v));
  }
  function setPlaying(value){playing=value;$('monitor-play').textContent=value?'Ⅱ Pause monitoring':'▶ Play monitoring';$('monitor-play').setAttribute('aria-pressed',String(value));}
  function makePanel(record){
    const method=record.method,article=document.createElement('article');article.className='monitor-card';
    const clipId=root.id+'-'+record.id+'-monitor-clip';
    const visited=env.regions.filter(region=>record.samples.some(s=>s[1]>=region.bounds[0][0]&&s[1]<=region.bounds[1][0]&&s[2]>=region.bounds[0][1]&&s[2]<=region.bounds[1][1])).map(region=>region.label).join(', ')||'none';
    const summary=record.summary,name=method==='aware'?'Proposed · disturbance-aware MBD':'Baseline · MBD + execution RPCBF';
    const regions=design.regions.map(r=>`<rect x="${r.x}" y="${r.y}" width="${r.width}" height="${r.height}" fill="none" stroke="${r.label==='A'?'#2563eb':'#b45309'}" stroke-width="2"/><text x="${r.x+18}" y="${r.y+28}" class="region-label monitor-region-label">Region ${r.label}</text>`).join('');
    const obstacles=design.obstacles.map(o=>`<circle cx="${o.center[0]}" cy="${o.center[1]}" r="${o.radius}" fill="#73828a"/>`).join('');
    article.innerHTML=`<div class="robot-card-heading"><h4>${name}</h4><span class="outcome ${summary.contact?'collision':summary.infeasible_execution_steps||summary.min_clearance_m<(manifest.planner_settings?.safety_margin??.06)?'stopped':''}">${summary.contact?'Full run: contact':`${summary.simulated_seconds.toFixed(0)} s · no contact${summary.infeasible_execution_steps?' · QP infeasibility':''}`}</span></div>
    <p class="monitor-method">${method==='aware'?'Current wind belief + relaxed RPCBF inside planning.':'Still-air planning. Wind-aware RPCBF after optimization.'}</p>
    <div class="robot-map"><svg viewBox="-5 -5 810 472.5" role="img" aria-label="${name}, scenario ${record.scenario}, executed monitoring and clarity">
    <defs><clipPath id="${clipId}"><rect width="800" height="462.5"/></clipPath></defs><rect width="800" height="462.5" fill="#fafcf9"/><image class="monitor-heat" width="800" height="462.5"/>
    ${regions}${obstacles}<g clip-path="url(#${clipId})"><g class="monitor-belief-arrows"></g><g class="monitor-seed-paths"></g><g class="monitor-forecasts"></g><polyline class="monitor-trace-halo" fill="none" stroke="white" stroke-width="6" stroke-opacity=".9"/><polyline class="monitor-trace" fill="none" stroke="${colors[method]}" stroke-width="3.2"/><circle class="monitor-dot" r="6.5" fill="${colors[method]}" stroke="white" stroke-width="2.2"/><path class="monitor-contact" d="M-8 -8L8 8M-8 8L8 -8" stroke="#a93136" stroke-width="3" display="none"/></g><rect width="800" height="462.5" fill="none" stroke="#85938d"/>${renderWindOpenings(record.scenario)}</svg></div>
    <div class="monitor-color-scale" aria-label="Map color scale"></div><p class="monitor-probe">Move over the map to inspect a cell.</p><div class="live-metrics"><div><span>Clarity · A</span><b data-metric="a">—</b></div><div><span>Clarity · B</span><b data-metric="b">—</b></div><div><span>Whole-space deficit</span><b data-metric="deficit">—</b></div><div><span>Accumulated cost</span><b data-metric="cost">—</b></div></div>
    <p class="monitor-readout"></p><p class="monitor-plan-readout"></p><details class="monitor-decision"><summary>Inspect this planning decision</summary><div class="monitor-decision-content"></div></details>
    <div class="monitor-wind-belief"><h5>Learned wind · speed and direction</h5><p class="monitor-wind-readout"></p><div class="robot-map"><svg viewBox="-5 -5 810 472.5" role="img" aria-label="${name}, scenario ${record.scenario}, posterior mean wind velocity"><rect width="800" height="462.5" fill="#f7fbff"/><image class="monitor-wind-heat" width="800" height="462.5"/>${regions}${obstacles}<g class="monitor-wind-vectors" clip-path="url(#${clipId})"></g><circle class="monitor-wind-dot" r="6.5" fill="${colors[method]}" stroke="white" stroke-width="2.2"/><rect width="800" height="462.5" fill="none" stroke="#85938d"/></svg></div><div class="monitor-color-scale">${scaleMarkup('wind')}</div><p class="monitor-probe monitor-wind-probe">Move over the wind map to inspect its estimated velocity and uncertainty.</p></div><div class="monitor-region-chart"></div>
    <div class="run-summary">${summary.completed?'Full mission':`Recorded ${summary.simulated_seconds.toFixed(0)} s`}: cost <b>${summary.integrated_deficit.toFixed(3)}</b> · clearance <b>${(Math.abs(summary.min_clearance_m)<1e-12?0:summary.min_clearance_m).toFixed(3)} m</b> · infeasible execution steps <b>${summary.infeasible_execution_steps}</b> · fallback periods <b>${summary.fallback_periods}</b> · regions visited <b>${visited}</b>${summary.min_clearance_m<(manifest.planner_settings?.safety_margin??.06)?'<br><strong class="run-warning">Physical safety margin breached.</strong>':''}${summary.contact?`<br>Contact at ${summary.contact.time_s.toFixed(2)} s; subsequent clarity decays without sensing.`:''}</div>`;
    const panel={article,record,lastHeat:'',lastBelief:-1,lastPlan:-1};
    article.querySelector('.robot-map svg').addEventListener('pointermove',event=>{
      const svg=event.currentTarget,p=new DOMPoint(event.clientX,event.clientY).matrixTransform(svg.getScreenCTM().inverse());
      const x=Math.floor(p.x/12.5),y=36-Math.floor(p.y/12.5),cell=y*64+x,n=env.free_indices.indexOf(cell);
      const readout=article.querySelector('.monitor-probe');
      if(x<0||x>=64||y<0||y>=37||n<0){readout.textContent='Obstacle or outside the environment';return;}
      const q=record.clarity_frames[nearest(record.clarity_frames,time)].q[n],bf=record.belief_frames[nearest(record.belief_frames,time)];
      readout.textContent=`(${((x+.5)*.125).toFixed(2)}, ${((y+.5)*.125).toFixed(2)}) m · q ${q.toFixed(3)} · target ${env.targets[n].toFixed(2)} · deficit ${Math.max(0,env.targets[n]-q).toFixed(3)} · Q ${env.process_noise[n].toFixed(2)} s⁻¹ · σ ${bf.std[cell].toFixed(2)} m/s`;
    });
    article.querySelector('.monitor-wind-belief svg').addEventListener('pointermove',event=>{
      const svg=event.currentTarget,p=new DOMPoint(event.clientX,event.clientY).matrixTransform(svg.getScreenCTM().inverse());
      const x=Math.floor(p.x/12.5),y=36-Math.floor(p.y/12.5),cell=y*64+x;
      const readout=article.querySelector('.monitor-wind-probe');
      if(x<0||x>=64||y<0||y>=37||!env.free_indices.includes(cell)){readout.textContent='Obstacle or outside the environment';return;}
      const bf=record.belief_frames[nearest(record.belief_frames,time)],v=bf.mean[cell];
      readout.textContent=`Estimated velocity (${v[0].toFixed(2)}, ${v[1].toFixed(2)}) m/s · speed ${Math.hypot(...v).toFixed(2)} m/s · per-component uncertainty σ ${bf.std[cell].toFixed(2)} m/s · x right, y up`;
    });
    return panel;
  }
  function paint(panel){
    const record=panel.record,layer=$('monitor-layer').value;
    const qi=nearest(record.clarity_frames,time),bi=nearest(record.belief_frames,time),key=layer+':'+(layer==='uncertainty'?bi:layer==='decay'?0:qi);
    if(panel.lastHeat===key)return;panel.lastHeat=key;
    const canvas=document.createElement('canvas');canvas.width=64;canvas.height=37;
    const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(64,37),q=record.clarity_frames[qi].q;
    for(let n=0;n<env.free_indices.length;n++){
      const cell=env.free_indices[n],x=cell%64,y=Math.floor(cell/64),index=((36-y)*64+x)*4;
      let value;
      if(layer==='clarity')value=q[n];
      else if(layer==='deficit')value=Math.max(0,env.targets[n]-q[n]);
      else if(layer==='decay')value=env.process_noise[n];
      else value=record.belief_frames[bi].std[cell];
      const rgb=colorAt(scales[layer],value);for(let c=0;c<3;c++)pixels.data[index+c]=rgb[c];pixels.data[index+3]=255;
    }
    ctx.putImageData(pixels,0,0);panel.article.querySelector('.monitor-heat').setAttribute('href',canvas.toDataURL());
  }
  function drawPanel(panel){
    const {article,record}=panel,s=sample(record.samples,time),p=screen(s.slice(1,3));
    paint(panel);const points=line([...record.samples.filter(r=>r[0]<time).map(r=>r.slice(1,3)),s.slice(1,3)]);for(const selector of ['.monitor-trace','.monitor-trace-halo'])article.querySelector(selector).setAttribute('points',points);
    const dot=article.querySelector('.monitor-dot');dot.setAttribute('cx',p[0]);dot.setAttribute('cy',p[1]);
    const windDot=article.querySelector('.monitor-wind-dot');windDot.setAttribute('cx',p[0]);windDot.setAttribute('cy',p[1]);
    const contact=article.querySelector('.monitor-contact');contact.setAttribute('transform',`translate(${p[0]} ${p[1]})`);contact.setAttribute('display',s[18]?'none':'');
    const metrics={a:s[15].toFixed(3),b:s[16].toFixed(3),deficit:s[13].toFixed(3),cost:s[14].toFixed(3)};
    for(const [key,value] of Object.entries(metrics))article.querySelector(`[data-metric="${key}"]`).textContent=value;
    const bi=nearest(record.belief_frames,time),bf=record.belief_frames[bi];
    if(panel.lastBelief!==bi){
      panel.lastBelief=bi;let arrows='';
      for(let j=2;j<37;j+=4)for(let i=2;i<64;i+=4){
        const n=j*64+i,v=bf.mean[n],x=(i+.5)*12.5,y=462.5-(j+.5)*12.5;
        if(design.obstacles.some(o=>Math.hypot(x-o.center[0],y-o.center[1])<=o.radius))continue;
        const len=Math.hypot(v[0],v[1]);if(len<.08)continue;
        const scale=Math.min(10,25/len),xx=x+scale*v[0],yy=y-scale*v[1],angle=Math.atan2(-v[1],v[0]);
        arrows+=`<path d="M${x} ${y}L${xx} ${yy}M${xx-4*Math.cos(angle-.5)} ${yy-4*Math.sin(angle-.5)}L${xx} ${yy}L${xx-4*Math.cos(angle+.5)} ${yy-4*Math.sin(angle+.5)}" fill="none" stroke="#335d78" stroke-width="1.3"/>`;
      }
      article.querySelector('.monitor-belief-arrows').innerHTML=arrows;
      article.querySelector('.monitor-wind-vectors').innerHTML=arrows;
      const canvas=document.createElement('canvas');canvas.width=64;canvas.height=37;
      const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(64,37);
      for(const cell of env.free_indices){const x=cell%64,y=Math.floor(cell/64),index=((36-y)*64+x)*4,rgb=colorAt(scales.wind,Math.hypot(...bf.mean[cell]));for(let c=0;c<3;c++)pixels.data[index+c]=rgb[c];pixels.data[index+3]=255;}
      ctx.putImageData(pixels,0,0);article.querySelector('.monitor-wind-heat').setAttribute('href',canvas.toDataURL());
      article.querySelector('.monitor-wind-readout').textContent=`Posterior at ${bf.t.toFixed(1)} s · ${bf.observations.toLocaleString()} wind measurements · fixed 0–8 m/s scale across every run`;
    }
    article.querySelector('.monitor-belief-arrows').style.display=$('monitor-belief').checked?'':'none';
    let pi=record.plans.length?nearest(record.plans,time):-1;
    if(pi!==panel.lastPlan){
      panel.lastPlan=pi;const plan=pi<0?null:record.plans[pi];article.querySelector('.monitor-seed-paths').innerHTML=(plan?.initialization?.coarse_paths||[]).map((path,i)=>`<polyline points="${line(path)}" fill="none" stroke="#86198f" stroke-width="2" stroke-dasharray="4 6"><title>Online coarse path ${i+1}</title></polyline>`).join('');article.querySelector('.monitor-plan-readout').textContent=plan?.initialization?`Plan at ${plan.t.toFixed(1)} s · ${plan.seed_costs.length} MBD starts · ${plan.evaluations} full-objective evaluations · ${(Array.isArray(plan.predicted_contacts)?plan.predicted_contacts:[plan.predicted_contacts]).filter(Boolean).length}/${plan.predicted_states.length} predicted ${record.method==='aware'?'wind worlds':'still-air forecast'} contact · initialization ${plan.initialization.total_seed_milliseconds.toFixed(1)} ms`:'';const decision=article.querySelector('.monitor-decision-content');
      if(plan){
        const valid=plan.seed_costs.filter(v=>v!==null&&Number.isFinite(v)),best=valid.length?Math.min(...valid):Infinity;
        const improved=Number.isFinite(plan.expected_cost)&&plan.expected_cost<best-1e-8;
        decision.innerHTML=`<p>${improved?'Diffusion improved on every initial candidate.':'The best initial candidate was retained at this finite diffusion budget.'} Selected expected planning cost: <strong>${plan.expected_cost===null?'invalid':plan.expected_cost.toFixed(3)}</strong>.</p><table><thead><tr><th>Initialization</th><th>Full planning cost</th></tr></thead><tbody>${plan.seed_costs.map((value,i)=>`<tr><td>${plan.initialization?(i===0?'Shifted continuation':i===1?'Hold reference':`Coarse path ${i-1}`):`Start ${i+1}`}</td><td>${value===null?'Rejected':value.toFixed(3)}</td></tr>`).join('')}</tbody></table><p>These are complete physical-rollout costs. The coarse graph’s approximate sensing score is used only to construct the starts.</p>`;
        if(manifest.implementation==='legacy-mbd'){
          const origin=plan.selected_origin==='diffusion'?'A diffusion proposal was selected.':'An initialized trajectory was selected.';
          decision.innerHTML=`<p>${origin} Expected cost on separate validation wind samples: <strong>${plan.expected_cost.toFixed(3)}</strong>.</p><p>Planning time: <strong>${plan.wall_seconds.toFixed(3)} s</strong>${plan.timing?.compilation_included?' (includes first-call compilation)':''}. ${plan.evaluations} search evaluations; ${plan.validation_evaluations} finalists checked on a fresh common wind bank.</p>${plan.timing?`<p>Module times: geometric initialization ${(1000*plan.timing.initialization).toFixed(1)} ms · field sampling ${(1000*plan.timing.field_sampling).toFixed(1)} ms · MBD + final validation ${(1000*plan.timing.optimization_validation).toFixed(1)} ms. The compiled optimizer measures MBD and validation together.</p>`:''}<table><thead><tr><th>Initialization</th><th>Search-bank mean cost</th></tr></thead><tbody>${plan.seed_costs.map((v,i)=>`<tr><td>${i===0?'Hold':i===plan.seed_costs.length-1?'Shifted continuation':`Geometric path ${i}`}</td><td>${v===null?'Rejected':v.toFixed(3)}</td></tr>`).join('')}</tbody></table><p>Search and validation use independent wind draws. Their costs are not directly comparable as evidence of improvement. Every control coordinate remains free.</p>`;
        }
      }else decision.textContent='No plan at this time.';
      article.querySelector('.monitor-forecasts').innerHTML=pi<0?'':record.plans[pi].predicted_states.map(rows=>`<polyline points="${line(rows.map(r=>r.slice(0,2)))}" fill="none" stroke="#596d76" stroke-width="1.3" stroke-dasharray="5 5" opacity=".7"/>`).join('');
    }
    article.querySelector('.monitor-seed-paths').style.display=$('monitor-seeds').checked?'':'none';
    article.querySelector('.monitor-forecasts').style.display=$('monitor-plans').checked?'':'none';
    article.querySelector('.monitor-readout').textContent=`Speed ${Math.hypot(s[3],s[4]).toFixed(2)} m/s · clarity target A/B 0.70 · ${bf.observations} wind measurements · belief at ${bf.t.toFixed(1)} s${s[18]?'': ' · stopped after contact'}`;
    article.querySelector('.monitor-chart-cursor')?.setAttribute('x1',48+516*time/duration);article.querySelector('.monitor-chart-cursor')?.setAttribute('x2',48+516*time/duration);
  }
  function chart(panel){
    const r=panel.record,path=column=>r.samples.filter(s=>s[0]<=duration).map(s=>`${48+516*s[0]/duration},${160-130*s[column]}`).join(' ');
    const grid=[0,.25,.5,.75,1].map(v=>`<line x1="48" x2="564" y1="${160-130*v}" y2="${160-130*v}" stroke="#e2e8f0"/><text x="38" y="${164-130*v}" text-anchor="end">${v.toFixed(2)}</text>`).join('');
    const ticks=[0,.25,.5,.75,1].map(f=>`<text x="${48+516*f}" y="180" text-anchor="middle">${(duration*f).toFixed(0)}</text>`).join('');
    panel.article.querySelector('.monitor-region-chart').innerHTML=`<div class="monitor-chart-heading"><strong>Mean clarity over time</strong><span>q · 0–1</span></div><div class="monitor-chart-legend"><span><i style="--line:#2563eb"></i>Region A</span><span><i class="dashed" style="--line:#b45309"></i>Region B</span><span><i class="dotted" style="--line:#64748b"></i>Whole space</span><span><i class="dashed" style="--line:#334155"></i>A/B target 0.70</span></div><svg viewBox="0 0 600 204" role="img" aria-label="Mean clarity from zero to one, versus time in seconds, for Region A, Region B and the whole free environment">${grid}${ticks}<path d="M48 ${160-130*.7}H564" stroke="#334155" stroke-width="1" stroke-dasharray="6 5"/><polyline points="${path(17)}" fill="none" stroke="#64748b" stroke-width="1.8" stroke-dasharray="2 4"/><polyline points="${path(15)}" fill="none" stroke="#2563eb" stroke-width="2.6"/><polyline points="${path(16)}" fill="none" stroke="#b45309" stroke-width="2.6" stroke-dasharray="8 3"/><text x="306" y="199" text-anchor="middle">Simulation time (s)</text><line class="monitor-chart-cursor" y1="26" y2="163" stroke="#0f172a" stroke-width="1.2"/></svg>`;
  }
  function draw(){
    $('monitor-time').value=time;$('monitor-clock').textContent=`${time.toFixed(1)} / ${duration.toFixed(1)} s`;panels.forEach(drawPanel);
    const layer=$('monitor-layer').value;$('monitor-scale').textContent='Identical color limits for both methods and scenarios';
    for(const panel of panels)if(panel.scaleLayer!==layer){panel.article.querySelector('.monitor-color-scale').innerHTML=scaleMarkup(layer);panel.scaleLayer=layer;}
  }
  async function load(preserve=false){
    const savedTime=time,savedPlaying=playing,generation=++version;if(!preserve)setPlaying(false);$('monitor-status').textContent='Loading computed MBD missions…';
    try{
      const scenario=Number($('monitor-scenario').value);
      const ids=['aware','baseline'].map(method=>`s${scenario}-${method}`);
      const loaded=await Promise.all(ids.map(async id=>{if(manifest.state==='computing'||!cache.get(id)?.summary.completed){const response=await fetch(`${dataDirectory}/${id}.json`,{cache:'no-store'});if(!response.ok)throw Error('This mission has not finished its first planning period yet. Computation continues.');cache.set(id,await response.json());}return cache.get(id);}));
      if(generation!==version)return;records=loaded;$('monitor-seeds-label').hidden=!records.some(r=>r.plans.some(p=>p.initialization));duration=Math.min(...records.map(r=>r.summary.simulated_seconds));time=preserve?Math.min(savedTime,duration):0;
      $('monitor-time').max=duration;panels=records.map(makePanel);$('monitor-panels').replaceChildren(...panels.map(p=>p.article));panels.forEach(chart);draw();
      $('monitor-play').disabled=false;$('monitor-restart').disabled=false;
      $('monitor-status').textContent=`Scenario ${scenario} · ${records.every(r=>r.summary.completed&&r.summary.simulated_seconds===manifest.duration_s)?`${duration.toFixed(0)} seconds completed`:`showing ${duration.toFixed(0)} of ${Math.max(manifest.duration_s,...records.map(r=>r.summary.planned_seconds))} seconds; computation continues`} · same initial state, clarity, wind field, noise stream and MBD budget · each method learns from its own executed measurements.`;
      if(preserve)setPlaying(savedPlaying);
      $('monitor-results').textContent=records.every(r=>r.summary.completed&&r.summary.simulated_seconds===manifest.duration_s)?records.map(r=>`${r.method==='aware'?'Proposed':'Baseline'}: integrated whole-space deficit ${r.summary.integrated_deficit.toFixed(3)}; ${r.summary.contact?'contact':'no contact'}; ${r.summary.passage_crossings.length} passage crossings.`).join(' '):`Common computed prefix: ${duration.toFixed(0)} s. Accumulated cost at that same time: ${records.map(r=>`${r.method==='aware'?'proposed':'baseline'} ${sample(r.samples,duration)[14].toFixed(3)}`).join(' · ')}. Full mission comparison is pending.`;
      if(records.every(r=>r.summary.completed&&r.summary.simulated_seconds===manifest.duration_s)){
        const proposed=records.find(r=>r.method==='aware');
        const bypass=proposed?.summary.passage_crossings.some(c=>c.route==='upper'||c.route==='lower');
        if(!bypass)$('monitor-results').textContent+=' The proposed run has not demonstrated a bypass; passing the implementation audit does not establish the requested detour behavior.';
      }
    }catch(e){$('monitor-status').textContent=e.message;}
  }
  try{
    const response=await fetch(`${dataDirectory}/manifest.json`,{cache:'no-store'});if(!response.ok)throw Error('Monitoring missions are being computed. Existing simulations above remain available.');manifest=await response.json();root.querySelector('.review-badge').textContent=`${manifest.duration_s} s · computed clarity dynamics`;
    const summaryTable=$('monitor-summary-table');
    if(summaryTable&&manifest.runs?.length)summaryTable.innerHTML=`<table class="monitor-all-results"><caption>All ${manifest.duration_s} s runs · ${manifest.initial_wind_knowledge==='survey'?'with initial survey':'without initial survey'}</caption><thead><tr><th>Scenario / method</th><th>Contact</th><th>Minimum clearance</th><th>Infeasible steps</th><th>Deficit cost ↓</th><th>Crossings</th></tr></thead><tbody>${manifest.runs.map(r=>`<tr><th>Scenario ${r.scenario} · ${r.method==='aware'?'Proposed':'Baseline'}</th><td>${r.summary.contact?`At ${r.summary.contact.time_s.toFixed(2)} s`:'None'}</td><td>${r.summary.min_clearance_m.toFixed(3)} m</td><td>${r.summary.infeasible_execution_steps}</td><td>${r.summary.integrated_deficit.toFixed(3)}</td><td>${r.summary.passage_crossings.length}</td></tr>`).join('')}</tbody></table>`;
    const sourceLink=root.querySelector('a[download]');if(manifest.state==='computing'||manifest.source_available===false){sourceLink.removeAttribute('href');sourceLink.textContent=manifest.state==='computing'?'Source package follows the completed mission audit':'Archived pilot: records and audit in the manifest';}
    env=await(await fetch(`${dataDirectory}/environment.json`,{cache:'no-store'})).json();
    if(manifest.planner_settings?.multimode){
      const c=manifest.planner_settings;
      $('monitor-learning-details').textContent=`Both methods start at B with zero velocity, the same clarity field, stationary wind and measurement-noise stream. They begin with an untrained POAM prior and one actual local measurement. Full POAM uses actual 5 Hz observations, inducing-point relocation and ten training steps every ${c.commit_seconds} s.`;
      $('monitor-objective-details').textContent=`The planning objective integrates whole-free-space mean clarity deficit${c.legacy_safety_costs?` plus the original clearance and speed penalty terms: max(${c.planning_clearance.toFixed(2)} − clearance, 0)² + 2 max(speed − 3.1, 0)² while alive`:''}. The displayed accumulated monitoring cost excludes those planning penalties. Contact stops motion and sensing; clarity continues decaying for the full mission.`;
      $('monitor-budget').textContent=`Original model-based diffusion optimizes ${Math.round(2*c.horizon_seconds/c.dt)} reference-acceleration coordinates over ${c.horizon_seconds} s, replanning every ${c.commit_seconds} s. Eight independent reverse chains preserve the coarse alternatives and continuation. ${c.diffusion_steps} diffusion levels use ${c.proposals} total proposals per level (${c.proposals/8} per chain). Every seed remains a candidate; selection uses the full expected cost. All acceleration coordinates remain free.`;
      $('monitor-initialization-details').innerHTML=`<h3>Online coarse initialization</h3><p>A fresh 64 × 37 collision-free grid is constructed from the known obstacles at each planning call. No-sensing clarity forecasts and the sensor footprint rank free-space destinations. Compiled Dijkstra searches and ${c.legacy_safety_costs?'temporary cuts at geometrically discovered bottlenecks':'temporary passage penalties'} produce six coarse alternatives, plus a hold and shifted continuation. Destinations follow the current clarity field; no named-region destination or stored route graph is used. Rounded corners and acceleration-limited timing provide the initial references.</p><p>This approximate ranking initializes paths only. Every MBD candidate is evaluated with the full current POAM wind belief, physical air-force law, relaxed RPCBF and clarity ODE. No future belief updates are assumed.</p><h3>Reference tracking</h3><p class="equation">ṗ = v · v̇ = u + a_air(p,v)<br>u_nom = a_ref + ${c.tracking_kp.toFixed(3)}(p_ref − p)<br> + ${c.tracking_kv.toFixed(3)}(v_ref − v) − â_air(p,v)</p><p>The reference follows the nominal double integrator. The physical robot retains its original wind-force model and actuator limits. The old monitoring tracker is restored before the filter; both methods use the same tracking gains. The baseline forecast uses still air.</p><h3>MBD implementation checks</h3><p>The proposal covariance, cost-weighted score and reverse update follow <a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/6a5c38a730b17f3827c09cf6b192be04-Paper-Conference.pdf">Pan et al., equations 6, 8 and 9</a>. Tests compare the score with analytic Gaussian convolution and independent quadrature. Saved histories retain all proposal coordinates, costs, weights and reverse states for every chain.</p><p>Predicted contact creates an absorbing failure: motion and sensing stop while clarity continues decaying for the entire remaining horizon. This restores the old study’s failure accounting. Executed contacts are reported and receive no further sensing credit.</p>`;
    }

    if(manifest.implementation==='legacy-mbd'){
      const c=manifest.planner_settings,l=manifest.legacy_settings,r=manifest.runtime,e=manifest.execution_belief;
      const update=e?.posterior_update_seconds??l.commit;
      const learning=`The posterior conditions on actual 5 Hz local observations every ${update} s, with inducing-point relocation and fixed kernel parameters. The viewer records posterior maps every 5 s; speed colors and arrows show the learned mean, while the uncertainty layer shows posterior standard deviation.`;
      $('monitor-learning-details').textContent=(manifest.initial_wind_knowledge==='survey'?`Both methods receive the same initial 1,884 noisy spatial wind measurements (noise standard deviation 0.1 m/s), followed by local observations during execution. Full POAM uses 128 inducing points, posterior covariance and coherent field samples. `:`Both methods start with a zero-mean POAM prior and one actual noisy local observation at B. ${manifest.prior_configuration?.startsWith('survey-matched')?'The kernel and 128 inducing locations match the surveyed experiment; geometry alone initializes them, without accessing survey measurements or wind values. ':''}`)+learning;
      $('monitor-budget').textContent=`The original quarter MBD optimizer uses its cosine schedule, seven geometric reverse chains, batched candidate evaluation and fresh-field validation. It optimizes ${2*l.horizon/l.dt} acceleration coordinates over ${l.horizon} s and replans every ${l.commit} s. ${l.iterations} levels × ${l.proposals} total proposals; ${l.worlds} search wind draws and ${l.validation_worlds} independent validation draws. No optimism, archive-mixture proposal or demonstration likelihood is active.`;
      const filter=$('monitor-filter-details');if(filter)filter.textContent=`The proposed planner applies the original relaxed RPCBF inside each wind-dependent forecast, using ${l.construction_worlds} separate construction fields, full backup rollouts and spline extrema. Baseline planning uses still air and no planning filter. Both execute the original hard RPCBF with the current learned wind belief${e?`, ${e.construction_fields} coherent construction fields, a ${e.safety_margin_m.toFixed(2)} m barrier margin and posterior/field refresh every ${update} s`:''}.`;
      const belief=$('monitor-belief-numerics');if(belief)belief.textContent='The old implementation’s C1 wind-table interpolation at 0.2 m spacing is restored. Tables are sampled from the full current POAM posterior at each replan. No anticipated measurements, optimistic objective or information bonus is used.';
      const integration=$('monitor-integration-details');if(integration)integration.textContent=`Planning uses the old ${l.dt} s RK4 forecast step and 32-bit arithmetic. Executed robot dynamics and clarity retain 0.01 s RK4 with 64-bit arithmetic; execution RPCBF runs every 0.05 s. All 1,884 free-space clarity cells are retained. Warm planning has a 1 s runtime cutoff. First-call compilation is reported separately. A finite sampling budget and learned wind belief do not guarantee safety.`;
      $('monitor-initialization-details').innerHTML='<h3>Geometric warm starts</h3><p>The current circle geometry is searched online. Six clarity-guided paths include alternative passages; a hold completes the seven initial modes. Shifted previous controls remain an additional evaluated candidate. Paths initialize full acceleration sequences and do not constrain the final solution.</p>';
      const timing=$('monitor-runtime');if(timing)timing.textContent=`Measured warm planning: ${r.warm_min_s.toFixed(3)}–${r.warm_max_s.toFixed(3)} s across ${r.warm_calls} calls. Includes geometric initialization, wind sampling, MBD and final validation. First calls, including compilation: ${r.first_call_min_s.toFixed(2)}–${r.first_call_max_s.toFixed(2)} s. ${manifest.initial_wind_knowledge==='survey'?'Same measured wind map supplied to both methods.':'Initially unknown wind; no pre-mission survey.'}`;
      if(timing&&r.rejected_calls?.length)timing.textContent+=` Runtime exceptions: ${r.rejected_calls.map(p=>`${p.id} at ${p.t.toFixed(0)} s took ${p.wall_seconds.toFixed(3)} s`).join('; ')}. The guard stopped before applying these plans; execution continued from saved checkpoints. These overruns are retained in the audit and are not included in the accepted-call range above.`;
    }
    $('monitor-scenario').addEventListener('change',()=>load());$('monitor-layer').addEventListener('change',draw);
    $('monitor-belief').addEventListener('change',draw);$('monitor-plans').addEventListener('change',draw);$('monitor-seeds').addEventListener('change',draw);
    $('monitor-play').addEventListener('click',()=>{if(time>=duration)time=0;setPlaying(!playing);draw();});
    $('monitor-restart').addEventListener('click',()=>{time=0;setPlaying(true);draw();});
    $('monitor-time').addEventListener('input',e=>{time=Number(e.target.value);setPlaying(false);draw();});
    await load();
    setInterval(()=>{if(manifest.state==='computing'||!records.length||records.some(r=>!r.summary.completed))load(true);},30000);
    function frame(t){const dt=previous===null?0:Math.min(.1,(t-previous)/1000);previous=t;if(playing&&!document.hidden){time=Math.min(duration,time+dt*Number($('monitor-speed').value));draw();if(time>=duration)setPlaying(false);}requestAnimationFrame(frame);}requestAnimationFrame(frame);
  }catch(e){$('monitor-status').textContent=e.message;}
}
document.querySelectorAll('[data-monitoring-directory],#monitoring-review').forEach(root=>monitoringComparison(root));
