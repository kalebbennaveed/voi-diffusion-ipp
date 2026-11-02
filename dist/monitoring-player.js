'use strict';
// Recorded MBD missions only. Playback never integrates robot or clarity models.
(async function monitoringComparison(){
  const root=document.querySelector('#monitoring-review');if(!root)return;
  const $=id=>root.querySelector('#'+id),colors={aware:'#147861',baseline:'#c46a35'};
  let env,manifest,records=[],panels=[],time=0,duration=60,playing=false,previous=null,version=0;
  const cache=new Map();
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
    const visited=env.regions.filter(region=>record.samples.some(s=>s[1]>=region.bounds[0][0]&&s[1]<=region.bounds[1][0]&&s[2]>=region.bounds[0][1]&&s[2]<=region.bounds[1][1])).map(region=>region.label).join(', ')||'none';
    const summary=record.summary,name=method==='aware'?'Proposed · disturbance-aware MBD':'Baseline · MBD + execution RPCBF';
    const regions=design.regions.map(r=>`<rect x="${r.x}" y="${r.y}" width="${r.width}" height="${r.height}" fill="none" stroke="#477763" stroke-width="1.8"/><text x="${r.x+18}" y="${r.y+28}" class="region-label">Region ${r.label}</text>`).join('');
    const obstacles=design.obstacles.map(o=>`<circle cx="${o.center[0]}" cy="${o.center[1]}" r="${o.radius}" fill="#73828a"/>`).join('');
    article.innerHTML=`<div class="robot-card-heading"><h4>${name}</h4><span class="outcome ${summary.contact?'collision':'reached'}">${summary.contact?'Full run: contact':summary.completed?'Completed':'Computing'}</span></div>
    <p class="monitor-method">${method==='aware'?'Current wind belief + relaxed RPCBF inside planning.':'Still-air planning. Wind-aware RPCBF after optimization.'}</p>
    <div class="robot-map"><svg viewBox="-5 -5 810 472.5" role="img" aria-label="${name}, scenario ${record.scenario}, executed monitoring and clarity">
    <defs><clipPath id="${record.id}-monitor-clip"><rect width="800" height="462.5"/></clipPath></defs><rect width="800" height="462.5" fill="#fafcf9"/><image class="monitor-heat" width="800" height="462.5"/>
    ${regions}${obstacles}<g clip-path="url(#${record.id}-monitor-clip)"><g class="monitor-belief-arrows"></g><g class="monitor-forecasts"></g><polyline class="monitor-trace" fill="none" stroke="${colors[method]}" stroke-width="3.2"/><circle class="monitor-dot" r="6.5" fill="${colors[method]}" stroke="white" stroke-width="2.2"/><path class="monitor-contact" d="M-8 -8L8 8M-8 8L8 -8" stroke="#a93136" stroke-width="3" display="none"/></g><rect width="800" height="462.5" fill="none" stroke="#85938d"/>${renderWindOpenings(record.scenario)}</svg></div>
    <div class="live-metrics"><div><span>Clarity · A</span><b data-metric="a">—</b></div><div><span>Clarity · B</span><b data-metric="b">—</b></div><div><span>Whole-space deficit</span><b data-metric="deficit">—</b></div><div><span>Accumulated cost</span><b data-metric="cost">—</b></div></div>
    <p class="monitor-readout"></p><div class="monitor-region-chart"></div>
    <div class="run-summary">${summary.completed?'Full mission':`Recorded ${summary.simulated_seconds.toFixed(0)} s`}: cost <b>${summary.integrated_deficit.toFixed(3)}</b> · clearance <b>${(Math.abs(summary.min_clearance_m)<1e-12?0:summary.min_clearance_m).toFixed(3)} m</b> · infeasible execution steps <b>${summary.infeasible_execution_steps}</b> · fallback periods <b>${summary.fallback_periods}</b> · regions visited <b>${visited}</b>${summary.contact?`<br>Contact at ${summary.contact.time_s.toFixed(2)} s; subsequent clarity decays without sensing.`:''}</div>`;
    return {article,record,lastHeat:'',lastBelief:-1,lastPlan:-1};
  }
  function paint(panel){
    const record=panel.record,layer=$('monitor-layer').value;
    const qi=nearest(record.clarity_frames,time),bi=nearest(record.belief_frames,time),key=layer+':'+(layer==='uncertainty'?bi:layer==='decay'?0:qi);
    if(panel.lastHeat===key)return;panel.lastHeat=key;
    const canvas=document.createElement('canvas');canvas.width=64;canvas.height=37;
    const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(64,37),q=record.clarity_frames[qi].q;
    for(let n=0;n<env.free_indices.length;n++){
      const cell=env.free_indices[n],x=cell%64,y=Math.floor(cell/64),index=((36-y)*64+x)*4;
      let value,low,high;
      if(layer==='clarity'){value=q[n];low=[237,157,105];high=[224,241,220];}
      else if(layer==='deficit'){value=Math.max(0,env.targets[n]-q[n])/.7;low=[250,251,235];high=[222,99,49];}
      else if(layer==='decay'){value=env.process_noise[n]/.16;low=[246,246,229];high=[180,107,154];}
      else{value=record.belief_frames[bi].std[cell]/3.;low=[246,250,252];high=[33,110,157];}
      value=Math.max(0,Math.min(1,value));for(let c=0;c<3;c++)pixels.data[index+c]=Math.round(low[c]+value*(high[c]-low[c]));pixels.data[index+3]=255;
    }
    ctx.putImageData(pixels,0,0);panel.article.querySelector('.monitor-heat').setAttribute('href',canvas.toDataURL());
  }
  function drawPanel(panel){
    const {article,record}=panel,s=sample(record.samples,time),p=screen(s.slice(1,3));
    paint(panel);article.querySelector('.monitor-trace').setAttribute('points',line([...record.samples.filter(r=>r[0]<time).map(r=>r.slice(1,3)),s.slice(1,3)]));
    const dot=article.querySelector('.monitor-dot');dot.setAttribute('cx',p[0]);dot.setAttribute('cy',p[1]);
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
    }
    article.querySelector('.monitor-belief-arrows').style.display=$('monitor-belief').checked?'':'none';
    let pi=record.plans.length?nearest(record.plans,time):-1;
    if(pi!==panel.lastPlan){
      panel.lastPlan=pi;article.querySelector('.monitor-forecasts').innerHTML=pi<0?'':record.plans[pi].predicted_states.map(rows=>`<polyline points="${line(rows.map(r=>r.slice(0,2)))}" fill="none" stroke="#596d76" stroke-width="1.3" stroke-dasharray="5 5" opacity=".7"/>`).join('');
    }
    article.querySelector('.monitor-forecasts').style.display=$('monitor-plans').checked?'':'none';
    article.querySelector('.monitor-readout').textContent=`Speed ${Math.hypot(s[3],s[4]).toFixed(2)} m/s · clarity target A/B 0.70 · ${bf.observations} wind measurements · belief at ${bf.t.toFixed(1)} s${s[18]?'': ' · stopped after contact'}`;
    article.querySelector('.monitor-chart-cursor')?.setAttribute('x1',40+530*time/duration);article.querySelector('.monitor-chart-cursor')?.setAttribute('x2',40+530*time/duration);
  }
  function chart(panel){
    const r=panel.record,path=column=>r.samples.filter(s=>s[0]<=duration).map(s=>`${40+530*s[0]/duration},${145-115*s[column]}`).join(' ');
    panel.article.querySelector('.monitor-region-chart').innerHTML=`<p>Mean clarity: <span style="color:#8e507f">A</span> · <span style="color:#207c88">B</span> · dashed target 0.70</p><svg viewBox="0 0 600 170" role="img" aria-label="Region A and B mean clarity through the mission"><path d="M40 20V145H575" fill="none" stroke="#b8c8bf"/><path d="M40 ${145-115*.7}H575" stroke="#8c9991" stroke-dasharray="5 4"/><polyline points="${path(15)}" fill="none" stroke="#8e507f" stroke-width="2"/><polyline points="${path(16)}" fill="none" stroke="#207c88" stroke-width="2"/><text x="18" y="31">1</text><text x="18" y="149">0</text><text x="40" y="164">0 s</text><text x="545" y="164">${duration} s</text><line class="monitor-chart-cursor" y1="20" y2="145" stroke="#4d5b54"/></svg>`;
  }
  function draw(){
    $('monitor-time').value=time;$('monitor-clock').textContent=`${time.toFixed(1)} / ${duration.toFixed(1)} s`;panels.forEach(drawPanel);
    const layer=$('monitor-layer').value;$('monitor-scale').textContent=layer==='clarity'?'Clarity: 0 → 1':layer==='deficit'?'Deficit below local target: 0 → 0.70':layer==='decay'?'Process noise Q: background 0.02 · A/B 0.16':'Wind-belief standard deviation: 0 → 3 m/s';
  }
  async function load(preserve=false){
    const savedTime=time,savedPlaying=playing,generation=++version;if(!preserve)setPlaying(false);$('monitor-status').textContent='Loading computed MBD missions…';
    try{
      const scenario=Number($('monitor-scenario').value);
      const ids=['aware','baseline'].map(method=>`s${scenario}-${method}`);
      const loaded=await Promise.all(ids.map(async id=>{if(!cache.get(id)?.summary.completed){const response=await fetch(`data/monitoring/${id}.json`,{cache:'no-store'});if(!response.ok)throw Error('This mission has not finished its first planning period yet. Computation continues.');cache.set(id,await response.json());}return cache.get(id);}));
      if(generation!==version)return;records=loaded;duration=Math.min(...records.map(r=>r.summary.simulated_seconds));time=preserve?Math.min(savedTime,duration):0;
      $('monitor-time').max=duration;panels=records.map(makePanel);$('monitor-panels').replaceChildren(...panels.map(p=>p.article));panels.forEach(chart);draw();
      $('monitor-play').disabled=false;$('monitor-restart').disabled=false;
      $('monitor-status').textContent=`Scenario ${scenario} · ${records.every(r=>r.summary.completed)?`${duration.toFixed(0)} seconds completed`:`showing ${duration.toFixed(0)} of 60 seconds; computation continues`} · same initial state, clarity, wind field, noise stream and MBD budget · each method learns from its own executed measurements.`;
      if(preserve)setPlaying(savedPlaying);
      $('monitor-results').textContent=records.every(r=>r.summary.completed)?records.map(r=>`${r.method==='aware'?'Proposed':'Baseline'}: integrated whole-space deficit ${r.summary.integrated_deficit.toFixed(3)}; ${r.summary.contact?'contact':'no contact'}; ${r.summary.passage_crossings.length} passage crossings.`).join(' '):`Common computed prefix: ${duration.toFixed(0)} s. Accumulated cost at that same time: ${records.map(r=>`${r.method==='aware'?'proposed':'baseline'} ${sample(r.samples,duration)[14].toFixed(3)}`).join(' · ')}. Full mission comparison is pending.`;
    }catch(e){$('monitor-status').textContent=e.message;}
  }
  try{
    const response=await fetch('data/monitoring/manifest.json',{cache:'no-store'});if(!response.ok)throw Error('Monitoring missions are being computed. Existing simulations above remain available.');manifest=await response.json();
    const sourceLink=root.querySelector('a[download]');if(manifest.state==='computing'){sourceLink.removeAttribute('href');sourceLink.textContent='Source package follows the completed mission audit';}
    env=await(await fetch('data/monitoring/environment.json',{cache:'no-store'})).json();
    $('monitor-scenario').addEventListener('change',()=>load());$('monitor-layer').addEventListener('change',draw);
    $('monitor-belief').addEventListener('change',draw);$('monitor-plans').addEventListener('change',draw);
    $('monitor-play').addEventListener('click',()=>{if(time>=duration)time=0;setPlaying(!playing);draw();});
    $('monitor-restart').addEventListener('click',()=>{time=0;setPlaying(true);draw();});
    $('monitor-time').addEventListener('input',e=>{time=Number(e.target.value);setPlaying(false);draw();});
    await load();
    setInterval(()=>{if(!records.length||records.some(r=>!r.summary.completed))load(true);},30000);
    function frame(t){const dt=previous===null?0:Math.min(.1,(t-previous)/1000);previous=t;if(playing&&!document.hidden){time=Math.min(duration,time+dt*Number($('monitor-speed').value));draw();if(time>=duration)setPlaying(false);}requestAnimationFrame(frame);}requestAnimationFrame(frame);
  }catch(e){$('monitor-status').textContent=e.message;}
})();
