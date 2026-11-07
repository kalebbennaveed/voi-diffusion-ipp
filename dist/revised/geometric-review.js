'use strict';
// Display only: all forecasts and execution are existing recorded trajectories.
(async function geometricReview(){
  const root=document.querySelector('#geometric-paths');if(!root)return;
  const $=s=>root.querySelector(s),colors=['#b45309','#2563eb','#8b4a9e','#be3455','#087f8c','#61842b'];
  const cache=new Map(),NS='http://www.w3.org/2000/svg';
  let env,manifest,record,frame,stage=1,selectedPath=-1,playing=false,started=0,raf=0,loadId=0;
  const descriptions=[
    ['Where will sensing help?', 'Predict clarity decay without sensing at 3.75, 7.5, and 15 seconds. Combine that deficit with the sensor footprint to identify useful sensing locations.', 'The orange map is an approximate sensing-value score for initialization. Travel distance and clearance also affect destination ranking. It is not wind uncertainty or the final MBD objective.'],
    ['Find different geometric paths', 'Dijkstra search supplies paths to four spatially separated sensing locations. Two more starts seek alternative passages by temporarily blocking discovered bottlenecks.', 'These are the actual six coarse paths in this planning call. Some may share segments or destinations. No named central, upper, or lower route is forced.'],
    ['Make double-integrator references', 'Round the corners and retime each path with acceleration and curvature limits. The resulting 60 acceleration pairs initialize independent MBD chains.', 'Solid curves are reconstructed from the saved control sequences. Dashed curves are their coarse paths. Only the first 15 seconds of each reference are shown; a longer geometric path may not finish within the horizon.'],
    ['Evaluate wind, filtering, and clarity', 'MBD evaluates the candidates with known wind, relaxed RPCBF, and the clarity dynamics, then retains the best evaluated sequence. All 120 acceleration coordinates remain free.', 'The 15-second forecast uses the relaxed planning filter. Only the first 5 seconds are executed with the hard filter before replanning. The reference, forecast, and execution can differ.']
  ];
  const screen=p=>[p[0]*100,600-p[1]*100];
  const points=ps=>ps.map(p=>screen(p).map(v=>v.toFixed(3)).join(',')).join(' ');
  async function read(url){const r=await fetch(url);if(!r.ok)throw Error(`Cannot load paths (${r.status})`);if(!url.endsWith('.gz'))return r.json();const bytes=new Uint8Array(await r.arrayBuffer());return bytes[0]===31?new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).json():JSON.parse(new TextDecoder().decode(bytes));}
  function setPlaying(value){playing=value;$('#geometry-play').textContent=value?'Ⅱ Pause construction':'▶ Play construction';$('#geometry-play').setAttribute('aria-pressed',String(value));cancelAnimationFrame(raf);if(value){started=performance.now();raf=requestAnimationFrame(animate);}}
  function polyline(ps,color,width=3,dashed=false,animateLine=false,opacity=1){const el=document.createElementNS(NS,'polyline');el.setAttribute('points',points(ps));el.setAttribute('fill','none');el.setAttribute('stroke',color);el.setAttribute('stroke-width',width);el.setAttribute('stroke-linejoin','round');el.setAttribute('stroke-linecap','round');el.setAttribute('opacity',opacity);if(dashed)el.setAttribute('stroke-dasharray','7 6');if(animateLine)el.dataset.trace='true';$('#geometry-path-layer').append(el);return el;}
  function heat(){const c=document.createElement('canvas');[c.width,c.height]=env.grid;const ctx=c.getContext('2d'),im=ctx.createImageData(c.width,c.height),max=Math.max(...frame.sensing_value,1e-8);const stops=[[255,247,236],[253,187,132],[179,0,0]];
    const free=new Set(env.free_indices);frame.sensing_value.forEach((v,i)=>{if(!free.has(i))return;const z=Math.min(1,v/max)*2,a=Math.min(1,Math.floor(z)),f=z-a,k=((c.height-1-Math.floor(i/c.width))*c.width+i%c.width)*4;for(let j=0;j<3;j++)im.data[k+j]=stops[a][j]*(1-f)+stops[a+1][j]*f;im.data[k+3]=215;});ctx.putImageData(im,0,0);$('#geometry-heat').setAttribute('href',c.toDataURL());}
  function draw(){
    if(!frame)return;const [title,text,note]=descriptions[stage];$('#geometry-step-title').textContent=title;$('#geometry-explanation').textContent=text;$('#geometry-note').textContent=note;
    root.querySelectorAll('[data-geometry-stage]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.geometryStage)===stage)));
    $('#geometry-step-output').textContent=`Step ${stage+1} of 4 · recorded plan at ${frame.t} s`;
    $('#geometry-path-layer').replaceChildren();$('#geometry-destinations').replaceChildren();
    $('#geometry-grid').style.display=stage===0?'':'none';$('#geometry-heat').style.display=stage===0?'':'none';
    const position=screen(frame.state);$('#geometry-start').setAttribute('cx',position[0]);$('#geometry-start').setAttribute('cy',position[1]);
    $('#geometry-start-label').setAttribute('x',position[0]+12);$('#geometry-start-label').setAttribute('y',position[1]-12);
    if(stage<3){
      frame.coarse.forEach((path,i)=>{const active=selectedPath<0||selectedPath===i,opacity=active?1:.12;
        if(stage===1)polyline(path,colors[i],active?3.4:2,false,active,opacity);
        if(stage===2){polyline(path,colors[i],1.5,true,false,opacity*.45);polyline(frame.seeds[i+1],colors[i],active?3.4:2,false,active,opacity);}
        const end=screen(path[path.length-1]),node=document.createElementNS(NS,'circle');node.setAttribute('cx',end[0]);node.setAttribute('cy',end[1]);node.setAttribute('r',6+i*.15);node.setAttribute('fill',colors[i]);node.setAttribute('stroke','white');node.setAttribute('stroke-width',1.5);node.setAttribute('opacity',opacity);$('#geometry-destinations').append(node);
      });
    }else{
      polyline(frame.selected_reference,'#7c3fa0',2.5,true);
      polyline(frame.forecast,'#147861',4,false,true);
      polyline(frame.executed,'#202f43',4,false,true);
    }
    $('#geometry-path-buttons').hidden=stage===3;
    $('#geometry-legend').innerHTML=stage===0?'<span>Footprint-weighted predicted deficit <span class="geometry-heat-scale"></span> lower → higher (relative scale)</span><span>Dots: geometric-path destinations</span>':stage===1?'<span>Colors identify P1–P6. Select a path to separate overlapping alternatives.</span>':stage===2?'<span><i style="--line:#64748b"></i>15 s reference</span><span><i class="dashed" style="--line:#64748b"></i>Coarse geometric path</span>':'<span><i class="dashed" style="--line:#7c3fa0"></i>Selected reference · 15 s</span><span><i style="--line:#147861"></i>Wind + relaxed RPCBF forecast · 15 s</span><span><i style="--line:#202f43"></i>Executed with hard RPCBF · first 5 s</span>';
    const matches=frame.selected_seed_indices.map(i=>manifest.seed_order[i]);
    $('#geometry-selected').textContent=frame.selected_origin==='diffusion'?'Diffusion candidate':matches.join(' / ')||'Initialized sequence';
    $('#geometry-selected-cost').textContent=frame.selected_cost.toFixed(3);
    $('#geometry-timing').textContent=frame.initialization.total_seed_milliseconds.toFixed(1)+' ms';
    $('#geometry-cost-body').innerHTML=frame.seed_costs.map((v,i)=>`<tr class="${frame.selected_seed_indices.includes(i)?'selected':''}"><th scope="row">${i>0&&i<7?`<span style="color:${colors[i-1]}">●</span> `:''}${manifest.seed_order[i]}</th><td>${v.toFixed(3)}</td></tr>`).join('');
    const fallbacks=frame.initialization.retiming_fallbacks.filter(Boolean).length;
    $('#geometry-source').textContent=`Recorded disturbance-aware planning call at ${frame.t} s · ${frame.initialization.nodes} free graph nodes · ${frame.initialization.edges} directed edges · ${fallbacks} retiming fallbacks. ${frame.forecast_contact?'Selected forecast includes contact. ':'Selected forecast has no contact. '}All existing mission recordings are unchanged.`;
    root.dataset.stage=String(stage);root.dataset.loaded=record.id;root.dataset.planTime=String(frame.t);
  }
  function animate(now){if(!playing)return;if(document.hidden){setPlaying(false);return;}const progress=Math.min(1,(now-started)/2800);root.querySelectorAll('[data-trace]').forEach(el=>{const length=el.getTotalLength();el.setAttribute('stroke-dasharray',`${length} ${length}`);el.setAttribute('stroke-dashoffset',String(length*(1-progress)));});if(progress===1){if(stage===3){setPlaying(false);draw();return;}stage++;started=now;draw();}raf=requestAnimationFrame(animate);}
  async function load(){const ticket=++loadId;setPlaying(false);$('#geometry-status').textContent='Loading recorded geometric paths…';root.querySelectorAll('button').forEach(b=>b.disabled=true);
    const id=`s${$('#geometry-scenario').value}-${$('#geometry-case').value}-aware`;
    try{if(!cache.has(id))cache.set(id,read(`data/geometric-review/${id}.json.gz`).catch(e=>{cache.delete(id);throw e;}));const loaded=await cache.get(id);if(ticket!==loadId)return;record=loaded;frame=record.frames.find(f=>f.t===Number($('#geometry-time').value));heat();draw();$('#geometry-status').textContent='';root.querySelectorAll('button').forEach(b=>b.disabled=false);}catch(e){if(ticket===loadId)$('#geometry-status').textContent=e.message;}
  }
  try{
    [manifest,env]=await Promise.all([read('data/geometric-review/manifest.json'),read('data/monitoring-cluster-300/environment.json')]);
    $('#geometry-case').innerHTML=manifest.conditions.map(c=>`<option value="${c.id}">${c.label}</option>`).join('');
    $('#geometry-time').innerHTML=manifest.times.map(t=>`<option value="${t}">${t} s${t===0?' · initial plan':''}</option>`).join('');
    const regions=env.regions.map(r=>{const [lo,hi]=r.bounds;return `<rect x="${lo[0]*100}" y="${600-hi[1]*100}" width="${(hi[0]-lo[0])*100}" height="${(hi[1]-lo[1])*100}" fill="#e3f0e9" fill-opacity=".25" stroke="#6e9383" stroke-width="2"/><text x="${lo[0]*100+12}" y="${600-hi[1]*100+27}" font-size="22" fill="#365c4c">Region ${r.label}</text>`;}).join('');
    $('#geometry-map').innerHTML=`<svg viewBox="-8 -8 816 616" role="img" aria-label="Recorded geometric path construction in the five-column environment"><defs><pattern id="geometry-grid-pattern" width="12.5" height="12.5" patternUnits="userSpaceOnUse"><path d="M12.5 0H0V12.5" fill="none" stroke="#61798a" stroke-opacity=".2" stroke-width=".7"/></pattern><clipPath id="geometry-clip"><rect width="800" height="600"/></clipPath></defs><rect width="800" height="600" fill="#fafcfb"/><image id="geometry-heat" width="800" height="600"/><rect id="geometry-grid" width="800" height="600" fill="url(#geometry-grid-pattern)"/>${regions}${env.obstacles.map(o=>`<circle cx="${100*o[0]}" cy="${600-100*o[1]}" r="${100*o[2]}" fill="#73828a"/>`).join('')}<g clip-path="url(#geometry-clip)"><g id="geometry-path-layer"></g><g id="geometry-destinations"></g><circle id="geometry-start" r="7" fill="#1f2937" stroke="white" stroke-width="2"/><text id="geometry-start-label" font-size="17" fill="#1f2937" stroke="white" stroke-width="3" paint-order="stroke">Start</text></g><rect width="800" height="600" fill="none" stroke="#687d73" stroke-width="1.5"/></svg>`;
    $('#geometry-path-buttons').innerHTML='<button type="button" data-path="-1" aria-pressed="true">All paths</button>'+colors.map((c,i)=>`<button type="button" data-path="${i}" aria-pressed="false"><i style="--path:${c}"></i>P${i+1}</button>`).join('');
    root.querySelectorAll('[data-path]').forEach(b=>b.addEventListener('click',()=>{setPlaying(false);selectedPath=Number(b.dataset.path);root.querySelectorAll('[data-path]').forEach(p=>p.setAttribute('aria-pressed',String(p===b)));draw();}));
    root.querySelectorAll('[data-geometry-stage]').forEach(b=>b.addEventListener('click',()=>{setPlaying(false);stage=Number(b.dataset.geometryStage);draw();}));
    for(const id of ['scenario','case','time'])$('#geometry-'+id).addEventListener('change',load);
    $('#geometry-play').addEventListener('click',()=>{if(playing){setPlaying(false);draw();}else{stage=0;draw();setPlaying(true);}});
    $('#geometry-next').addEventListener('click',()=>{setPlaying(false);stage=(stage+1)%4;draw();});
    await load();
  }catch(e){$('#geometry-status').textContent=`Geometric-path visualization: ${e.message}`;}
})();
