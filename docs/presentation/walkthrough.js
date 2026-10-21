/* Manual, independent teaching players. Schematic method figures never claim mission results. */
(function(){
 'use strict';
 const navy='#00274c',blue='#2474bf',orange='#d77717',purple='#a13b8e',green='#008653',gray='#80929e';
 const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
 function canvas(prefix){
  const a=[`<defs><marker id="${prefix}-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="context-stroke"/></marker></defs>`];
  return {a,
   text(x,y,t,size=20,c=navy,anchor='start',weight=400){a.push(`<text x="${x}" y="${y}" font-family="Arial,Helvetica,sans-serif" font-size="${size}" fill="${c}" text-anchor="${anchor}" font-weight="${weight}">${esc(t)}</text>`)},
   line(x,y,xx,yy,c=navy,w=2,arrow=false,dash=false){a.push(`<line x1="${x}" y1="${y}" x2="${xx}" y2="${yy}" stroke="${c}" stroke-width="${w}" ${arrow?`marker-end="url(#${prefix}-arrow)"`:''} ${dash?'stroke-dasharray="7 5"':''}/>`);},
   path(d,c=blue,w=3,dash=false,fill='none'){a.push(`<path d="${d}" stroke="${c}" stroke-width="${w}" fill="${fill}" ${dash?'stroke-dasharray="7 5"':''}/>`);},
   rect(x,y,w,h,fill='#f5f8fa',stroke='#b9cad8',rx=8){a.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}"/>`)},
   circle(x,y,r=6,c=blue){a.push(`<circle cx="${x}" cy="${y}" r="${r}" fill="${c}"/>`)},
   result(){return a.join('')}
  };
 }
 function curve(x,y){return x.map((v,i)=>`${i?'L':'M'}${v.toFixed(2)},${y[i].toFixed(2)}`).join(' ')}
 function poam(step){
  const c=canvas('poam'),d=window.poamTeachingData;
  const n=step<2?0:step<4?1:step===4?2:3,s=d.states[n],old=d.states[Math.max(0,n-1)];
  const xx=r=>76+r*76, yy=v=>164-v*70, sy=v=>383-v*65;
  c.text(76,27,'Wind estimate along the environment',24,navy,'start',700);
  c.text(76,53,'Mean ± 2 standard deviations',18,blue);
  c.text(837,53,'Purple: measured wind',18,purple,'end');
  [-1,0,1].forEach(v=>{c.line(76,yy(v),836,yy(v),'#e1e8ee',1);c.text(62,yy(v)+6,v,17,navy,'end')});
  [0,2,4,6,8,10].forEach(v=>{c.line(xx(v),66,xx(v),269,'#e8edf2',1);c.text(xx(v),290,v,17,navy,'middle')});
  c.text(16,169,'d(r)',18);
  const x=d.position.map(xx),upper=s.mean.map((v,i)=>yy(v+2*s.std[i])),lower=s.mean.map((v,i)=>yy(v-2*s.std[i]));
  c.path(curve(x,upper)+' '+curve([...x].reverse(),[...lower].reverse()).replace(/^M/,'L')+' Z','none',0,false,'#d4e7f7');
  c.path(curve(x,s.mean.map(yy)),blue,3);
  const visible=step===1?1:n;
  d.observations.slice(0,visible).forEach(([r,v],i)=>{c.line(xx(r),yy(v-.2),xx(r),yy(v+.2),purple,2);c.circle(xx(r),yy(v),6,purple);c.text(xx(r)+10,yy(v)-10,i+1,17,purple)});
  d.inducing.forEach(z=>c.path(`M${xx(z)},262 l5,5 l-5,5 l-5,-5 Z`,navy,1,false,navy));
  c.text(456,312,'Position r · Diamonds: inducing positions (not observations)',18,navy,'middle');
  c.text(76,337,'Predictive standard deviation · dashed: initial',19,blue);
  c.line(76,383,836,383,'#e1e8ee',1);c.line(76,sy(.55),836,sy(.55),'#edf1f5',1);
  c.path(curve(x,d.states[0].std.map(sy)),gray,1.5,true);
  c.path(curve(x,s.std.map(sy)),blue,3);
  c.text(62,388,'0',16,navy,'end');c.text(62,sy(.55)+5,'0.55',16,navy,'end');
  c.text(76,419,n?`Spatial mean of std. dev.: ${s.mean_std.toFixed(3)}   (initial ${d.states[0].mean_std.toFixed(3)})`:'Spatially correlated prior · observation-noise std. dev.: 0.10',20,navy);
  if(step===3){const qi=60,r=d.position[qi];c.line(xx(r),68,xx(r),258,purple,2,true,true);c.circle(xx(r),yy(s.mean[qi]),6,blue);c.rect(455,74,355,68,'white');c.text(472,101,'Unobserved position r = 3.0',19,purple);c.text(472,128,`Mean ${s.mean[qi].toFixed(2)} · std. dev. ${s.std[qi].toFixed(2)}`,19,blue);}
  return c.result();
 }

 function environment(c,step){
  c.rect(20,48,510,330,'#fbfcfd');
  c.text(36,78,'Predicted motion in one possible field',21,navy,'start',700);
  c.rect(48,268,97,72,'#fff1b8','#d7b54a');c.text(96,311,'Region A',19,navy,'middle');
  c.rect(415,100,94,72,'#fff1b8','#d7b54a');c.text(462,183,'Region B',19,navy,'middle');
  c.circle(303,259,42,'#b4c0c8');c.text(303,317,'Obstacle',18,gray,'middle');
  [[75,108],[175,108],[275,108],[375,108],[72,202],[175,202],[370,335],[460,335]].forEach(([x,y])=>c.line(x,y,x+23,y-11,'#8a9ba8',1.8,true));
  const prefix='M96,276 C145,263 191,232 263,190';
  const after='M263,190 C304,140 358,122 455,130';
  const frozen='M263,190 C330,208 389,193 455,160';
  if(step>=1)c.path('M96,276 Q266,243 455,130',orange,3,true);
  if(step>=3)c.path(prefix,blue,5);
  if(step>=4){c.circle(263,190,8,purple);c.text(258,172,'rₘ',21,purple,'end');}
  if(step>=6){c.path(frozen,gray,3,true);c.path(after,blue,5);}
  if(step>=2&&step<3){
   c.line(164,248,235,230,orange,4,true);c.line(164,248,194,188,blue,4,true);
   c.text(195,177,'u',22,blue);c.text(235,254,'uₙₒₘ',19,orange);
  }
  const pos=step<3?[96,276]:step<6?[263,190]:[455,130];
  c.circle(...pos,9,navy);c.circle(...pos,4,'white');
  c.text(36,405,step>=6?'Gray dashed: continuation with the frozen belief':'Gray arrows: possible physical wind (fixed for this evaluation)',18,step>=6?gray:navy);
 }
 function rollout(step){
  const c=canvas('rollout');
  environment(c,step);
  c.text(20,28,step<5?'Wind-aware execution prediction':'Anticipated learning changes the continuation',22,navy,'start',700);
  const x=551,w=309;
  if(step===0){
   c.rect(x,55,w,318);c.text(x+18,91,'Current information',23,navy,'start',700);
   ['Robot state xₜ','Clarity qₜ','Wind belief Bₜᵈ','Geometry and constraints','Target clarity'].forEach((s,i)=>c.text(x+18,139+i*47,s,21));
  }else if(step===1){
   c.rect(x,55,w,318);c.text(x+18,92,'Hold these fixed',23,navy,'start',700);
   c.text(x+18,142,'One policy vector θ',22,orange);c.text(x+18,174,'sampled by MBD',20);
   c.text(x+18,222,'One possible field d',22,gray);c.text(x+18,254,'drawn from the belief',20);
   c.text(x+18,316,'The controller knows Bᵈ,',20);c.text(x+18,344,'not the hidden field d.',20);
  }else if(step===2){
   c.rect(x,55,w,318);c.text(x+18,91,'Predict controller response',20,navy,'start',700);
   c.text(x+18,131,'Belief → design-policy paths',18);
   c.text(x+18,160,'→ maximum constraint value',18);
   c.text(x+18,189,'→ value-rate inequalities',18);
   c.line(x+20,216,x+w-20,216,'#b9cad8',1);
   c.text(x+18,251,'Choose u closest to uₙₒₘ',20,blue);
   c.text(x+18,282,'within U and the tested',20);c.text(x+18,310,'constraints, when feasible.',20);
   c.text(x+18,351,'Recompute at each state.',19,navy,'start',700);
  }else if(step===3){
   c.rect(x,55,w,318);c.text(x+18,93,'Motion determines sensing',21,navy,'start',700);
   c.text(x+18,146,'Applied input u + field d',21,blue);c.line(x+150,165,x+150,195,blue,2,true);
   c.text(x+18,228,'Reached state x → sensing',20);c.line(x+150,247,x+150,278,blue,2,true);
   c.text(x+18,311,'Update clarity q',23,blue);c.text(x+18,345,'Accumulate clarity deficit.',19);
  }else if(step===4){
   c.rect(x,55,w,318);c.text(x+18,94,'A future measurement',23,purple,'start',700);
   c.text(x+18,145,'Use the reached position rₘ.',20);
   c.text(x+18,191,'Read the same possible field.',19);
   c.text(x+18,238,'Add observation noise.',20);
   c.text(x+18,302,'This is a predicted value.',20,purple);c.text(x+18,337,'No real measurement yet.',20);
   c.line(274,190,542,190,purple,2,true);
  }else if(step===5||step===6){
   c.rect(x,55,w,318);c.text(x+18,91,'Wind belief before / after',21,navy,'start',700);
   const pd=window.poamTeachingData,ps=pd.states[1],px=pd.position.map(r=>x+24+(w-48)*r/10),py=v=>205-v*48;
   const up=ps.mean.map((m,i)=>py(m+2*ps.std[i])),lo=ps.mean.map((m,i)=>py(m-2*ps.std[i]));
   c.path(curve(px,up)+' '+curve([...px].reverse(),[...lo].reverse()).replace(/^M/,'L')+' Z','none',0,false,'#d4e7f7');
   c.path(curve(px,ps.mean.map(py)),blue,3);
   c.line(x+24,205,x+w-24,205,gray,1.5,false,true);
   c.circle(x+24+(w-48)*2.2/10,py(.65),6,purple);
   c.text(x+18,291,'Blue: mean ±2 std. dev.',19,blue);
   c.text(x+18,326,step===5?'Carry forward x and q.':'Recompute policy and controller.',18);
   c.text(x+18,354,step===5?'Keep the same θ and field d.':'No nested MBD optimization.',18,navy,'start',700);
   if(step===5)c.line(274,190,542,190,purple,2,true);
  }else{
   c.rect(x,55,w,318);c.text(x+18,92,'Full cost with learning',23,navy,'start',700);
   c.line(x+35,285,x+w-20,285,navy,2,true);c.line(x+35,285,x+35,130,navy,2,true);
   c.path(`M${x+35},168 Q${x+90},203 ${x+120},195 T${x+230},242 L${x+230},285 L${x+35},285 Z`,'none',0,false,'#d4e7f7');
   c.path(`M${x+35},168 Q${x+90},203 ${x+120},195 T${x+230},242`,blue,3);
   c.a.push(`<text x="${x+15}" y="121" font-family="Arial" font-size="18" fill="${navy}">q<tspan baseline-shift="sub" font-size="13">d</tspan></text>`);c.text(x+w-20,309,'T',18);c.text(x+35,309,'0',18);
   c.text(x+18,346,'Area = full monitoring cost.',20,blue);
  }
  return c.result();
 }
 function box(c,x,y,w,h,title,lines,color=navy){
  c.rect(x,y,w,h);c.text(x+18,y+35,title,23,color,'start',700);
  lines.forEach((t,i)=>c.text(x+18,y+77+36*i,t,20));
 }
 function cycle(step){
  const c=canvas('cycle');
  if(step<=1){
   c.text(32,28,'Each entry is one complete motion, sensing and learning prediction',21,navy,'start',700);
   c.text(38,96,'Policy vector',21,orange);c.text(275,96,'Field 1',20);c.text(425,96,'Field 2',20);c.text(585,96,'…',22);c.text(705,96,step===1?'Mean cost':'Field S',20);
   ['θ⁽¹⁾','θ⁽²⁾','⋮','θ⁽ᴷ⁾'].forEach((theta,i)=>{
    const y=122+i*66;c.rect(35,y,160,51,'#fff5df');c.text(115,y+33,theta,24,orange,'middle');
    [255,405,555,695].forEach((x,j)=>{c.rect(x,y,130,51,j===3&&step===1?'#e0eef9':'#f5f8fa');c.text(x+65,y+33,i===2?'⋮':j===2?'…':j===3&&step===1?'Ĵ(θ)':`J${j===3?'S':j+1}(θ)`,23,j===3&&step===1?blue:navy,'middle')});
    c.line(202,y+25,245,y+25,orange,2,true);
   });
   c.text(35,421,step===1?'Average within each policy row; then compare policy parameters.':'Rows: sampled policy parameter vectors. Columns: possible physical fields.',20);
  }else if(step===2){
   box(c,30,55,345,290,'Monitoring cost → weight',['Evaluate Ĵ for each θ.','Apply domain / validity checks.','Lower cost → larger weight.','Normalize: sum of weights = 1.'],blue);
   c.line(385,188,455,188,blue,3,true);
   box(c,468,55,375,290,'Weighted parameter mean',['Multiply each θ by its weight.','Sum in parameter space.','This gives θ̄ for the score.','Do not average executed paths.'],orange);
   c.text(30,405,'Costs come from predicted execution; the MBD decision remains θ.',22,navy,'start',700);
  }else if(step===3){
   box(c,30,75,245,245,'Current MBD state',['Noisy vector yᵢ','Noise schedule αᵢ, ᾱᵢ','Weighted mean θ̄']);
   c.line(285,194,320,194,blue,3,true);
   box(c,333,75,220,245,'Score estimate',['Combine yᵢ and θ̄.','Use the MBD','score formula.'],blue);
   c.line(564,194,600,194,blue,3,true);
   box(c,613,75,235,245,'Denoising update',['Apply score ascent.','Obtain yᵢ₋₁.','Robot has not moved.'],orange);
   c.text(30,395,'The next noisy vector is used to draw a new set of policy parameters.',21,navy,'start',700);
  }else if(step===4){
   box(c,279,32,335,115,'More reverse levels?',['Check the current level i.']);
   c.line(295,162,156,228,blue,3,true);c.line(597,162,735,228,green,3,true);
   c.text(150,191,'YES · i > 1',21,blue,'middle',700);c.text(741,191,'NO · i = 1',21,green,'middle',700);
   box(c,24,244,385,131,'Sample at level i − 1',['Return to policy evaluations.','No physical execution yet.'],blue);
   box(c,460,244,395,131,'Finish this planning call',['Select and check the best','evaluated policy parameter vector.'],green);
   c.text(440,419,'These are exclusive branches, not two actions to perform in sequence.',20,navy,'middle');
  }else{
   box(c,30,50,245,270,'Execute next interval',['Use the selected policy.','Apply the execution','controller on the robot.','Physical time advances.'],green);
   c.line(283,180,319,180,green,3,true);
   box(c,330,50,236,270,'Receive real data',['Measure the robot state.','Update clarity.','Assimilate observed wind.'],purple);
   c.line(575,180,610,180,purple,3,true);
   box(c,622,50,232,270,'New planning call',['Use updated x, q, Bᵈ.','Restart the MBD','optimization.'],blue);
   c.text(30,390,'Real observations replace hypothetical predictions at the next planning call.',21,navy,'start',700);
  }
  return c.result();
 }
 const comments={
  poam:[
   'The prior connects nearby wind values. The diamonds are representative inducing positions; they are not sensed data.',
   'The first observation is available at r = 2.2. Its purple error bar shows ±2 observation-noise standard deviations. The blue model has not yet been updated.',
   'Assimilate the observation. The mean moves toward the measured wind, and predictive uncertainty decreases at spatially correlated positions.',
   'At r = 3.0 there is no observation. The kernel and inducing posterior still provide a predicted mean and uncertainty there.',
   'The second observation is negative. It changes the local mean and reduces uncertainty around its location; the first observation remains assimilated.',
   'After three observations, the posterior retains uncertainty between observations. The short-scale region on the right needs more local evidence.'
  ],
  rollout:[
   'Start this predicted trajectory from the measured robot state, current clarity and current wind belief. Nothing has physically executed yet.',
   'Take one sampled policy parameter vector θ and one possible physical wind field d. The orange route is schematic nominal motion, not the trajectory assumed to execute.',
   'Use the wind belief to predict the execution controller’s response to the nominal request. It accounts for actuator limits and the value-rate constraint tests.',
   'Propagate the selected input under the same possible field. The reached state determines sensing and clarity. Recompute the controller at every prediction step.',
   'At a scheduled sensing time, generate a noisy wind observation at the predicted reached position, using the same physical field as this trajectory.',
   'Assimilate that predicted observation with POAM. Carry forward the reached state and clarity; do not restart the robot trajectory.',
   'Keep θ fixed. The updated belief changes the nominal request and execution controller. The frozen-belief continuation is shown only for comparison.',
   'Integrate the full learning-updated monitoring cost, including motion before learning. Its matched fixed-belief cost gives the signed learning-value diagnostic. MBD uses the learning-updated cost directly.'
  ],
  cycle:[
   'Repeat the preceding prediction for each sampled policy parameter vector and each possible field, including the scheduled noisy observations.',
   'Average the full learning-updated monitoring costs. This equals fixed-belief cost minus signed learning value. Do not subtract another learning reward.',
   'Apply feasibility checks, normalize the cost weights, and form a weighted mean of policy parameter vectors. The executed trajectories supply the costs.',
   'Use the same score estimator and denoising update as MBD. Only the model-based evaluation of the monitoring objective has changed.',
   'If reverse levels remain, draw new parameters from the updated noisy vector. Only after the final level can this planning call proceed to execution.',
   'Execute the selected policy for the next interval, assimilate actual observations, and begin a new MBD planning call from the newly measured state.'
  ]};
 const statuses={
  poam:['Fixed representation · spatial prior','Observation available · before assimilation','One observation assimilated','Query the model between observations','Two observations assimilated','Three observations assimilated'],
  rollout:['Inside the planner · one predicted trajectory','Policy and possible field selected','Wind awareness: predict execution','Wind awareness: predict delivered monitoring','Anticipated learning: predict a measurement','Anticipated learning: update the belief','Anticipated learning: continue from the reached state','One complete monitoring-cost evaluation'],
  cycle:['Within one MBD reverse level','Within one MBD reverse level','Within one MBD reverse level','Within one MBD reverse level','End of reverse level: choose the appropriate branch','After the final reverse level · real execution']};
 function setup(){
  if(window.walkPlayers)return;
  window.walkPlayers=[];
  document.querySelectorAll('.walk-player').forEach(root=>{
   const kind=root.dataset.walk,max=comments[kind].length-1,range=root.querySelector('input'),play=root.querySelector('[data-action="play"]');
   const api={root,kind,index:(location.search.includes('print-pdf')||location.search.includes('snapshot'))?Number(root.dataset.printStep):0,playing:false,timer:null};
   function pause(){clearTimeout(api.timer);api.timer=null;api.playing=false;play.textContent='Play';play.setAttribute('aria-pressed','false')}
   function render(){
    root.dataset.step=api.index;range.value=api.index;
    root.querySelector('.walk-stage').innerHTML=({poam,rollout,cycle})[kind](api.index);
    root.querySelector('.walk-status').textContent=statuses[kind][api.index];
    root.querySelector('.walk-explanation').textContent=comments[kind][api.index];
    root.querySelector('output').textContent=`Step ${api.index+1} / ${max+1}`;
    root.querySelectorAll('.walk-step').forEach((el,i)=>{el.classList.toggle('active',i===api.index);el.classList.toggle('done',i<api.index);if(i===api.index)el.setAttribute('aria-current','step');else el.removeAttribute('aria-current')});
    root.querySelectorAll('.walk-equation').forEach((el,i)=>el.hidden=i!==api.index);
    root.querySelector('[data-action="prev"]').disabled=api.index===0;
    root.querySelector('[data-action="next"]').disabled=api.index===max;
   }
   function seek(i){pause();api.index=Math.max(0,Math.min(max,i));render()}
   function tick(){if(!api.playing)return;if(api.index>=max){pause();return}api.index++;render();if(api.index===max)pause();else api.timer=setTimeout(tick,2500)}
   root.querySelectorAll('button').forEach(btn=>btn.addEventListener('click',event=>{
    event.stopPropagation();const action=btn.dataset.action;
    if(action==='reset')seek(0);if(action==='prev')seek(api.index-1);if(action==='next')seek(api.index+1);
    if(action==='play'){if(api.playing)pause();else{if(api.index===max){api.index=0;render()}api.playing=true;play.textContent='Pause';play.setAttribute('aria-pressed','true');api.timer=setTimeout(tick,2500)}}
   }));
   range.addEventListener('input',()=>seek(Number(range.value)));
   root.querySelectorAll('button,input').forEach(el=>el.addEventListener('keydown',event=>event.stopPropagation()));
   Reveal.on('slidechanged',()=>{if(Reveal.getCurrentSlide()!==root.closest('section'))pause()});
   document.addEventListener('visibilitychange',()=>{if(document.hidden)pause()});
   Object.assign(api,{seek,pause,render});window.walkPlayers.push(api);render();
  });
 }
 if(window.Reveal?.isReady())setup();else Reveal.on('ready',setup);
})();
