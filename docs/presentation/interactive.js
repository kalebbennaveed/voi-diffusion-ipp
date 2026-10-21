/* Manually controlled playback of saved numerical MBD evaluations. */
(function(){
 'use strict';
 const N='#00274c',B='#2474bf',O='#d77717',G='#008653',GRAY='#aab7c0';
 const fmt=x=>Number(x).toFixed(3);
 const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
 function denoising(h,step,iteration){
  const d=window.mbdTeachingData,ab=d.alpha_bar[h.level-1],prev=h.level===1?1:d.alpha_bar[h.level-2];
  const alpha=ab/prev,corrected=h.noisy+(1-ab)*h.score,nextStd=Math.sqrt((1-prev)/prev);
  const next=d.history[iteration+1],v=[];
  const txt=(x,y,s,size=18,c=N,anchor='start',weight=400)=>v.push(`<text x="${x}" y="${y}" font-family="Arial,Helvetica,sans-serif" font-size="${size}" fill="${c}" text-anchor="${anchor}" font-weight="${weight}">${esc(s)}</text>`);
  const line=(x,y,xx,yy,c=N,w=2,extra='')=>v.push(`<line x1="${x}" y1="${y}" x2="${xx}" y2="${yy}" stroke="${c}" stroke-width="${w}" ${extra}/>`);
  const dot=(x,y,c,r=6)=>v.push(`<circle cx="${x}" cy="${y}" r="${r}" fill="${c}"/>`);
  v.push('<defs><clipPath id="denoise-path-clip"><rect x="18" y="78" width="390" height="265"/></clipPath><clipPath id="denoise-density-clip"><rect x="455" y="100" width="370" height="225"/></clipPath><marker id="denoise-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#d77717"/></marker></defs>');
  const titles=['8a · Apply the score correction','8b · Rescale to the next level','8c · Recenter and narrow sampling','8d · Draw a new set of paths'];
  txt(20,27,h.level===1&&step>=9?'Final level · no further sampling':titles[step-7],24,N,'start',700);
  if(step<10){
   txt(20,59,'Noisy optimizer coordinate y',19,N,'start',700);
   const lo=Math.min(h.noisy,corrected,h.next_noisy)-.08,hi=Math.max(h.noisy,corrected,h.next_noisy)+.08;
   const xx=a=>45+330*(a-lo)/(hi-lo);
   const row=(y,val,label,c)=>{txt(24,y-26,label,19,c);line(40,y,386,y,'#c2ced7',2);dot(xx(val),y,c);};
   row(128,h.noisy,`Start: yᵢ = ${fmt(h.noisy)}`,N);
   row(221,corrected,`After score correction: ${fmt(corrected)}`,O);
   dot(xx(h.noisy),221,GRAY,4);
   line(xx(h.noisy),221,xx(corrected),221,O,3,'marker-end="url(#denoise-arrow)"');
   txt(22,167,`Add ${(1-ab).toFixed(3)} × ${fmt(h.score)}`,17,O);
   if(step>=8){
    row(315,h.next_noisy,`Rescaled: yᵢ₋₁ = ${fmt(h.next_noisy)}`,G);
    dot(xx(corrected),315,GRAY,4);
    line(xx(corrected),315,xx(h.next_noisy),315,O,3,'marker-end="url(#denoise-arrow)"');
    txt(22,261,`Divide by √αᵢ = ${fmt(Math.sqrt(alpha))}`,17,G);
   }else txt(24,313,'Next: divide by √αᵢ to change level.',18,GRAY);
   txt(213,361,'All three rows use the same y scale.',17,N,'middle');
   txt(213,403,'This changes the optimizer, not the robot.',18,N,'middle');
  }else{
   txt(20,59,next?'New proposals, before cost evaluation':'Endpoint and best feasible path',19,N,'start',700);
   const px=x=>212+x*65,py=y=>211-y*65;
   for(const a of [-2,-1,0,1,2]){line(px(a),80,px(a),340,'#e9eef2',1);line(25,py(a),405,py(a),'#e9eef2',1);}
   dot(212,211,GRAY,.5*65);v.push(`<circle cx="212" cy="211" r="39" fill="none" stroke="${N}" stroke-dasharray="4 4"/>`);
   const path=(a,c,opacity,w,dash=false)=>{const p=Array.from({length:101},(_,j)=>{const s=j/100;return `${j?'L':'M'}${px(-2+4*s)},${py(4*a*s*(1-s))}`}).join(' ');v.push(`<path d="${p}" stroke="${c}" opacity="${opacity}" stroke-width="${w}" fill="none" ${dash?'stroke-dasharray="5 4"':''}/>`)};
   v.push('<g clip-path="url(#denoise-path-clip)">');
   h.samples.forEach(a=>path(a,GRAY,.21,1));
   if(next)next.samples.forEach(a=>path(a,B,.32,1.7));
   else path(h.best[0],G,1,3);
   path(h.weighted_mean,O,1,3,true);v.push('</g>');dot(px(-2),py(0),N);dot(px(2),py(0),N);
   txt(212,363,next?'Gray: previous paths · Blue: new paths':'Green: best evaluated feasible path',17,N,'middle');
   txt(212,402,'Orange dashed: weighted mean path',18,O,'middle');
  }
  line(430,48,430,414,'#dce5ec',1);
  txt(455,59,'Clean path-height coordinate a',19,N,'start',700);
  txt(455,86,'Gaussian density · same vertical scale',17,N);
  const xx=a=>640+a*82,base=319,peak=nextStd>0?nextStd:h.proposal_std;
  [-2,-1,0,1,2].forEach(a=>{line(xx(a),108,xx(a),base,'#e9eef2',1);txt(xx(a),341,a,16,N,'middle')});
  const curve=(mean,std,c,dashed)=>{const p=Array.from({length:301},(_,j)=>{const a=-2.25+4.5*j/300;return `${j?'L':'M'}${xx(a)},${base-185*peak/std*Math.exp(-.5*((a-mean)/std)**2)}`}).join(' ');v.push(`<path d="${p}" fill="none" stroke="${c}" stroke-width="3" ${dashed?'stroke-dasharray="7 5"':''}/>`);};
  v.push('<g clip-path="url(#denoise-density-clip)">');curve(h.proposal_mean,h.proposal_std,GRAY,true);
  if(step>=9&&nextStd>0)curve(h.weighted_mean,nextStd,B,false);
  line(xx(h.weighted_mean),112,xx(h.weighted_mean),base,O,2,'stroke-dasharray="5 4"');
  if(step===10&&next)next.samples.forEach((a,k)=>dot(xx(a),315-(k%3)*6,B,2.5));
  if(step>=9&&nextStd===0)dot(xx(h.weighted_mean),base,O,7);
  v.push('</g>');
  txt(455,369,`Before: mean ${fmt(h.proposal_mean)} · std. ${fmt(h.proposal_std)}`,17,GRAY);
  txt(455,394,step>=9?(next?`Next: mean ${fmt(h.weighted_mean)} · std. ${fmt(nextStd)}`:`Endpoint a = ${fmt(h.weighted_mean)}; no next Gaussian`):`Orange line: weighted mean ā = ${fmt(h.weighted_mean)}`,17,step>=9?B:O);
  txt(455,419,'View: −2.2 to 2.2; Gaussian tails continue.',15,N);
  return v.join('');
 }
 function diagram(h,step,iteration){
  if(step>=7)return denoising(h,step,iteration);
  let v=[];
  const text=(x,y,s,size=18,color=N,anchor='start',weight=400)=>v.push(`<text x="${x}" y="${y}" font-family="Arial,Helvetica,sans-serif" font-size="${size}" fill="${color}" text-anchor="${anchor}" font-weight="${weight}">${esc(s)}</text>`);
  const line=(x1,y1,x2,y2,color=N,width=1,extra='')=>v.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${extra}/>`);
  const circle=(x,y,r,c,extra='')=>v.push(`<circle cx="${x}" cy="${y}" r="${r}" fill="${c}" ${extra}/>`);
  v.push('<defs><clipPath id="world-clip"><rect x="28" y="62" width="392" height="278"/></clipPath><clipPath id="parameter-clip"><rect x="482" y="62" width="345" height="278"/></clipPath><marker id="mbd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#d77717"/></marker></defs>');
  text(222,26,'Path model',23,N,'middle',700);text(655,26,step>=6?'Denoising calculation':'Parameter space',23,N,'middle',700);
  const wx=x=>224+x*80,wy=y=>205-y*65;
  for(const x of [-2,-1,0,1,2])line(wx(x),65,wx(x),337,'#e6edf2');
  for(const y of [-2,-1,0,1,2])line(28,wy(y),420,wy(y),'#e6edf2');
  // Equal x/y units: the environment uses one isotropic spatial scale.
  const sy=65, sx=65, cx=224, cy=205, px=x=>cx+x*sx, py=y=>cy-y*sy;
  v=v.filter(s=>!s.startsWith('<line'));
  for(const x of [-2,-1,0,1,2])line(px(x),65,px(x),337,'#e6edf2');
  for(const y of [-2,-1,0,1,2])line(28,py(y),420,py(y),'#e6edf2');
  circle(cx,cy,.5*sx,'#aab7c0');circle(cx,cy,.6*sx,'none','stroke="#00274c" stroke-dasharray="3 4"');
  const route=(a,color,width,opacity=1,dashed=false)=>{
   const points=Array.from({length:101},(_,j)=>{const s=j/100;return `${j?'L':'M'}${px(-2+4*s).toFixed(2)},${py(4*a*s*(1-s)).toFixed(2)}`;}).join(' ');
   v.push(`<path d="${points}" fill="none" stroke="${color}" stroke-width="${width}" opacity="${opacity}" ${dashed?'stroke-dasharray="7 5"':''}/>`);
  };
  v.push('<g clip-path="url(#world-clip)">');
  if(step>=2)h.samples.forEach((a,k)=>{
   const valid=h.valid[k], w=h.weights[k];
   if(Math.abs(a)>2.1)return;
   route(a,step<3?B:(valid?B:GRAY),step>=4&&valid?1+w*42:1.2,step<3?.27:(valid?(step>=4?.12+Math.min(.8,w*8):.45):.16));
  });
  if(step>=5){route(h.weighted_mean,O,3,1,true);route(h.best[0],G,3);}
  v.push('</g>');circle(px(-2),py(0),6,N);circle(px(2),py(0),6,N);
  text(px(-2)-13,py(0)+29,'Start',18,N,'end');text(px(2)+13,py(0)+29,'Goal');
  text(224,369,step<2?'Paths appear after the model step.':'Each path is defined by one sampled height.',18,N,'middle');
  text(224,402,step>=5?'Orange: mean · Green: best feasible':'Required center clearance: 0.6',18,step>=5?O:N,'middle');
  if(step<6){
   const xx=a=>655+a*75, yy=c=>333-(c-3.8)*104;
   v.push(`<rect x="${xx(-.6)}" y="63" width="90" height="270" fill="#eff1f3"/>`);
   for(const a of [-2,-1,0,1,2]){line(xx(a),64,xx(a),333,'#e6edf2');text(xx(a),355,a,16,N,'middle');}
   for(const c of [4,5,6]){line(482,yy(c),827,yy(c),'#e6edf2');text(476,yy(c)+5,c,16,N,'end');}
   text(655,382,'Path-height parameter a',18,N,'middle');text(485,55,step>=3?'Path length':'Gaussian proposal (scaled)',17);
   v.push('<g clip-path="url(#parameter-clip)">');
   if(step<3){
    const points=Array.from({length:201},(_,j)=>{const a=-2.3+4.6*j/200;return `${j?'L':'M'}${xx(a)},${320-230*Math.exp(-.5*((a-h.proposal_mean)/h.proposal_std)**2)}`;}).join(' ');
    v.push(`<path d="${points}" stroke="${B}" stroke-width="2.5" fill="none"/>`);
   }
   if(step>=1)h.samples.forEach((a,k)=>circle(xx(a),step<3?320:yy(h.costs[k]),step>=4?2.5+Math.sqrt(h.weights[k])*15:3,step<3||h.valid[k]?B:GRAY,'opacity="0.7"'));
   if(step>=5)line(xx(h.weighted_mean),65,xx(h.weighted_mean),333,O,3,'stroke-dasharray="7 5"');
   v.push('</g>');
   text(655,407,step===0?`Mean ${fmt(h.proposal_mean)} · std. dev. ${fmt(h.proposal_std)}`:step<4?'Only heights within the displayed domain are drawn.':step===4?'Marker area increases with weight.':`Weighted mean = ${fmt(h.weighted_mean)}`,17,step>=5?O:N,'middle');
  }else{
   const d=window.mbdTeachingData,ab=d.alpha_bar[h.level-1],alpha=ab/(h.level===1?1:d.alpha_bar[h.level-2]);
   text(486,78,`Current noisy value  yᵢ = ${fmt(h.noisy)}`,21,N);
   text(486,121,`Weighted mean  ā = ${fmt(h.weighted_mean)}`,21,O);
   text(486,164,`Score estimate  ŝᵢ = ${fmt(h.score)}`,21,B);
   text(486,207,`ᾱᵢ = ${fmt(ab)}    αᵢ = ${fmt(alpha)}`,20,N);
   const from=h.noisy,to=h.next_noisy,lo=Math.min(from,to)-.04,hi=Math.max(from,to)+.04;
   const xx=x=>500+(x-lo)/(hi-lo)*300;
   line(492,285,810,285,'#bbcbd7',2);circle(xx(from),285,6,N);
   text(xx(from),263,'yᵢ',20,N,'middle');
   if(step===7){
    line(xx(from),285,xx(to),285,O,4,'marker-end="url(#mbd-arrow)"');
    circle(xx(to),285,6,O);text(xx(to),322,'yᵢ₋₁',20,O,'middle');
    text(655,370,`Next noisy value = ${fmt(to)}`,23,O,'middle',700);
    text(655,407,h.level===1?'End of reverse levels.':'Continue sampling at the next lower-noise level.',17,N,'middle');
   }else{text(655,354,'The score determines the next update.',19,B,'middle');text(655,395,'Next step applies the denoising equation.',18,N,'middle');}
  }
  return v.join('');
 }

 function setup(){
  if(window.mbdPlayers)return;
  window.mbdPlayers=[];
  document.querySelectorAll('.mbd-player').forEach(root=>{
   const detail=root.dataset.player==='detail';
   const range=root.querySelector('input[type="range"]'),out=root.querySelector('output');
   const play=root.querySelector('[data-action="play"]');
   const select=root.querySelector('select');
   const api={index:detail&&location.search.includes('print-pdf')?9:0,iteration:detail?Number(select.value):0,playing:false,timer:null};
   const max=Number(range.max);
   function pause(){clearTimeout(api.timer);api.timer=null;api.playing=false;play.textContent='Play';play.setAttribute('aria-pressed','false');}
   function render(){
    range.value=api.index;root.dataset.frame=api.index;root.dataset.iteration=api.iteration;
    root.querySelector('[data-action="prev"]').disabled=api.index===0;
    root.querySelector('[data-action="next"]').disabled=api.index===max;
    if(!detail){
     root.querySelectorAll('.mbd-frame').forEach((e,i)=>e.hidden=i!==api.index);
     out.textContent=`Iteration ${Math.floor(api.index/3)+1}/8 · phase ${api.index%3+1}/3`;
    }else{
     const h=window.mbdTeachingData.history[api.iteration];
     const algorithmStep=Math.min(api.index,7);
     root.querySelectorAll('.algorithm-step').forEach((e,i)=>{e.classList.toggle('active',i===algorithmStep);e.classList.toggle('done',i<algorithmStep);if(i===algorithmStep)e.setAttribute('aria-current','step');else e.removeAttribute('aria-current');});
     root.querySelector('[data-action="continue"]').disabled=api.index<max||h.level===1;
     root.querySelectorAll('.mbd-equation').forEach((e,i)=>e.hidden=i!==api.index);
     root.querySelector('.mbd-stage').innerHTML=diagram(h,api.index,api.iteration);
     const valid=h.valid.filter(Boolean).length;
     const comments=[
      `Start from yᵢ = ${fmt(h.noisy)} at diffusion level ${h.level}. This is the optimizer's current noisy decision, not the robot state.`,
      `Draw 64 path-height values. The proposal has mean ${fmt(h.proposal_mean)} and standard deviation ${fmt(h.proposal_std)}.`,
      'Insert each sampled height into the path equation. No cost weights have been assigned yet.',
      `${valid} of 64 paths satisfy the parameter bounds and clearance requirement. Infeasible paths receive zero weight.`,
      'Use temperature λ = 0.12. Feasible shorter paths receive larger weights; all 64 weights sum to one.',
      `The weighted height is ā = ${fmt(h.weighted_mean)}. This mean is separate from the best evaluated feasible path and need not be feasible.`,
      `The score estimate is ${fmt(h.score)}. It combines the current noisy value and the cost-weighted parameter mean.`,
      '8a. Add the score correction. It places the intermediate noisy value at the current noise-scaled weighted mean. The sampling distribution has not changed yet.',
      '8b. Divide by √αᵢ. This expresses the weighted mean at the next noise level, giving the next noisy optimizer value yᵢ₋₁.',
      h.level===1?'8c. The final noise level has been removed: y₀ = ā. There is no next Gaussian proposal. Feasibility of the mean still needs checking.':'8c. The next Gaussian is centered at ā and has smaller standard deviation. Gray dashed is the old proposal; blue is the next proposal.',
      h.level===1?'8d. Reverse iterations are complete. The weighted mean and best evaluated feasible path remain separate; no new samples are drawn.':'8d. Draw fresh heights from the blue Gaussian. The new blue paths have not been cost-weighted yet. Use Continue to next iteration to evaluate them.'
     ];
     root.querySelector('.mbd-explanation').textContent=comments[api.index];
     out.textContent=`Iteration ${api.iteration+1}/8 · ${api.index<7?`step ${api.index+1}/8`:`step 8${'abcd'[api.index-7]}`}`;
    }
   }
   function seek(i){pause();api.index=Math.max(0,Math.min(max,i));render();}
   function tick(){
    if(!api.playing)return;
    if(api.index>=max){pause();return;}
    api.index++;render();
    if(api.index===max)pause();else api.timer=setTimeout(tick,detail?2200:1500);
   }
   root.querySelectorAll('button').forEach(btn=>btn.addEventListener('click',e=>{
    e.stopPropagation();
    const action=btn.dataset.action;
    if(action==='reset')seek(0);
    if(action==='prev')seek(api.index-1);
    if(action==='next')seek(api.index+1);
    if(action==='continue'&&api.index===max&&api.iteration<7){api.iteration++;select.value=String(api.iteration);seek(0);}
    if(action==='play'){
     if(api.playing)pause();else{if(api.index===max){api.index=0;render();}api.playing=true;play.textContent='Pause';play.setAttribute('aria-pressed','true');api.timer=setTimeout(tick,detail?2200:1500);}
    }
   }));
   range.addEventListener('input',()=>seek(Number(range.value)));
   if(select)select.addEventListener('change',()=>{api.iteration=Number(select.value);seek(0);});
   root.querySelectorAll('button,input,select').forEach(el=>el.addEventListener('keydown',e=>e.stopPropagation()));
   Reveal.on('slidechanged',()=>{if(Reveal.getCurrentSlide()!==root.closest('section'))pause();});
   document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
   api.seek=seek;api.pause=pause;api.root=root;api.render=render;window.mbdPlayers.push(api);render();
  });
 }
 if(window.Reveal?.isReady())setup();else Reveal.on('ready',setup);
})();
