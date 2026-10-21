/* Presentation-only arithmetic and walkthrough. Does not run the monitoring planner. */
(function(){
 const NS='http://www.w3.org/2000/svg', navy='#00274c',blue='#2474bf',green='#008653',orange='#b77316',grey='#607486';
 const rows=[['Established route',[9,10,10,11]],['Uncertain alternative',[6,8,14,20]],['Poor alternative',[11,13,16,20]],['Known costly route',[12,12,12,12]]];
 const explanations=[
 'Begin one planning call with the measured state, current clarity and current disturbance posterior. The posterior stays fixed until physical execution.',
 'Evaluate each nominal trajectory through the state-dependent filter in every coherent performance field. Predict the applied inputs, motion, sensing, and constraint penalties. Filter slack does not automatically reject a trajectory.',
 'Here η = 0.5 retains the two smallest monitoring costs. Add the average penalty from all four fields. The uncertain alternative has monitoring score 7 plus mean penalty 1, giving total score 8. These are illustrative values.',
 'The established route is the average-cost incumbent. The uncertain alternative has priority 9.5 − 8 = 1.5 and a positive difference between average and favorable monitoring costs. Additional sampling is centered around that evaluated parameter vector.',
 'Draw a fresh parameter batch from the mixture. Correct each weight by ordinary-proposal density divided by mixture density, then use the weighted parameter mean in the original MBD score update.',
 'Repeat sampling and denoising while diffusion levels remain. After the last level, reevaluate final choices in fresh shared fields, including average constraint penalties. Select the lowest-scoring finite, in-domain vector. The final weighted vector is evaluated too.',
 'Execute only the next replanning interval. Real wind measurements update POAM. Begin the next planning call from the attained state, clarity and updated posterior.'
 ];
 const equations=[
 String.raw`x_k,\ q_k,\ \mathcal B_k^d,\qquad \mathcal B^d(\tau)=\mathcal B_k^d`,
 String.raw`u^{[s]}=\Gamma_k(x^{[s]},u_{\mathrm{nom}}^{[s]}),\quad J_s=\int_0^TD(q^{[s]}(\tau))\,d\tau`,
 String.raw`F_k(\theta)=\tfrac12\big(J_{(1)}(\theta)+J_{(2)}(\theta)\big)+\overline P(\theta)`,
 String.raw`q_i^{\mathrm{prop}}=(1-\rho)\phi_i+\rho\psi_i`,
 String.raw`\widetilde w_n\propto\mathbf1_\Theta\chi_k\,e^{-F_k/\lambda}\,\phi_i/q_i^{\mathrm{prop}},\quad y_{i-1}=\sqrt{\bar\alpha_{i-1}}\bar\theta`,
 String.raw`\theta_k^\star\in\arg\min_{\theta\in\mathcal C_k}\{J_\eta^{\mathrm{val}}(\theta)+\overline P^{\mathrm{val}}(\theta)\}`,
 String.raw`\mathcal B_{k+1}^d=\operatorname{POAMUpdate}(\mathcal B_k^d,\mathcal D_k^{d,\mathrm{real}})`
 ];
 function draw(step){
  const a=[];const t=(x,y,s,size=23,color=navy,weight=400)=>a.push(`<text x="${x}" y="${y}" font-family="Arial" font-size="${size}" fill="${color}" font-weight="${weight}">${s}</text>`);
  const line=(x,y,xx,yy,color=navy,width=2)=>a.push(`<line x1="${x}" y1="${y}" x2="${xx}" y2="${yy}" stroke="${color}" stroke-width="${width}"/>`);
  const box=(x,y,w,h,label,sub,color=blue)=>{a.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="#f4f8fb" stroke="${color}" stroke-width="2"/>`);t(x+16,y+34,label,23,color,700);sub.forEach((v,i)=>t(x+16,y+70+31*i,v,21));};
  const arrow=(x,y,xx,yy,color=navy)=>a.push(`<path d="M${x},${y} L${xx},${yy}" stroke="${color}" stroke-width="2.5" fill="none" marker-end="url(#opt-arrow)"/>`);
  if(step===0){
   t(30,35,'One physical replanning time',28,navy,700);
   box(30,75,340,190,'Measured inputs',['Robot state and clarity','Disturbance posterior','Geometry and clarity targets']);
   arrow(375,170,445,170);
   box(450,75,370,190,'One planning call',['Draw fixed spatial fields','Evaluate nominal policies','Keep the posterior fixed']);
   t(30,330,'The robot has not executed the new plan yet.',24,green,700);
  }else if(step===1){
   t(30,35,'Same policy, different possible execution',28,navy,700);
   box(30,90,215,185,'One policy',['Policy parameters','Same initial state','Same initial clarity']);
   for(let i=0;i<3;i++){let y=80+98*i;arrow(248,175,315,y+35);box(320,y,480,80,`Performance field ${i+1}`,['State-dependent filter, motion and sensing']);}
   t(30,402,'Include the mean constraint penalty from every field.',23,green,700);
  }else if(step===2||step===3){
   t(25,33,step===2?'Illustrative monitoring costs':'Illustrative archive priorities',28,navy,700);
   const headers=step===2?['Trajectory','Monitoring costs','Penalty','Score']:['Trajectory','Mean + penalty','Score','Priority'];
   const xs=step===2?[25,345,610,705]:[25,380,520,705];
   headers.forEach((v,i)=>t(xs[i],85,v,21,grey,700));line(25,101,825,101,'#aebfcb');
   rows.forEach((r,i)=>{let y=145+62*i;const mean=r[1].reduce((a,b)=>a+b)/4, penalty=i===1?1:0, opt=(r[1][0]+r[1][1])/2,score=opt+penalty,priority=Math.max(9.5-score,0)*(mean>opt);if(i===1)a.push(`<rect x="15" y="${y-29}" width="820" height="51" fill="#e6f3ec"/>`);t(25,y,r[0],22,i===1?green:navy,i===1?700:400);if(step===2){t(345,y,r[1].join(', '),24);t(610,y,String(penalty),24);t(715,y,String(score),24,i===1?green:navy,700);}else{t(380,y,String(mean+penalty),24);t(540,y,String(score),24);t(725,y,String(priority),24,i===1?green:navy,700);}});
   t(25,405,'Illustrative arithmetic. Finite, in-domain evaluations.',20,grey);
  }else if(step===4){
   t(25,35,'One diffusion level',28,navy,700);
   box(25,80,235,170,'Fresh samples',['Ordinary Gaussian','plus archive mixture']);
   arrow(264,165,300,165);
   box(305,80,250,170,'Correct the weights',['Optimistic target weight','× density ratio']);
   arrow(559,165,595,165);
   box(600,80,225,170,'Denoise',['Weighted mean','Original score update']);
   t(25,315,'Sampling preference and target preference are distinct.',24,green,700);
   t(25,358,'Archive entries do not count as newly drawn samples.',23);
  }else if(step===5){
   t(25,35,'After a score update',28,navy,700);
   box(25,80,305,150,'More levels remain',['Return to mixture sampling','Then evaluate a fresh batch']);
   box(450,80,365,150,'Last level finished',['Fresh-field reevaluation','Compare complete scores']);
   t(25,280,'These are alternative branches, determined by the level.',23,green,700);
   t(25,330,'No finite choice: record failure and use the declared fallback.',22);
   t(25,375,'A weighted parameter mean still needs its own rollout check.',22);
  }else{
   t(25,35,'Physical execution and online learning',28,navy,700);
   box(25,90,240,170,'Execute an interval',['Actual execution filter','Actual disturbance']);
   arrow(269,175,310,175);
   box(315,90,245,170,'Measure and update',['Real wind observations','POAM posterior update']);
   arrow(564,175,605,175);
   box(610,90,215,170,'Replan',['Attained state','Attained clarity']);
   t(25,330,'Only real measurements update the disturbance belief.',25,green,700);
  }
  return `<svg xmlns="${NS}" viewBox="0 0 860 430" role="img" aria-label="${explanations[step]}"><defs><marker id="opt-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="${navy}"/></marker></defs>${a.join('')}</svg>`;
 }
 function setup(){const root=document.getElementById('optimistic-walkthrough-player');if(!root)return;
  const range=root.querySelector('input'),play=root.querySelector('[data-action="play"]');
  const api={index:location.search.includes('snapshot')||location.search.includes('print-pdf')?2:0,playing:false,timer:null};
  function pause(){clearTimeout(api.timer);api.playing=false;play.textContent='Play';play.setAttribute('aria-pressed','false');}
  function render(){root.dataset.step=api.index;root.querySelector('.optimistic-visual').innerHTML=draw(api.index);root.querySelector('.optimistic-explanation').textContent=explanations[api.index];katex.render(equations[api.index],root.querySelector('.optimistic-equation'),{displayMode:true,throwOnError:true});root.querySelectorAll('.optimistic-steps li').forEach((el,i)=>{el.classList.toggle('active',i===api.index);if(i===api.index)el.setAttribute('aria-current','step');else el.removeAttribute('aria-current');});range.value=api.index;root.querySelector('output').textContent=`Step ${api.index+1} / 7`;root.querySelector('[data-action="prev"]').disabled=api.index===0;root.querySelector('[data-action="next"]').disabled=api.index===6;}
  function seek(n){pause();api.index=Math.min(6,Math.max(0,n));render();}
  function tick(){if(!api.playing)return;if(api.index===6){pause();return;}api.index++;render();if(api.index===6)pause();else api.timer=setTimeout(tick,3500);}
  root.querySelectorAll('button').forEach(b=>b.addEventListener('click',e=>{e.stopPropagation();const act=b.dataset.action;if(act==='reset')seek(0);if(act==='prev')seek(api.index-1);if(act==='next')seek(api.index+1);if(act==='play'){if(api.playing)pause();else{if(api.index===6){api.index=0;render();}api.playing=true;play.textContent='Pause';play.setAttribute('aria-pressed','true');api.timer=setTimeout(tick,3500);}}}));
  range.addEventListener('input',()=>seek(Number(range.value)));root.querySelectorAll('button,input').forEach(e=>e.addEventListener('keydown',ev=>ev.stopPropagation()));
  Reveal.on('slidechanged',()=>{if(Reveal.getCurrentSlide()!==root.closest('section'))pause();});document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  Object.assign(api,{seek,pause,render});window.optimisticPlayer=api;render();
 }
 if(window.Reveal?.isReady())setup();else Reveal.on('ready',setup);
})();
