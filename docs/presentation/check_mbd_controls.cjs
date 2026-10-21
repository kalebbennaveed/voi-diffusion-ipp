const {chromium}=require('/Users/kalebbennaveed/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const path=require('path'),fs=require('fs'),assert=require('assert');
(async()=>{
 const portable=process.argv.includes('--portable');
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',args:['--allow-file-access-from-files']});
 const page=await browser.newPage({viewport:{width:1600,height:900}});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto('file://'+path.resolve('output/presentation/wind_aware_mbd/'+(portable?'slides-standalone.html':'slides.html')),{waitUntil:'load'});
 await page.waitForFunction(()=>window.mbdPlayers?.length===2&&Reveal.isReady());
 await page.evaluate(()=>document.fonts.ready);
 const state=i=>page.evaluate(i=>{const p=mbdPlayers[i];return {frame:p.index,playing:p.playing,iteration:p.iteration};},i);
 const goto=async i=>{await page.evaluate(i=>Reveal.slide([...document.querySelectorAll('.reveal .slides>section')].indexOf(mbdPlayers[i].root.closest('section'))),i);await page.waitForTimeout(150);};
 const players=[page.locator('[data-player="overview"]'),page.locator('[data-player="detail"]')];
 assert.deepStrictEqual(await state(0),{frame:0,playing:false,iteration:0});assert.deepStrictEqual(await state(1),{frame:0,playing:false,iteration:6});
 await goto(0);
 await page.waitForTimeout(1800);assert.equal((await state(0)).frame,0,'must start paused');
 await players[0].locator('[data-action="next"]').click();assert.equal((await state(0)).frame,1);
 await players[0].locator('[data-action="prev"]').click();assert.equal((await state(0)).frame,0);
 await players[0].locator('input').fill('20');assert.equal((await state(0)).frame,20);
 await players[0].locator('[data-action="play"]').click();await page.waitForTimeout(1700);assert.equal((await state(0)).frame,21);
 await players[0].locator('[data-action="play"]').click();await page.waitForTimeout(1700);assert.equal((await state(0)).frame,21);assert.equal((await state(0)).playing,false);
 await players[0].locator('input').focus();const slide=await page.evaluate(()=>Reveal.getIndices().h);await page.keyboard.press('ArrowLeft');assert.equal(await page.evaluate(()=>Reveal.getIndices().h),slide,'range key must not navigate slide');
 await players[0].locator('[data-action="reset"]').click();assert.equal((await state(0)).frame,0);
 await players[0].locator('[data-action="play"]').click();await goto(1);assert.equal((await state(0)).playing,false,'leaving slide pauses player');
 await players[1].locator('select').selectOption('0');assert.equal((await state(1)).iteration,0);
 fs.mkdirSync('tmp/slides/qa/mbd',{recursive:true});
 const checks=[];
 for(const iteration of [0,6,7]){
  await players[1].locator('select').selectOption(String(iteration));
  for(let step=0;step<11;step++){
   if(step)await players[1].locator('[data-action="next"]').click();
   const s=await state(1);assert.equal(s.frame,step);assert.equal(await players[1].locator('.algorithm-step.active').getAttribute('data-step'),String(Math.min(step,7)));
   assert.equal(await players[1].locator('.mbd-equation:not([hidden])').count(),1);
   checks.push(await page.evaluate(()=>{const r=document.querySelector('[data-player="detail"]'),s=r.closest('section').getBoundingClientRect();return [...r.querySelectorAll('.mbd-controls,.mbd-equation:not([hidden]),.mbd-explanation,.algorithm-step,svg text')].filter(e=>{const b=e.getBoundingClientRect();return b.right>s.right+3||b.left<s.left-3||b.bottom>s.bottom+3}).map(e=>e.textContent);}));
   if(!portable)await page.screenshot({path:`tmp/slides/qa/mbd/iteration-${iteration+1}-step-${step+1}.png`});
  }
 }
 assert(checks.every(x=>!x.length),'interactive content must fit slide');
 assert(await page.evaluate(()=>{const p=mbdPlayers[1].root,g=p.querySelector('.mbd-detail-grid').getBoundingClientRect(),note=p.querySelector('.mbd-explanation').getBoundingClientRect();return [...p.querySelectorAll('.algorithm-step')].every(e=>e.getBoundingClientRect().bottom<=g.bottom+1&&e.getBoundingClientRect().bottom<note.top);}), 'algorithm must remain inside its column, above the explanation');
 assert.equal((await state(1)).playing,false);
 await players[1].locator('[data-action="play"]').click();assert.equal((await state(1)).frame,0,'Play at end restarts');
 await page.waitForTimeout(2400);assert.equal((await state(1)).frame,1);
 await players[1].locator('[data-action="play"]').click();assert.equal((await state(1)).playing,false);
 await players[1].locator('input').fill('9');await players[1].locator('[data-action="play"]').click();await page.waitForTimeout(2400);assert.equal((await state(1)).frame,10);assert.equal((await state(1)).playing,false,'auto stop at end');
 assert(await players[1].locator('[data-action="continue"]').isDisabled(),'final level has no next iteration');
 await players[1].locator('select').selectOption('6');assert(await players[1].locator('[data-action="continue"]').isDisabled());
 await players[1].locator('input').fill('10');assert(await players[1].locator('[data-action="continue"]').isEnabled());
 await players[1].locator('[data-action="continue"]').click();assert.deepStrictEqual(await state(1),{frame:0,playing:false,iteration:7});
 const numeric=await page.evaluate(()=>mbdTeachingData.history.map(h=>{const ab=mbdTeachingData.alpha_bar[h.level-1],alpha=ab/(h.level===1?1:mbdTeachingData.alpha_bar[h.level-2]);return {weightSum:h.weights.reduce((a,b)=>a+b,0),scoreError:Math.abs(h.score-(-h.noisy+Math.sqrt(ab)*h.weighted_mean)/(1-ab)),updateError:Math.abs(h.next_noisy-(h.noisy+(1-ab)*h.score)/Math.sqrt(alpha))};}));
 assert(numeric.every(x=>Math.abs(x.weightSum-1)<1e-12&&x.scoreError<1e-12&&x.updateError<1e-12));
 const transitions=await page.evaluate(()=>mbdTeachingData.history.map((h,j)=>{const next=mbdTeachingData.history[j+1],prev=h.level===1?1:mbdTeachingData.alpha_bar[h.level-2],std=Math.sqrt((1-prev)/prev);return {level:h.level,meanError:next?Math.abs(next.proposal_mean-h.weighted_mean):Math.abs(h.next_noisy-h.weighted_mean),stdError:next?Math.abs(next.proposal_std-std):std,iterateError:next?Math.abs(next.noisy-h.next_noisy):0,narrows:std<h.proposal_std};}));
 assert(transitions.every(x=>x.meanError<1e-12&&x.stdError<1e-12&&x.iterateError<1e-12&&x.narrows));
 assert.equal(errors.length,0);assert.equal(await page.evaluate(()=>mathErrors.length),0);
 const report={portable,players:2,startsPaused:true,playPause:true,previousNext:true,scrubbing:true,restart:true,iterationSelection:true,algorithmHighlight:true,pauseOnLeave:true,keyIsolation:true,stopsAtEnd:true,denoisingFrames:4,nextIterationButton:true,finalLevelTerminates:true,numericalChecks:numeric,transitions,overflows:checks.flat(),browserErrors:errors};
 fs.writeFileSync(`tmp/slides/qa/mbd-controls${portable?'-portable':''}.json`,JSON.stringify(report,null,2));console.log(JSON.stringify(report));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
