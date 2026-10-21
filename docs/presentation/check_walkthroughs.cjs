const {chromium}=require('/Users/kalebbennaveed/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',args:['--allow-file-access-from-files']});
 const page=await browser.newPage({viewport:{width:1600,height:900}}),errors=[],failures=[];
 const target=process.argv[2]||'file://'+path.resolve('output/presentation/wind_aware_mbd/slides.html');
 page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(target,{waitUntil:'load'});await page.waitForFunction(()=>window.walkPlayers?.length===3);await page.evaluate(()=>document.fonts.ready);
 fs.mkdirSync('tmp/slides/qa/walkthroughs',{recursive:true});
 for(const kind of ['poam','rollout','cycle']){
  const root=page.locator(`[data-walk="${kind}"]`),count=await root.locator('.walk-step').count();
  await page.evaluate(kind=>{const a=walkPlayers.find(a=>a.kind===kind);Reveal.slide([...document.querySelectorAll('.slides>section')].indexOf(a.root.closest('section')))},kind);
  assert(await page.evaluate(kind=>{const a=walkPlayers.find(a=>a.kind===kind);return a.index===0&&!a.playing},kind));
  await root.locator('[data-action="next"]').click();assert(await root.getAttribute('data-step')==='1');
  await root.locator('[data-action="prev"]').click();assert(await root.getAttribute('data-step')==='0');
  await root.locator('[data-action="play"]').click();await page.waitForTimeout(2650);assert(await root.getAttribute('data-step')==='1');
  await root.locator('[data-action="play"]').click();assert(await page.evaluate(kind=>!walkPlayers.find(a=>a.kind===kind).playing,kind));
  await root.locator('input').fill('2');assert(await root.getAttribute('data-step')==='2');
  const before=await page.evaluate(()=>Reveal.getIndices().h);await root.locator('input').press('ArrowRight');assert(await page.evaluate(()=>Reveal.getIndices().h)===before);
  for(let i=0;i<count;i++){
   await root.locator('input').fill(String(i));await page.waitForTimeout(70);
   assert(await root.locator('.walk-step.active').count()===1);
   assert(await root.locator('.walk-step').nth(i).getAttribute('aria-current')==='step');
   const bounds=await root.evaluate(el=>{
    const svg=el.querySelector('svg'),vb=svg.viewBox.baseVal,issues=[];
    for(const t of svg.querySelectorAll('text')){const b=t.getBBox();if(b.x<0||b.y<0||b.x+b.width>vb.width+1||b.y+b.height>vb.height+1)issues.push({type:'svg-text-outside',text:t.textContent,b:{x:b.x,y:b.y,w:b.width,h:b.height}})}
    const grid=el.querySelector('.walk-grid').getBoundingClientRect(),ex=el.querySelector('.walk-explanation').getBoundingClientRect();
    if(grid.bottom>ex.top+1)issues.push({type:'grid-explanation-overlap'});
    const all=[...svg.querySelectorAll('text')].map(t=>({t:t.textContent,b:t.getBBox()}));
    for(let i=0;i<all.length;i++)for(let j=i+1;j<all.length;j++){let a=all[i].b,b=all[j].b;if(Math.min(a.x+a.width,b.x+b.width)-Math.max(a.x,b.x)>3&&Math.min(a.y+a.height,b.y+b.height)-Math.max(a.y,b.y)>3)issues.push({type:'svg-text-overlap',text:[all[i].t,all[j].t]})}
    const sec=el.closest('section').getBoundingClientRect();for(const e of el.querySelectorAll('.walk-step,.walk-equation,.walk-controls,.walk-explanation')){const r=e.getBoundingClientRect();if(r.width&&r.bottom>sec.bottom+4)issues.push({type:'slide-overflow',text:e.textContent.slice(0,70)})}
    return issues;
   });
   if(bounds.length)failures.push({kind,step:i,issues:bounds});
   await page.screenshot({path:`tmp/slides/qa/walkthroughs/${kind}-${i+1}.png`});
  }
  await root.locator('[data-action="reset"]').click();assert(await root.getAttribute('data-step')==='0');
  await root.locator('[data-action="play"]').click();await page.evaluate(()=>Reveal.next());assert(await page.evaluate(kind=>!walkPlayers.find(a=>a.kind===kind).playing,kind));
 }
 const result={target,players:3,steps:20,errors,failures,mathErrors:await page.evaluate(()=>window.mathErrors||[])};
 fs.writeFileSync('tmp/slides/qa/walkthroughs.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result));
 await browser.close();if(errors.length||failures.length||result.mathErrors.length)process.exitCode=1;
})().catch(e=>{console.error(e);process.exit(1)});
