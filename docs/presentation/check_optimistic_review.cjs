const {chromium}=require('/Users/kalebbennaveed/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path');
(async()=>{
 const out=path.resolve('tmp/slides/optimistic-review/qa');fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',args:['--allow-file-access-from-files']});
 const p=await browser.newPage({viewport:{width:1600,height:900}});const errors=[];p.on('pageerror',e=>errors.push(String(e)));
 await p.goto('file://'+path.resolve('output/presentation/wind_aware_mbd/slides.html')+'?snapshot',{waitUntil:'load'});await p.waitForFunction(()=>window.Reveal?.isReady());await p.evaluate(()=>document.fonts.ready);
 const slides=await p.evaluate(()=>Reveal.getSlides().map(s=>({id:s.id,title:s.querySelector('h2')?.innerText,archive:s.dataset.archived==='true'})));
 let report=[];
 for(let i=0;i<slides.length;i++){
  await p.evaluate(i=>Reveal.slide(i),i);await p.waitForTimeout(100);
  await p.screenshot({path:path.join(out,`${String(i+1).padStart(3,'0')}.png`)});
  report.push(await p.evaluate(()=>{const s=Reveal.getCurrentSlide(),r=s.getBoundingClientRect();return{id:s.id,title:s.querySelector('h2')?.innerText,images:[...s.querySelectorAll('img')].filter(e=>!e.complete||e.naturalWidth===0).map(e=>e.getAttribute('src')),overflow:[...s.querySelectorAll('h2,p,li,table,img,.key-line,.small-note,.source,.katex-display,.definition-grid,.two-points,.optimistic-player-grid,.optimistic-explanation,.optimistic-equation,.caption')].filter(e=>!e.closest('.notes')).filter(e=>{const b=e.getBoundingClientRect();return b.width>0&&b.height>0&&(b.bottom>r.bottom+3||b.right>r.right+3||b.left<r.left-3);}).map(e=>({tag:e.tagName,text:(e.innerText||e.getAttribute('alt')||'').slice(0,80),rect:e.getBoundingClientRect().toJSON()}))}}));
 }
 const idx=slides.findIndex(s=>s.id==='optimistic-walkthrough');await p.evaluate(i=>Reveal.slide(i),idx);
 const stages=[];for(let i=0;i<7;i++){await p.evaluate(i=>window.optimisticPlayer.seek(i),i);await p.screenshot({path:path.join(out,`walk-${i}.png`)});stages.push(await p.evaluate(()=>({step:window.optimisticPlayer.index,active:document.querySelectorAll('#optimistic-walkthrough .active').length,eq:document.querySelector('#optimistic-walkthrough .optimistic-equation').textContent})));}
 fs.writeFileSync(path.join(out,'layout.json'),JSON.stringify({slides,errors,report},null,2));
 await p.evaluate(()=>document.documentElement.classList.remove('snapshot'));
 await p.locator('#optimistic-walkthrough [data-action="reset"]').click();
 await p.locator('#optimistic-walkthrough [data-action="next"]').click();const next=await p.evaluate(()=>window.optimisticPlayer.index);
 await p.locator('#optimistic-walkthrough [data-action="prev"]').click();const prev=await p.evaluate(()=>window.optimisticPlayer.index);
 await p.locator('#optimistic-walkthrough [data-action="play"]').click();await p.waitForTimeout(3700);const playing=await p.evaluate(()=>({index:window.optimisticPlayer.index,playing:window.optimisticPlayer.playing}));await p.evaluate(()=>Reveal.next());const paused=await p.evaluate(()=>!window.optimisticPlayer.playing);
 const math=await p.evaluate(()=>({errors:window.mathErrors,rendered:document.querySelectorAll('.katex').length}));
 const output={slides,errors,math,report,stages,controls:{next,prev,playing,paused}};fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(output,null,2));
 console.log(JSON.stringify({count:slides.length,errors,math,overflow:report.map((r,i)=>({slide:i+1,title:r.title,items:r.overflow})).filter(r=>r.items.length),missing:report.filter(r=>r.images.length),controls:output.controls},null,2));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
