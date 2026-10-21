const {chromium}=require('/Users/kalebbennaveed/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert');
(async()=>{
 const revision=JSON.parse(fs.readFileSync('output/presentation/wind_aware_mbd/revision_manifest.json','utf8'));
 const url=process.argv[2]||`https://kalebbennaveed.github.io/dis-aware-ipp/update-slides/?revision=${revision.revision}`;
 const out=path.resolve('tmp/slides/optimistic-review/public');fs.mkdirSync(out,{recursive:true});
 const live=url.startsWith('https:');
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',args:['--allow-file-access-from-files']});
 const p=await browser.newPage({viewport:{width:1600,height:900}}),errors=[],failed=[];
 p.on('pageerror',e=>errors.push(String(e)));p.on('response',r=>{if(r.status()>=400)failed.push({url:r.url(),status:r.status()})});
 const response=await p.goto(url+'#/optimistic-problem',{waitUntil:'load',timeout:60000});assert.equal(response.status(),200);
 await p.waitForFunction(()=>window.Reveal?.isReady()&&window.optimisticPlayer);await p.evaluate(()=>document.fonts.ready);
 const go=async id=>{await p.evaluate(id=>{const s=document.getElementById(id);Reveal.slide(Reveal.getIndices(s).h)},id);await p.waitForTimeout(100)};
 assert.equal(await p.evaluate(()=>Reveal.getTotalSlides()),revision.slides);
 assert(await p.evaluate(()=>{
  const ids=Reveal.getSlides().map(s=>s.id);
  return ids.indexOf('current-algorithm')<ids.indexOf('two-field-banks')&&ids.indexOf('optimistic-long-mission')<ids.indexOf('current-implementation')&&ids.indexOf('current-implementation')<ids.indexOf('compute-time');
 }));
 assert.equal(await p.locator('meta[name="robots"]').getAttribute('content'),'noindex, nofollow, noarchive');
 await go('optimistic-architecture');await p.screenshot({path:path.join(out,(live?'live':'package')+'-architecture.png')});
 await go('optimistic-walkthrough');
 await p.locator('#optimistic-walkthrough [data-action="reset"]').click();
 for(let n=1;n<=6;n++){await p.locator('#optimistic-walkthrough [data-action="next"]').click();assert.equal(await p.evaluate(()=>window.optimisticPlayer.index),n);}
 assert(await p.locator('#optimistic-walkthrough [data-action="next"]').isDisabled());
 await p.locator('#optimistic-walkthrough [data-action="prev"]').click();assert.equal(await p.evaluate(()=>window.optimisticPlayer.index),5);
 await p.locator('#optimistic-walkthrough [data-action="reset"]').click();await p.locator('#optimistic-walkthrough [data-action="play"]').click();
 await p.waitForTimeout(3700);assert.equal(await p.evaluate(()=>window.optimisticPlayer.index),1);
 await p.locator('#optimistic-walkthrough [data-action="play"]').click();assert.equal(await p.evaluate(()=>window.optimisticPlayer.playing),false);
 await p.screenshot({path:path.join(out,(live?'live':'package')+'-walkthrough.png')});
 await go('archived-method');assert(await p.locator('#archive-label').isVisible());await p.screenshot({path:path.join(out,(live?'live':'package')+'-archive.png')});
 const gifs=[];
 for(const name of ['original','reversed']){
  await p.evaluate(name=>{const img=document.querySelector(`img[data-original-src="assets/${name}.gif"]`);Reveal.slide(Reveal.getIndices(img.closest('section')).h)},name);
  await p.waitForFunction(name=>{const img=document.querySelector(`img[data-original-src="assets/${name}.gif"]`);return img.complete&&img.naturalWidth>0},name,{timeout:60000});
  gifs.push(name);
 }
 const result=await p.evaluate(()=>({slides:Reveal.getTotalSlides(),mathErrors:window.mathErrors,math:document.querySelectorAll('.katex').length,archiveSlides:document.querySelectorAll('[data-archived="true"]').length,brokenImages:[...document.images].filter(i=>i.getAttribute('src')&&(!i.complete||!i.naturalWidth)).map(i=>i.getAttribute('src'))}));
 Object.assign(result,{url,httpStatus:response.status(),errors,failed,gifs,manualControlsVerified:true,robots:'noindex, nofollow, noarchive'});
 fs.writeFileSync(path.join(out,live?'live.json':'package.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
 await browser.close();assert.equal(result.archiveSlides,44);assert.equal(result.mathErrors.length+errors.length+failed.length+result.brokenImages.length,0);
})().catch(e=>{console.error(e);process.exit(1)});
