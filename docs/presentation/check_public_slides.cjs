const {chromium}=require('/Users/kalebbennaveed/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),assert=require('assert');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
 const page=await browser.newPage({viewport:{width:1600,height:900}}),errors=[],responses=[];
 page.on('pageerror',e=>errors.push(String(e)));
 page.on('response',r=>{if(r.status()>=400)responses.push({url:r.url(),status:r.status()})});
 const url='https://kalebbennaveed.github.io/dis-aware-ipp/update-slides/';
 const response=await page.goto(url+'#/poam-visual',{waitUntil:'load',timeout:60000});assert(response.status()===200);
 await page.waitForFunction(()=>window.walkPlayers?.length===3&&window.mbdPlayers?.length===2);await page.evaluate(()=>document.fonts.ready);
 assert(await page.locator('meta[name="robots"]').getAttribute('content')==='noindex, nofollow, noarchive');
 await page.evaluate(()=>Reveal.slide([...document.querySelectorAll('.slides>section')].indexOf(document.querySelector('#mbd-one-iteration'))));
 const mbd=page.locator('[data-player="detail"]');
 assert(await mbd.locator('select').inputValue()==='6');
 for(const frame of [7,8,9,10]){await mbd.locator('input').fill(String(frame));assert(await mbd.locator('.algorithm-step.active').getAttribute('data-step')==='7');if(frame===9)await page.screenshot({path:'tmp/slides/qa/public-denoising.png'});}
 await mbd.locator('[data-action="continue"]').click();assert(await mbd.locator('select').inputValue()==='7');
 await mbd.locator('input').fill('10');assert(await mbd.locator('[data-action="continue"]').isDisabled());
 await page.evaluate(()=>Reveal.slide([...document.querySelectorAll('.slides>section')].indexOf(document.querySelector('#poam-visual'))));
 for(let i=0;i<2;i++)await page.locator('[data-walk="poam"] [data-action="next"]').click();
 assert(await page.locator('[data-walk="poam"]').getAttribute('data-step')==='2');
 await page.screenshot({path:'tmp/slides/qa/public-poam.png'});
 await page.evaluate(()=>Reveal.slide([...document.querySelectorAll('.slides>section')].indexOf(document.querySelector('#proposed-policy-walkthrough'))));
 for(let i=0;i<6;i++)await page.locator('[data-walk="rollout"] [data-action="next"]').click();
 assert(await page.locator('[data-walk="rollout"]').getAttribute('data-step')==='6');
 await page.screenshot({path:'tmp/slides/qa/public-method.png'});
 const gifs=[];
 for(const name of ['original','reversed','open']){
  await page.evaluate(name=>{const img=document.querySelector(`img[data-original-src="assets/${name}.gif"]`);Reveal.slide([...document.querySelectorAll('.slides>section')].indexOf(img.closest('section')))},name);
  await page.waitForFunction(name=>{const i=document.querySelector(`img[data-original-src="assets/${name}.gif"]`);return i.complete&&i.naturalWidth>0},name,{timeout:60000});
  gifs.push({name,loaded:true});
  await page.screenshot({path:`tmp/slides/qa/public-${name}.png`});
 }
 const result={url,httpStatus:response.status(),slides:await page.evaluate(()=>Reveal.getTotalSlides()),players:5,manualControlsVerified:true,denoisingTransitionVerified:true,finalLevelTerminates:true,gifs,robots:'noindex, nofollow, noarchive',errors,responses,mathErrors:await page.evaluate(()=>window.mathErrors)};
 fs.writeFileSync('tmp/slides/qa/public.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result));
 await browser.close();if(errors.length||responses.length||result.mathErrors.length)process.exitCode=1;
})().catch(e=>{console.error(e);process.exit(1)});
