const {chromium}=require('/Users/kalebbennaveed/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const path=require('path'),fs=require('fs');
(async()=>{
 const b=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
 const p=await b.newPage({viewport:{width:1600,height:900}}),errs=[],req=[];
 p.setDefaultTimeout(30000);
 p.on('crash',()=>{console.error('Portable page renderer crashed');process.exit(1)});
 p.on('pageerror',e=>errs.push(String(e)));p.on('request',r=>{if(/^https?:/.test(r.url()))req.push(r.url());});
 await p.goto('file://'+path.resolve('output/presentation/wind_aware_mbd/slides-standalone.html'),{waitUntil:'load',timeout:120000});
 console.log('Portable file loaded');
 await p.waitForFunction(()=>window.Reveal&&Reveal.isReady(),null,{timeout:30000});await p.evaluate(()=>document.fonts.ready);
 const result=await p.evaluate(()=>({slides:Reveal.getTotalSlides(),mathErrors:window.mathErrors,math:document.querySelectorAll('.katex').length,brokenImages:[...document.images].filter(i=>!i.complete||i.naturalWidth===0).map(i=>i.alt),nonEmbeddedResources:[...document.querySelectorAll('script[src],link[rel="stylesheet"],img[src]')].map(e=>e.getAttribute('src')||e.getAttribute('href')).filter(u=>!u.startsWith('data:')),looseTermOccurrences:(document.querySelector('.slides').innerText.match(/\bcandidate\b|\binspection cost\b/gi)||[]).length}));
 result.browserErrors=errs;result.externalRequests=req;
 fs.writeFileSync('tmp/slides/qa/portable.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result));
 const expected=(fs.readFileSync('output/presentation/wind_aware_mbd/slides.qmd','utf8').match(/^## /gm)||[]).length;
 await b.close();if(result.slides!==expected||result.mathErrors.length||result.brokenImages.length||errs.length||req.length||result.nonEmbeddedResources.length||result.looseTermOccurrences)process.exit(1);
})().catch(e=>{console.error(e);process.exit(1)});
