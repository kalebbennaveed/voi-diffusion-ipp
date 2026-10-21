const {chromium}=require('/Users/kalebbennaveed/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs');const path=require('path');
(async()=>{
 const dir=path.resolve('output/presentation/wind_aware_mbd');
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',args:['--allow-file-access-from-files']});
 const page=await browser.newPage({viewport:{width:1600,height:900},deviceScaleFactor:1});
 const network=[];page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url());});
 await page.goto('file://'+dir+'/slides.html?print-pdf',{waitUntil:'load'});
 await page.waitForFunction(()=>window.Reveal&&Reveal.isReady()&&document.querySelector('.pdf-page'));
 await page.evaluate(()=>document.fonts.ready);
 const mathErrors=await page.evaluate(()=>window.mathErrors||[]);
 if(mathErrors.length)throw Error(JSON.stringify(mathErrors));
 await page.evaluate(()=>{
  const logo=document.getElementById('brand-logo');
  document.querySelectorAll('.reveal .slides section').forEach(s=>{
   const l=logo.cloneNode();l.removeAttribute('id');l.className='print-logo';s.append(l);
   if(s.querySelector('.print-poster')){
    const caption=s.querySelector('.caption');
    if(caption)caption.textContent=s.querySelector('.print-poster').dataset.printCaption || 'Static snapshot at 60 s · Open the HTML presentation for the synchronized animation';
   }
  });
 });
 await page.addStyleTag({content:'.print-logo{position:absolute!important;right:20px!important;top:0!important;width:180px!important;height:auto!important;margin:0!important;}'});
 await page.emulateMedia({media:'print'});
 await page.waitForFunction(()=>[...document.images].every(i=>i.complete));
 await page.waitForTimeout(1000);
 await page.pdf({path:dir+'/slides.pdf',printBackground:true,preferCSSPageSize:true,margin:{top:0,bottom:0,left:0,right:0}});
 console.log(JSON.stringify({pdf:dir+'/slides.pdf',pages:await page.locator('.pdf-page').count(),externalRequests:network}));
 const figure=await browser.newPage({viewport:{width:2000,height:930}});
 await figure.setContent('<!doctype html><html><head><style>@page{size:2000px 930px;margin:0}html,body{width:2000px;height:930px;margin:0;padding:0}svg{display:block;width:2000px;height:930px}</style></head><body>'+fs.readFileSync(dir+'/assets/architecture.svg','utf8')+'</body></html>');
 await figure.pdf({path:dir+'/assets/architecture.pdf',printBackground:true,preferCSSPageSize:true,margin:{top:0,bottom:0,left:0,right:0}});
 await figure.screenshot({path:dir+'/assets/architecture.png'});
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
