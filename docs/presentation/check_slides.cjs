const {chromium}=require('/Users/kalebbennaveed/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs');const path=require('path');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',args:['--allow-file-access-from-files']});
 const page=await browser.newPage({viewport:{width:1600,height:900},deviceScaleFactor:1});const errors=[],network=[];
 fs.mkdirSync('tmp/slides/qa',{recursive:true});
 page.on('pageerror',err=>errors.push(String(err)));
 page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url());});
 await page.goto('file://'+path.resolve('output/presentation/wind_aware_mbd/slides.html'),{waitUntil:'load'});
 await page.waitForFunction(()=>window.Reveal&&Reveal.isReady());
 await page.evaluate(()=>document.fonts.ready);
 const math=await page.evaluate(()=>({errors:window.mathErrors,rendered:document.querySelectorAll('.katex').length}));
 const svgFiles=await page.evaluate(()=>[...new Set([...document.querySelectorAll('img')].map(e=>e.src).filter(s=>s.endsWith('.svg')))]);
 const n=await page.evaluate(()=>Reveal.getTotalSlides());let report=[];
 for(let i=0;i<n;i++){
  await page.evaluate(i=>Reveal.slide(i),i);await page.waitForTimeout(650);
  await page.screenshot({path:`tmp/slides/qa/slide-${String(i+1).padStart(2,'0')}.png`});
  report.push(await page.evaluate(()=>{const s=Reveal.getCurrentSlide(),r=s.getBoundingClientRect();const nodes=[...s.querySelectorAll('h2,p,li,table,img,.key-line,.caption,.result-line,.small-note,.katex-display,.katex-html,.definition-grid,.source,.statement,.algorithm-grid,.two-points,.parameter-row')].filter(e=>!e.closest('.notes')&&e.getBoundingClientRect().height>0);return {title:s.querySelector('h2')?.innerText,images:[...s.querySelectorAll('img')].map(e=>({ok:e.complete&&e.naturalWidth>0,alt:e.alt})),overflow:nodes.filter(e=>{let b=e.getBoundingClientRect();return b.bottom>r.bottom+5||b.right>r.right+5||b.left<r.left-5}).map(e=>({tag:e.tagName,text:e.innerText?.slice(0,100),rect:e.getBoundingClientRect().toJSON()}))};}));
 }

 const svgReport=[];const svgPage=await browser.newPage();
 for(const url of svgFiles){
  await svgPage.goto(url);await svgPage.evaluate(()=>document.fonts.ready);
  svgReport.push(await svgPage.evaluate(()=>{
   const svg=document.querySelector('svg'), vb=svg.viewBox.baseVal;
   const panels=[...svg.querySelectorAll('[data-panel]')];
   const panelBounds=Object.fromEntries(panels.map(p=>[p.id,p.dataset.bounds.split(',').map(Number)]));
   const failures=[];
   for(const t of svg.querySelectorAll('text')){
    const local=t.getBBox(), matrix=svg.getCTM().inverse().multiply(t.getCTM()); const corners=[[local.x,local.y],[local.x+local.width,local.y],[local.x,local.y+local.height],[local.x+local.width,local.y+local.height]].map(([x,y])=>new DOMPoint(x,y).matrixTransform(matrix)); const xx=corners.map(p=>p.x), yy=corners.map(p=>p.y); const b={x:Math.min(...xx),y:Math.min(...yy),width:Math.max(...xx)-Math.min(...xx),height:Math.max(...yy)-Math.min(...yy)}; const p=t.closest('[data-panel]');
    const a=p?panelBounds[p.id]:[vb.x,vb.y,vb.width,vb.height];const pad=p?10:0;
    if(b.x<a[0]+pad-1||b.y<a[1]+pad-1||b.x+b.width>a[0]+a[2]-pad+1||b.y+b.height>a[1]+a[3]-pad+1)
     failures.push({type:'text-containment',panel:p?.id,text:t.textContent,bounds:{x:b.x,y:b.y,w:b.width,h:b.height},container:a});
    if(!p)for(const [id,[x,y,w,h]] of Object.entries(panelBounds)){
     if(Math.min(b.x+b.width,x+w-3)-Math.max(b.x,x+3)>2&&Math.min(b.y+b.height,y+h-3)-Math.max(b.y,y+3)>2)
      failures.push({type:'external-label-crosses-panel',panel:id,text:t.textContent});
    }
   }

   for(const panel of panels){
    const ts=[...panel.querySelectorAll('text')].map(e=>({text:e.textContent,b:e.getBoundingClientRect()}));
    for(let i=0;i<ts.length;i++)for(let j=i+1;j<ts.length;j++){
     const a=ts[i].b,b=ts[j].b;
     if(Math.min(a.right,b.right)-Math.max(a.left,b.left)>1&&Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>1)
      failures.push({type:'text-overlap',panel:panel.id,text:[ts[i].text,ts[j].text]});
    }
   }
   const connectors=[...svg.querySelectorAll('[data-connector]')];
   const onEdge=(p,a)=>{let[x,y,w,h]=a;return (Math.abs(p.x-x)<2||Math.abs(p.x-x-w)<2)&&p.y>=y&&p.y<=y+h||(Math.abs(p.y-y)<2||Math.abs(p.y-y-h)<2)&&p.x>=x&&p.x<=x+w;};
   for(const c of connectors){
    const [src,dst]=c.dataset.connector.split(':'), pts=[...c.points];
    for(const [id,p] of [[src,pts[0]],[dst,pts.at(-1)]])if(id&&!onEdge(p,panelBounds[id]))failures.push({type:'connector-endpoint',id,x:p.x,y:p.y});
    // Connecting lines must not cross an unrelated panel interior.
    for(const [id,[x,y,w,h]] of Object.entries(panelBounds)){
     if(id===src||id===dst)continue;
     for(let i=1;i<pts.length;i++){
      const a=pts[i-1],b=pts[i];
      if(a.x===b.x&&a.x>x+2&&a.x<x+w-2&&Math.max(a.y,b.y)>y+2&&Math.min(a.y,b.y)<y+h-2||a.y===b.y&&a.y>y+2&&a.y<y+h-2&&Math.max(a.x,b.x)>x+2&&Math.min(a.x,b.x)<x+w-2)
       failures.push({type:'connector-crosses-panel',id,connector:c.dataset.connector});
     }
    }
   }
   return {file:location.pathname.split('/').at(-1),panels:panels.length,connectors:connectors.filter(e=>e.dataset.connector!==':').length,failures};
  }));
 }
 fs.writeFileSync('tmp/slides/qa/svg-layout.json',JSON.stringify(svgReport,null,2));
 console.log(JSON.stringify({svgFailures:svgReport.filter(s=>s.failures.length)}));
 fs.writeFileSync('tmp/slides/qa/layout.json',JSON.stringify({slides:n,errors,math,network,report},null,2));console.log(JSON.stringify({slides:n,errors,math,network,overflow:report.map((r,i)=>({slide:i+1,items:r.overflow})).filter(x=>x.items.length)}));
 await browser.close();
})();
