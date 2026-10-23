const {chromium}=require('/Users/kalebbennaveed/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path');
(async()=>{
 const output=path.resolve(process.env.QA_OUTPUT||'output/disturbance_suite/qa');fs.mkdirSync(output,{recursive:true});
 const expected=JSON.parse(fs.readFileSync(process.env.REPORT_JSON||'output/disturbance_suite/final_report/results.json','utf8'));
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
 const page=await browser.newPage({viewport:{width:1560,height:1050}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(process.env.REPORT_URL||'http://127.0.0.1:8903/final_report/',{waitUntil:'load'});
 await page.waitForFunction(count=>document.querySelectorAll('#table tr').length===count&&document.getElementById('load').classList.contains('hidden'),expected.rows.length);
 const cases=await page.locator('#case option').evaluateAll(els=>els.map(e=>e.value));
 const table=await page.locator('#table tr').evaluateAll(els=>els.map(e=>Array.from(e.querySelectorAll('td'),td=>td.textContent))),readoutChecks=[];
 for(let i=0;i<expected.rows.length;i++){
  const row=expected.rows[i];
  if(row.valid_safe_comparison){
   if(table[i][2]!==row.aware.cost.toFixed(2)||table[i][3]!==row.blind_filter.cost.toFixed(2)||table[i][4]!==row.reduction_percent.toFixed(1)+'%')throw new Error('Table mismatch '+row.case);
  }else{
   for(const [method,column] of [['aware',2],['blind_filter',3]]){
    const value=row[method];if(!table[i][column].startsWith(value.dense_cost_before_failure.toFixed(2)+' s through '+value.completed_seconds.toFixed(2)+' s'))throw new Error('Terminated mission cost mismatch '+row.case);
   }
   if(!table[i][4].startsWith('Excluded:')||table[i][4].includes('%'))throw new Error('Contact cost presented as performance reduction '+row.case);
  }
 }
 for(const name of cases){
  await page.selectOption('#case',name);await page.waitForFunction(()=>document.getElementById('load').classList.contains('hidden'));
  await page.locator('#time').evaluate(e=>{e.value=225;e.dispatchEvent(new Event('input'))});
  for(const layer of ['clarity','physical','wind','std'])await page.selectOption('#layer',layer);
  await page.selectOption('#layer','clarity');
  const readouts=await page.evaluate(()=>['aware','blind_filter'].map(method=>{
   const d=DATA[active][method],frame=closest(d.frames,t),box=document.getElementById(method+'Readout');
   const radius=Math.max(frame.p[2],0)*Math.tan((d.sensing?.half_angle_degrees??45)*Math.PI/180);
   const result=REPORT.rows.find(r=>r.case===active)[method];
   return {method,wind:Number(box.dataset.windSpeed),expectedWind:Math.hypot(...frame.wind),radius:Number(box.dataset.sensingRadius),expectedRadius:radius,text:box.textContent,ended:box.dataset.ended==='true',expectedEnded:t>result.completed_seconds+1e-5};
  }));
  for(const check of readouts)if(Math.abs(check.wind-check.expectedWind)>1e-12||Math.abs(check.radius-check.expectedRadius)>1e-12||!check.text.includes('m/s')||!check.text.includes('sensing radius')||check.ended!==check.expectedEnded||(check.ended&&!check.text.includes('Recording ended at')))throw new Error('Readout mismatch '+name);
  const row=expected.rows.find(r=>r.case===name);
  if(!row.valid_safe_comparison){
   const metrics=await page.locator('#metrics').textContent();if(!metrics.includes('No valid full-mission comparison')||metrics.includes('%'))throw new Error('Invalid full-mission metric '+name);
   const maximum=Number(await page.locator('#cost').getAttribute('data-observed-maximum'));
   if(Math.abs(maximum-Math.max(row.aware.dense_cost_before_failure,row.blind_filter.dense_cost_before_failure))>1e-4)throw new Error('Unobserved penalty affects cost curve '+name);
   if(!await page.locator('#costStatus').isVisible())throw new Error('Missing curve termination label '+name);
  }
  readoutChecks.push({case:name,readouts});
  if(['original','reversed','courtyards','freight','interchange','distributed'].includes(name)||!row.valid_safe_comparison)await page.locator('section').screenshot({path:path.join(output,name+'.png')});
  if(name==='interchange'){await page.selectOption('#layer','physical');await page.locator('section').screenshot({path:path.join(output,name+'-physical-wind.png')});await page.selectOption('#layer','clarity');}
 }
 const swatch=await page.locator('.wind-swatch').evaluate(e=>getComputedStyle(e).backgroundImage);
 if(swatch==='none')throw new Error('Physical wind color scale did not render');
 await page.uncheck('#trueWind');if(await page.locator('#physicalWindKey').isVisible())throw new Error('Vector legend remains visible with vectors hidden');
 await page.selectOption('#layer','physical');if(!await page.locator('#physicalWindKey').isVisible())throw new Error('Physical wind layer needs a scale');
 await page.selectOption('#layer','clarity');await page.check('#trueWind');
 await page.selectOption('#mapsChoice','aware');const single=await page.locator('.panel:not(.hidden)').count();
 await page.selectOption('#mapsChoice','blind_filter');const baseline=await page.locator('.panel:not(.hidden)').count();
 await page.selectOption('#mapsChoice','both');
 for(const value of ['aware','blind_filter','both'])await page.selectOption('#costChoice',value);
 await page.click('#reset');await page.click('#play');await page.waitForTimeout(1200);await page.click('#play');
 const played=Number(await page.locator('#time').inputValue());
 const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
 const timingRows=await page.locator('#timingTable tr').count();
 const expectedTimingRows=expected.single_process_timing?.rows?.length??0;
 const result={cases,errors,single,baseline,played,overflow,timingRows,expectedTimingRows,table,readoutChecks,swatch};
 fs.writeFileSync(path.join(output,'browser_checks.json'),JSON.stringify(result,null,2));
 if(errors.length||cases.length!==expected.rows.length||single!==1||baseline!==1||played<=0||overflow||timingRows!==expectedTimingRows)throw new Error(JSON.stringify(result));
 console.log(JSON.stringify({cases:cases.length,errors,single,baseline,played,overflow,timingRows},null,2));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
