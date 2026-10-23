const {chromium}=require('/Users/kalebbennaveed/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path');
(async()=>{
 const output=path.resolve('output/disturbance_suite/qa');fs.mkdirSync(output,{recursive:true});
 const expected=JSON.parse(fs.readFileSync('output/disturbance_suite/final_report/results.json','utf8'));
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
 const page=await browser.newPage({viewport:{width:1560,height:1050}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));
 await page.goto('http://127.0.0.1:8903/final_report/',{waitUntil:'load'});
 await page.waitForFunction(()=>document.querySelectorAll('#table tr').length===10&&document.getElementById('load').classList.contains('hidden'));
 const cases=await page.locator('#case option').evaluateAll(els=>els.map(e=>e.value));
 const table=await page.locator('#table tr').evaluateAll(els=>els.map(e=>Array.from(e.querySelectorAll('td'),td=>td.textContent)));
 for(let i=0;i<expected.rows.length;i++){
  const row=expected.rows[i];if(table[i][2]!==row.aware.cost.toFixed(2)||table[i][3]!==row.blind_filter.cost.toFixed(2)||table[i][4]!==row.reduction_percent.toFixed(1)+'%')throw new Error('Table mismatch '+row.case);
 }
 for(const name of cases){
  await page.selectOption('#case',name);await page.waitForFunction(()=>document.getElementById('load').classList.contains('hidden'));
  await page.locator('#time').evaluate(e=>{e.value=225;e.dispatchEvent(new Event('input'))});
  for(const layer of ['clarity','wind','std'])await page.selectOption('#layer',layer);
  await page.selectOption('#layer','clarity');
  if(['original','reversed','courtyards','freight','distributed'].includes(name))await page.locator('section').screenshot({path:path.join(output,name+'.png')});
 }
 await page.selectOption('#mapsChoice','aware');const single=await page.locator('.panel:not(.hidden)').count();
 await page.selectOption('#mapsChoice','blind_filter');const baseline=await page.locator('.panel:not(.hidden)').count();
 await page.selectOption('#mapsChoice','both');
 for(const value of ['aware','blind_filter','both'])await page.selectOption('#costChoice',value);
 await page.click('#reset');await page.click('#play');await page.waitForTimeout(1200);await page.click('#play');
 const played=Number(await page.locator('#time').inputValue());
 const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
 const timingRows=await page.locator('#timingTable tr').count();
 const result={cases,errors,single,baseline,played,overflow,timingRows,table};
 fs.writeFileSync(path.join(output,'browser_checks.json'),JSON.stringify(result,null,2));
 if(errors.length||cases.length!==10||single!==1||baseline!==1||played<=0||overflow||timingRows!==6)throw new Error(JSON.stringify(result));
 console.log(JSON.stringify({cases:cases.length,errors,single,baseline,played,overflow,timingRows},null,2));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
