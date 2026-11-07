// Regression checks for the scientific/display boundary: fixed wind, SI units,
// bilinear interpolation, and no artificial nonzero wind floor.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const root='sites/belief-adaptive-monitoring/dist/';
const source=fs.readFileSync(root+'flow-player.js','utf8').split('(async function(){')[0];
const context={};vm.runInNewContext(source+'\nthis.Flow=StationaryFlow;',context);
const Flow=context.Flow;
const meta={grid:[2,2,2],dx_m:1,domain_m:[2,2,2],time_dependence:'none'};
const constant=new Flow(meta,new Float32Array([1,2,3,1,2,3,1,2,3,1,2,3]).buffer);
assert.equal(constant.at([100,100])[0],100);assert.equal(constant.at([100,100])[1],-200);
const zero=new Flow(meta,new Float32Array(12).buffer);assert.equal(Math.hypot(...zero.at([100,100])),0);
assert.throws(()=>new Flow({...meta,time_dependence:'transient'},new Float32Array(12).buffer));
assert.throws(()=>new Flow(meta,new Float32Array(24).buffer));
const linear=new Flow(meta,new Float32Array([0,0,0,2,0,0,0,4,0,2,4,0]).buffer);
assert.equal(linear.at([100,100])[0],100);assert.equal(linear.at([100,100])[1],-200);
for(const s of [1,2]){
 const m=JSON.parse(fs.readFileSync(root+`data/scenario${s}/metadata.json`)),b=fs.readFileSync(root+`data/scenario${s}/`+m.data_file);
 const flow=new Flow(m,b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));
 for(const point of [[150,375],[450,260],[600,440],[650,100]]){
  const expected=flow.at(point,0);assert(expected.every(Number.isFinite));
  for(const t of [1,4,16,1000])assert.deepEqual(flow.at(point,t),expected);
 }
}
console.log('PASS: fixed fields at all times; spatial interpolation; SI/display conversion; exact zero retained; transient data rejected.');
