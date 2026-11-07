'use strict';
// One fixed spatial field per scenario. No time argument or temporal blending.
// Air velocity is converted to display units, never to robot acceleration.
class StationaryFlow {
  constructor(metadata,buffer){
    this.meta=metadata;this.values=new Float32Array(buffer);this.nx=metadata.grid[0];this.ny=metadata.grid[1];this.dx=metadata.dx_m;
    if(metadata.time_dependence!=='none'||this.values.length!==this.nx*this.ny*3)throw new Error('Expected one time-independent velocity field');
  }
  sample(p){
    const px=p[0]*.01,py=this.meta.domain_m[1]-p[1]*.01;
    let x=Math.max(0,Math.min(this.nx-1,px/this.dx-.5)),y=Math.max(0,Math.min(this.ny-1,py/this.dx-.5));
    const ix=Math.min(this.nx-2,Math.floor(x)),iy=Math.min(this.ny-2,Math.floor(y)),fx=x-ix,fy=y-iy;
    const b=(iy*this.nx+ix)*3,v=this.values;
    const component=k=>(1-fy)*((1-fx)*v[b+k]+fx*v[b+k+3])+fy*((1-fx)*v[b+k+this.nx*3]+fx*v[b+k+this.nx*3+3]);
    return [component(0),component(1),component(2)];
  }
  at(p){
    const v=this.sample(p);return [v[0]*100,-v[1]*100];
  }
  paint(svgImage){
    const canvas=document.createElement('canvas');canvas.width=this.nx;canvas.height=this.ny;
    const ctx=canvas.getContext('2d'),img=ctx.createImageData(this.nx,this.ny);
    const stops=[[250,253,253],[221,239,245],[144,203,220],[55,145,178],[18,78,119]];
    for(let j=0;j<this.ny;j++)for(let i=0;i<this.nx;i++){
      const k=((this.ny-j-1)*this.nx+i)*3,speed=Math.hypot(this.values[k],this.values[k+1]);
      const z=Math.min(1,speed/12)*4,si=Math.min(3,Math.floor(z)),a=z-si,d=(j*this.nx+i)*4;
      for(let c=0;c<3;c++)img.data[d+c]=Math.round(stops[si][c]*(1-a)+stops[si+1][c]*a);img.data[d+3]=255;
    }
    ctx.putImageData(img,0,0);svgImage.setAttribute('href',canvas.toDataURL('image/png'));
  }
}
(async function(){
  const status=document.querySelector('#flow-status'),renderers=[];
  try{
    await Promise.all([1,2,3].filter(s=>document.querySelector(`#map${s}`)).map(async scenario=>{
      const directory=`data/scenario${scenario}/`;
      const meta=await(await fetch(directory+'metadata.json')).json(),field=new StationaryFlow(meta,await(await fetch(directory+meta.data_file)).arrayBuffer());
      const height=100*meta.domain_m[1],container=document.querySelector(`#map${scenario}`);
      const obstacles=meta.obstacles.map(o=>({center:[100*o[0],height-100*o[1]],radius:100*o[2]}));
      const displayRegions=scenario===3?[{label:'Monitoring region',bounds:[[Math.min(...meta.regions.map(r=>r.bounds[0][0])),Math.min(...meta.regions.map(r=>r.bounds[0][1]))],[Math.max(...meta.regions.map(r=>r.bounds[1][0])),Math.max(...meta.regions.map(r=>r.bounds[1][1]))]]}]:meta.regions;
      const regions=displayRegions.map(r=>{const [lo,hi]=r.bounds;return `<rect x="${100*lo[0]}" y="${height-100*hi[1]}" width="${100*(hi[0]-lo[0])}" height="${100*(hi[1]-lo[1])}" fill="#fff" fill-opacity=".13" stroke="#41675b" stroke-width="2"/><text class="region-label monitor-region-label" x="${100*lo[0]+14}" y="${height-100*hi[1]+27}">${scenario===3?r.label:'Region '+r.label}</text>`;}).join('');
      container.innerHTML=`<svg viewBox="-5 -5 810 ${height+10}" role="img" aria-label="Scenario ${scenario}, physical geometry and stationary wind"><rect width="800" height="${height}" fill="#f7fbff"/><image width="800" height="${height}"/>${regions}${obstacles.map(o=>`<circle cx="${o.center[0]}" cy="${o.center[1]}" r="${o.radius}" fill="#73828a"/>`).join('')}<rect width="800" height="${height}" fill="none" stroke="#536b67" stroke-width="2"/></svg>`;
      const svg=container.querySelector('svg');field.paint(svg.querySelector('image'));
      const renderer=new WindAnimation({container,svg,field:p=>field.at(p),bounds:[[0,0],[800,height]],obstacles,count:500,seed:scenario*813});renderer.setSpeed(.25);renderer.play();renderers.push(renderer);
      svg.addEventListener('pointermove',e=>{const p=new DOMPoint(e.clientX,e.clientY).matrixTransform(svg.getScreenCTM().inverse());const w=field.sample([p.x,p.y]);document.querySelector(`#probe${scenario}`).textContent=`(${(p.x/100).toFixed(2)}, ${((height-p.y)/100).toFixed(2)}) m · wind (${w[0].toFixed(2)}, ${w[1].toFixed(2)}) m/s · fixed in time`;});
      document.querySelector(`#wind-regions-${scenario}`).textContent=scenario===3?`Monitoring region mean ${meta.monitoring_band_metrics.mean_m_s.toFixed(2)} m/s · lower-right peak ${meta.lower_right_metrics.max_m_s.toFixed(2)} m/s`:`Mean wind: A ${meta.monitoring_region_metrics.A.mean_m_s.toFixed(2)} m/s · B ${meta.monitoring_region_metrics.B.mean_m_s.toFixed(2)} m/s`;
    }));
    let playing=true;const button=document.querySelector('#play-flow');button.disabled=false;button.addEventListener('click',()=>{playing=!playing;for(const r of renderers)playing?r.play():r.pause();button.textContent=playing?'Ⅱ Pause wind particles':'▶ Play wind particles';button.setAttribute('aria-pressed',String(playing));});
    document.querySelector('#flow-speed').addEventListener('change',e=>renderers.forEach(r=>r.setSpeed(Number(e.target.value))));
    document.addEventListener('visibilitychange',()=>renderers.forEach(r=>(document.hidden||!playing)?r.pause():r.play()));
    status.textContent='Recomputed 3D CFD · fixed vector mean over 16–32 s · particles move through the stationary field.';
  }catch(e){status.textContent=`Wind data: ${e.message}`;}
})();
