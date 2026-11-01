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
      const k=((this.ny-j-1)*this.nx+i)*3,speed=Math.hypot(this.values[k],this.values[k+1],this.values[k+2]);
      const z=Math.min(1,speed/12)*4,si=Math.min(3,Math.floor(z)),a=z-si,d=(j*this.nx+i)*4;
      for(let c=0;c<3;c++)img.data[d+c]=Math.round(stops[si][c]*(1-a)+stops[si+1][c]*a);img.data[d+3]=255;
    }
    ctx.putImageData(img,0,0);svgImage.setAttribute('href',canvas.toDataURL('image/png'));
  }
}
(async function(){
  const status=document.querySelector('#flow-status'),button=document.querySelector('#play-flow');
  try{
    const loaded=await Promise.all([1,2].map(async scenario=>{
      const root=`data/scenario${scenario}/`,response=await fetch(root+'metadata.json');if(!response.ok)throw new Error('Wind metadata unavailable');
      const meta=await response.json(),raw=await fetch(root+meta.data_file);if(!raw.ok)throw new Error('Wind field unavailable');
      const region=meta.monitoring_region_metrics;if(region)document.querySelector(`#monitoring-speeds-${scenario}`).textContent=`Mean wind in A: ${region.A.mean_m_s.toFixed(2)} m/s · B: ${region.B.mean_m_s.toFixed(2)} m/s`;
      const record=new StationaryFlow(meta,await raw.arrayBuffer()),container=document.querySelector(`#map${scenario}`),image=container.querySelector(`#s${scenario}-heat`);
      record.paint(image);
      const probe=document.querySelector(`#probe${scenario}`),svg=container.querySelector('svg');
      const showProbe=event=>{
        const box=svg.getBoundingClientRect(),scale=810/box.width,p=[(event.clientX-box.left)*scale-5,(event.clientY-box.top)*scale-5];
        if(p[0]<0||p[0]>800||p[1]<0||p[1]>462.5)return;
        const location=`(${(p[0]*.01).toFixed(2)}, ${(4.625-p[1]*.01).toFixed(2)}) m`;
        if(design.obstacles.some(o=>Math.hypot(p[0]-o.center[0],p[1]-o.center[1])<=o.radius)){probe.textContent=`${location} · solid obstacle`;return;}
        const v=record.sample(p);probe.textContent=`${location} · horizontal wind (${v[0].toFixed(2)}, ${v[1].toFixed(2)}) m/s · fixed`;
      };
      svg.addEventListener('pointermove',showProbe);svg.addEventListener('pointerdown',showProbe);
      const renderer=new WindAnimation({container,svg:container.querySelector('svg'),field:p=>record.at(p),bounds:design.bounds,obstacles:design.obstacles,count:500,seed:scenario*813});
      renderer.setSpeed(.25);return {record,renderer,image};
    }));
    button.disabled=false;document.querySelector('#restart-flow').disabled=false;document.querySelector('#flow-speed').disabled=false;
    const meta=loaded[0].record.meta;
    status.textContent='Wind is fixed in time · only the streamlets move · provisional CFD field, numerical validation incomplete';status.dataset.state='ready';
    document.querySelector('#computation-grid').textContent=meta.grid.join(' × ')+' cells';
    document.querySelector('#grid-spec').textContent=`The solver uses cubic cells of ${meta.dx_m} m and a time step of ${meta.dt_s} s.`;
    let playing=true;
    button.addEventListener('click',()=>{playing=!playing;for(const item of loaded)playing?item.renderer.play():item.renderer.pause();button.textContent=playing?'Ⅱ Pause particles':'▶ Animate particles';button.setAttribute('aria-pressed',String(playing));});
    document.querySelector('#restart-flow').addEventListener('click',()=>{for(const item of loaded)item.renderer.restart();});
    document.querySelector('#flow-speed').addEventListener('change',e=>{for(const item of loaded)item.renderer.setSpeed(Number(e.target.value));});
    document.querySelector('#wind').addEventListener('change',e=>{for(const item of loaded)item.renderer.setVisible(e.target.checked);});
    document.querySelector('#heatmap').addEventListener('change',e=>{for(const item of loaded)item.image.style.display=e.target.checked?'':'none';});
    document.addEventListener('visibilitychange',()=>{for(const item of loaded)(document.hidden||!playing)?item.renderer.pause():item.renderer.play();});
    for(const item of loaded)item.renderer.play();
    const response=await fetch('data/validation.json');if(response.ok){const v=await response.json();document.querySelector('#verification').textContent=v.summary;
      const s2=v.scenarios.find(s=>s.scenario===2);if(s2?.monitoring_regions){const before=s2.monitoring_regions_before,after=s2.monitoring_regions;document.querySelector('#shelter-wind-comparison').textContent=`Scenario 2, horizontal wind: A mean ${before.A.mean_m_s.toFixed(2)} → ${after.A.mean_m_s.toFixed(2)} m/s; B mean ${before.B.mean_m_s.toFixed(2)} → ${after.B.mean_m_s.toFixed(2)} m/s. Revised 95th percentiles: A ${after.A.p95_m_s.toFixed(2)}, B ${after.B.p95_m_s.toFixed(2)} m/s. Local maxima: A ${after.A.max_m_s.toFixed(2)}, B ${after.B.max_m_s.toFixed(2)} m/s. These are measurements of the computed horizontal slice.`;}
      const robot=await(await fetch('data/robot/manifest.json')).json(),upper=robot.cases.filter(c=>c.scenario===2&&c.route==='upper');const success=upper.length===2&&upper.every(c=>c.unfiltered.target_region_reached&&!c.unfiltered.contact);document.querySelector('#shelter-outcome').innerHTML=`<b>Scenario 2 · upper bypass</b><strong>${success?'Both directions reach the target':'Review the computed outcomes'}</strong><span>Without the safety filter · same reference and controller</span><span>Minimum clearance: ${Math.min(...upper.map(c=>c.unfiltered.min_clearance_m)).toFixed(3)} m · maximum control component: ${Math.max(...upper.map(c=>c.unfiltered.max_control_component_m_s2)).toFixed(2)} / 1.60 m/s²</span><a href="#robot-review">Inspect the robot runs ↓</a>`;}
  }catch(error){status.textContent=`Wind display could not load: ${error.message}`;status.dataset.state='error';}
})();
