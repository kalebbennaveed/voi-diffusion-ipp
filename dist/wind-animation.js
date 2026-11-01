'use strict';
/**
 * Display streamlets in one stationary CFD-derived velocity field. The three-dimensional solver is separate.
 * The caller must supply an explicitly selected display-velocity field in layout
 * units / display-second. An acceleration disturbance must never be passed here
 * as if it were a physical air velocity. All scientific units belong to the
 * selected field adapter; no wind or physical assumptions are made by this file.
 */
class WindAnimation {
  constructor({container, svg, field, bounds, obstacles, count=260, color='#2f7d9c', seed=1}) {
    if(typeof field!=='function')throw new TypeError('An explicitly selected display field is required.');
    this.container=container;this.svg=svg;this.field=field;this.bounds=bounds;this.obstacles=obstacles;
    this.color=color;this.initialSeed=seed;this.seed=seed;this.time=0;this.running=false;this.speed=1;this.visible=true;
    this.canvas=document.createElement('canvas');this.canvas.className='wind-particles';this.canvas.setAttribute('aria-hidden','true');
    Object.assign(this.canvas.style,{position:'absolute',left:'0',top:'0',width:'100%',height:'100%',pointerEvents:'none'});
    container.append(this.canvas);this.ctx=this.canvas.getContext('2d');this.particles=Array.from({length:count},()=>this.spawn());
    this.resizeObserver=new ResizeObserver(()=>this.resize());this.resizeObserver.observe(container);this.resize();
    this.frame=this.frame.bind(this);this.handle=null;this.previous=null;this.accumulator=0;
  }
  random(){this.seed=(1664525*this.seed+1013904223)>>>0;return this.seed/4294967296;}
  free(p){
    return p[0]>=this.bounds[0][0]&&p[0]<=this.bounds[1][0]&&p[1]>=this.bounds[0][1]&&p[1]<=this.bounds[1][1]&&
      this.obstacles.every(o=>Math.hypot(p[0]-o.center[0],p[1]-o.center[1])>o.radius+1);
  }
  validSegment(a,b){
    if(!this.free(b))return false;
    const dx=b[0]-a[0],dy=b[1]-a[1],len=dx*dx+dy*dy;
    return this.obstacles.every(o=>{
      const t=len?Math.max(0,Math.min(1,((o.center[0]-a[0])*dx+(o.center[1]-a[1])*dy)/len)):0;
      return Math.hypot(a[0]+t*dx-o.center[0],a[1]+t*dy-o.center[1])>o.radius+1;
    });
  }
  spawn(){
    for(let k=0;k<500;k++){
      const p=this.bounds[0].map((v,i)=>v+this.random()*(this.bounds[1][i]-v));
      if(!this.free(p))continue;const v=this.field(p);
      if(v&&v.every(Number.isFinite)&&Math.hypot(...v)>4)return {p,age:this.random()*3,lifetime:3+this.random()*5,trail:[]};
    }
    return {p:null,age:0,lifetime:0,trail:[]};
  }
  velocity(p){const v=this.field(p);return v&&v.every(Number.isFinite)?v:[0,0];}
  integrate(p,h){
    const k1=this.velocity(p),k2=this.velocity(p.map((v,i)=>v+h*k1[i]/2));
    const k3=this.velocity(p.map((v,i)=>v+h*k2[i]/2)),k4=this.velocity(p.map((v,i)=>v+h*k3[i]));
    return p.map((v,i)=>v+h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6);
  }
  step(h){
    for(let i=0;i<this.particles.length;i++){
      const particle=this.particles[i];
      if(!particle.p||particle.age>particle.lifetime){this.particles[i]=this.spawn();continue;}
      const next=this.integrate(particle.p,h);
      // Crossing an obstacle ends a display trace; it never invents reflected flow.
      if(!this.validSegment(particle.p,next)||Math.hypot(...this.velocity(next))<.05){this.particles[i]=this.spawn();continue;}
      particle.trail.push({p:particle.p,time:this.time});particle.p=next;particle.age+=h;
      while(particle.trail.length&&particle.trail[0].time<this.time-.12)particle.trail.shift();
    }
    this.time+=h;
  }
  resize(){
    const box=this.container.getBoundingClientRect(),s=this.svg.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
    this.canvas.width=Math.round(box.width*dpr);this.canvas.height=Math.round(box.height*dpr);
    this.transform={dpr,width:box.width,height:box.height,scale:s.width/810,left:s.left-box.left+5*s.width/810,top:s.top-box.top+5*s.width/810};
    this.draw();
  }
  draw(){
    if(!this.transform)return;
    const c=this.ctx,m=this.transform;c.setTransform(m.dpr,0,0,m.dpr,0,0);c.clearRect(0,0,m.width,m.height);
    if(!this.visible)return;
    c.save();c.translate(m.left,m.top);c.scale(m.scale,m.scale);
    c.beginPath();c.rect(this.bounds[0][0],this.bounds[0][1],this.bounds[1][0]-this.bounds[0][0],this.bounds[1][1]-this.bounds[0][1]);
    for(const o of this.obstacles){c.moveTo(o.center[0]+o.radius,o.center[1]);c.arc(...o.center,o.radius,0,Math.PI*2,true);}c.clip('evenodd');
    c.lineCap='round';c.strokeStyle=this.color;c.fillStyle=this.color;
    for(const particle of this.particles){if(!particle.p)continue;
      const life=Math.min(1,particle.age/.35,(particle.lifetime-particle.age)/.5);c.globalAlpha=Math.max(0,life)*.8;
      const magnitude=Math.hypot(...this.velocity(particle.p));c.strokeStyle=magnitude>250?'#f3fcff':'#12617e';c.fillStyle=c.strokeStyle;c.lineWidth=1.8;c.beginPath();const trail=particle.trail;
      if(trail.length){c.moveTo(...trail[0].p);for(let j=1;j<trail.length;j++)c.lineTo(...trail[j].p);c.lineTo(...particle.p);c.stroke();}
      c.beginPath();c.arc(...particle.p,1.3,0,Math.PI*2);c.fill();
    }
    c.restore();
  }
  frame(now){
    if(!this.running)return;
    if(this.previous!==null)this.accumulator+=Math.min((now-this.previous)/1000,.1)*this.speed;
    this.previous=now;
    while(this.accumulator>=1/120){this.step(1/120);this.accumulator-=1/120;}
    this.draw();this.handle=requestAnimationFrame(this.frame);
  }
  play(){if(this.running)return;this.running=true;this.previous=null;this.handle=requestAnimationFrame(this.frame);}
  pause(){this.running=false;if(this.handle!==null)cancelAnimationFrame(this.handle);this.handle=null;this.previous=null;}
  restart(){this.seed=this.initialSeed;this.time=0;this.accumulator=0;this.previous=null;this.particles=this.particles.map(()=>this.spawn());this.draw();}
  setSpeed(speed){if(!Number.isFinite(speed)||speed<=0||speed>4)throw new RangeError('Display speed must be in (0,4].');this.speed=speed;}
  setVisible(visible){this.visible=Boolean(visible);this.draw();}
  destroy(){this.pause();this.resizeObserver.disconnect();this.canvas.remove();}
}
