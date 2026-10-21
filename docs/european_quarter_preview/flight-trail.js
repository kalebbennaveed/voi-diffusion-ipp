import * as THREE from 'three';

// Recorded segments expanded in clip space: width stays legible at any zoom.
// The coordinates remain the measured flight positions; no path smoothing.
export function createFlightTrail(frames, color) {
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([
    0,-1,0, 1,-1,0, 1,1,0, 0,-1,0, 1,1,0, 0,1,0
  ],3));
  const starts=[], ends=[];
  for(let i=1;i<frames.length;i++) {
    const a=frames[i-1].p, b=frames[i].p;
    starts.push(a[0],a[2],a[1]); ends.push(b[0],b[2],b[1]);
  }
  geometry.setAttribute('segmentStart',new THREE.InstancedBufferAttribute(new Float32Array(starts),3));
  geometry.setAttribute('segmentEnd',new THREE.InstancedBufferAttribute(new Float32Array(ends),3));
  geometry.setAttribute('segmentIndex',new THREE.InstancedBufferAttribute(Float32Array.from({length:frames.length-1},(_,i)=>i),1));
  geometry.instanceCount=0;
  const material=new THREE.ShaderMaterial({
    uniforms:{resolution:{value:new THREE.Vector2(1,1)},startIndex:{value:0},width:{value:6.5},ink:{value:new THREE.Color(color)}},
    vertexShader:`
      attribute vec3 segmentStart, segmentEnd;
      attribute float segmentIndex;
      uniform float startIndex;
      uniform vec2 resolution;
      uniform float width;
      varying float edge;
      void main(){
        vec4 a=projectionMatrix*modelViewMatrix*vec4(segmentStart,1.0);
        vec4 b=projectionMatrix*modelViewMatrix*vec4(segmentEnd,1.0);
        // Discard segments behind the camera rather than stretching them across it.
        if(segmentIndex<startIndex || a.w<=0.0 || b.w<=0.0){gl_Position=vec4(2.,2.,2.,1.);edge=0.;return;}
        vec2 delta=(b.xy/b.w-a.xy/a.w)*resolution;
        vec2 direction=delta/max(length(delta),0.001);
        vec2 normal=vec2(-direction.y,direction.x);
        vec4 point=mix(a,b,position.x);
        point.xy+=normal*position.y*width/resolution*point.w;
        gl_Position=point;edge=position.y;
      }`,
    fragmentShader:`
      uniform vec3 ink;
      varying float edge;
      void main(){
        float a=1.0-smoothstep(.80,1.0,abs(edge));
        vec3 outline=mix(vec3(.98),ink,1.0-smoothstep(.61,.74,abs(edge)));
        gl_FragColor=vec4(outline,a*.94);
        #include <colorspace_fragment>
      }`,
    transparent:true,depthWrite:false,depthTest:true,toneMapped:false,
    side:THREE.DoubleSide
  });
  const mesh=new THREE.Mesh(geometry,material);mesh.frustumCulled=false;mesh.renderOrder=5;
  mesh.userData.showThrough=(index,start=0)=>{geometry.instanceCount=Math.max(0,Math.min(index,frames.length-1));material.uniforms.startIndex.value=start};
  mesh.onBeforeRender=renderer=>{const canvas=renderer.domElement;material.uniforms.resolution.value.set(canvas.clientWidth,canvas.clientHeight)};
  return mesh;
}
