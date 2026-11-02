'use strict';
// Sketch proportions assigned a physical scale of 0.01 m per display unit.
// Geometry is separate from the recorded CFD air velocity and from robot d(p).
const design = {
  status: 'Designed physical environment with stationary CFD-derived wind fields',
  coordinates: 'Display units of 0.01 m; origin upper-left, y downward',
  bounds: [[0,0],[800,462.5]],
  regions: [
    {label:'A', x:498, y:0, width:302, height:233},
    {label:'B', x:0, y:287, width:290, height:175.5}
  ],
  obstacles: [
    {center:[216,131.25],radius:98},
    {center:[373,199.25],radius:85.5},
    {center:[535.5,346.25],radius:85.5}
  ],
  geometryNote: 'Relative positions and sizes traced from the sketch. The first two circles overlap, as drawn, and form one connected obstacle cluster.',
  scenarios: [
    {id:1,central:'Wind toward the lower-left monitoring region',upper:'Sheltered bypass; computed speed',lower:'Sheltered bypass; computed speed'},
    {id:2,central:'Same central wind toward the lower-left monitoring region',upper:'Sheltered bypass; computed speed',lower:'Stronger incoming wind toward the lower-left; robot impassability not established'}
  ],
  airflowOpenings:{"scenarios": {"1": {"centralInletX": [5.6, 6.4], "lowerInletY": [0.35, 1.15], "floorOutlet": {"side": "floor", "x_m": [4.0, 4.9], "y_m": [1.35, 2.25], "z_m": 0.0}, "southOutletX": null}, "2": {"centralInletX": [6.3999999999999995, 7.2], "lowerInletY": [0.35, 1.15], "floorOutlet": {"side": "floor", "x_m": [4.0, 4.9], "y_m": [1.35, 2.25], "z_m": 0.0}, "southOutletX": [2.95, 4.45]}}},
  physicalScale:{metersPerLayoutUnit:.01,domainMeters:[8,4.625,2],obstacleHeightMeters:1.5}, windMagnitude:{"centralInletPeakByScenarioMps": {"1": 28.0, "2": 30.0}, "lowerInletPeakMps": 18}, windField:'Fixed vector mean from 3D MRT LBM / Smagorinsky LES; no time dependence',
  intendedRobotModel:'Nominal double integrator; separate robot review adds the existing relative-air-velocity quadratic force as acceleration.',
  illustrativePaths: {scenario1:'A -> central passage -> B -> lower bypass -> A',scenario2:'A -> central passage -> B -> upper bypass -> A',dynamicallyValidated:false},
  routes: {
    central:'M 650 110 C 600 155 555 208 505 242 C 465 268 425 300 374 331 C 308 368 224 375 145 375',
    upper:'M 650 110 C 600 70 535 37 470 28 C 386 15 330 16 250 16 C 170 12 110 18 82 64 C 52 105 53 205 76 280 C 90 317 110 350 145 375',
    lower:'M 650 110 C 680 175 700 246 695 309 C 693 365 641 430 575 444 C 536 454 483 449 431 440 C 337 425 231 403 145 375'
  }
};
const routeColors={central:'#c16a5d',upper:'#b49843',lower:'#3b9ba7'};
let openingDiagramId=0;
function renderWindOpenings(scenario){
  const openings=design.airflowOpenings.scenarios?.[scenario]||design.airflowOpenings;
  const segment=(bounds,side,color)=>side==='north'?`<path d="M${bounds[0]*100} 0H${bounds[1]*100}" stroke="${color}" stroke-width="5" stroke-dasharray="7 4"><title>Elevated central inlet: z = 1.5–2.0 m; robot slice z = 0.75 m</title></path>`:side==='east'?`<path d="M800 ${462.5-bounds[0]*100}V${462.5-bounds[1]*100}" stroke="${color}" stroke-width="5"/>`:`<path d="M${bounds[0]*100} 462.5H${bounds[1]*100}" stroke="${color}" stroke-width="5"/>`;
  let floor='';
  if(openings.floorOutlet){
    const f=openings.floorOutlet,id=`floor-opening-${++openingDiagramId}`,x=100*f.x_m[0],y=462.5-100*f.y_m[1],width=100*(f.x_m[1]-f.x_m[0]),height=100*(f.y_m[1]-f.y_m[0]);
    floor=`<defs><mask id="${id}" maskUnits="userSpaceOnUse" x="0" y="0" width="800" height="462.5"><rect width="800" height="462.5" fill="white"/>${design.obstacles.map(o=>`<circle cx="${o.center[0]}" cy="${o.center[1]}" r="${o.radius}" fill="black"/>`).join('')}</mask></defs><rect x="${x}" y="${y}" width="${width}" height="${height}" mask="url(#${id})" fill="#2a8d8312" stroke="#2a8d83" stroke-width="2" stroke-dasharray="5 5"><title>Floor pressure outlet at z = 0, below the robot plane. Solid column footprints remain closed.</title></rect>`;
  }
  return `<g class="air-openings" aria-label="Physical airflow openings; dashed north inlet is above and dashed floor outlet below the robot plane">${segment(openings.centralInletX,'north','#506f9a')}${scenario===2?segment(openings.lowerInletY,'east','#506f9a'):''}${openings.southOutletX?segment(openings.southOutletX,'south','#2a8d83'):''}${floor}</g>`;
}
function renderMap(scenario) {
  const id=`s${scenario}`;
  const zones = `<image id="${id}-heat" width="800" height="462.5" preserveAspectRatio="none"/>`;
  const guides=Object.entries(design.routes).map(([name,d])=>`<path data-route="${name}" d="${d}" fill="none" stroke="${routeColors[name]}" stroke-width="2" stroke-dasharray="4 5"/>`).join('');
  const selected=['central',scenario===1?'lower':'upper'];
  const examples=selected.map(name=>`<path d="${design.routes[name]}" fill="none" stroke="#fffffff0" stroke-width="6"/><path d="${design.routes[name]}" fill="none" stroke="#16704f" stroke-width="3" stroke-dasharray="9 6"/>`).join('');
  const regions=design.regions.map(r=>`<g><rect x="${r.x}" y="${r.y}" width="${r.width}" height="${r.height}" fill="#eef2e830" stroke="#3c6252" stroke-width="1.8"/><text x="${r.x+18}" y="${r.y+30}" class="region-label">Monitoring region ${r.label}</text></g>`).join('');
  const circles=design.obstacles.map(o=>`<circle cx="${o.center[0]}" cy="${o.center[1]}" r="${o.radius}" fill="#6a7a82"/>`).join('');
  return `<svg viewBox="-5 -5 810 472.5" role="img" aria-labelledby="${id}-title ${id}-desc" xmlns="http://www.w3.org/2000/svg">
    <title id="${id}-title">Scenario ${scenario}, following your sketch</title>
    <desc id="${id}-desc">Three circular obstacles. The first two overlap in the upper-left to center; the third sits lower-right. Large rectangular monitoring regions occupy the lower-left and upper-right corners. The narrow diagonal central passage has high wind toward the lower-left. ${scenario===1?'The computed field determines the wind in both outer bypasses.':'The lower bypass has a stronger incoming wind; actual passage conditions are computed, and robot impassability has not been evaluated.'} Dashed green lines are illustrative paths only.</desc>
    <defs>
      <clipPath id="${id}-mapclip"><rect width="800" height="462.5"/></clipPath>
      <marker id="${id}-windarrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 1L9 5L0 9" fill="none" stroke="#3f6e87" stroke-width="1.8"/></marker>
      <pattern id="${id}-hatch" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(36)"><path d="M0 0V10" stroke="#b98946" stroke-width="1.3" opacity=".7"/></pattern>
    </defs>
    <rect width="800" height="462.5" fill="#fdfefd"/>
    ${zones}${regions}
    <g class="route-layer">${guides}</g>
    <g class="obstacles">${circles}</g>
    <g class="example-layer">${examples}<path d="M-7 -5L7 0L-7 5L-4 0Z" transform="translate(447 281) rotate(142)" fill="#16704f" stroke="white" stroke-width="1"/>
      <path d="M-7 -5L7 0L-7 5L-4 0Z" transform="translate(${scenario===1?'510 449':'233 16'}) rotate(${scenario===1?'0':'0'})" fill="#16704f" stroke="white" stroke-width="1"/>
      <circle cx="650" cy="110" r="6" fill="#16704f" stroke="white" stroke-width="2"/>
    </g>
    <rect width="800" height="462.5" fill="none" stroke="#667b72" stroke-width="1.5"/>${renderWindOpenings(scenario)}
  </svg>`;
}
for(const scenario of [1,2])document.querySelector(`#map${scenario}`).innerHTML=renderMap(scenario);
function updateLayers(){
  for(const [control,layer] of [['paths','example-layer'],['routes','route-layer']]) {
    const hidden=!document.querySelector(`#${control}`).checked;
    document.querySelectorAll(`.${layer}`).forEach(el=>el.style.display=hidden?'none':'');
  }
}
document.querySelectorAll('input').forEach(input=>input.addEventListener('change',updateLayers));
updateLayers();
