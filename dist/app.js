'use strict';
// A layout transcription only: source-image proportions, no assumed physical scale.
// Wind zones communicate the user's design intent. They do not evaluate d(p).
const design = {
  status: 'User sketch transcribed for visual confirmation; no physical parameters selected',
  coordinates: 'Image-relative layout units; origin upper-left, y downward; not meters',
  bounds: [[0,0],[800,462]],
  regions: [
    {label:'A', x:498, y:0, width:302, height:233},
    {label:'B', x:0, y:287, width:290, height:175}
  ],
  obstacles: [
    {center:[216,131],radius:98},
    {center:[373,199],radius:85.5},
    {center:[535.5,346],radius:85.5}
  ],
  geometryNote: 'Relative positions and sizes traced from the sketch. The first two circles overlap, as drawn, and form one connected obstacle cluster.',
  scenarios: [
    {id:1,central:'High wind toward the lower-left monitoring region',upper:'No wind',lower:'No wind'},
    {id:2,central:'Same high wind toward the lower-left monitoring region',upper:'No wind',lower:'High wind; intended impassable, as marked in the sketch; direction unspecified'}
  ],
  physicalScale:null, windMagnitude:null, windField:null,
  intendedRobotModel:'p_dot = v; v_dot = u + d(p). Not simulated in this layout preview.',
  illustrativePaths: {scenario1:'A -> central passage -> B -> lower bypass -> A',scenario2:'A -> central passage -> B -> upper bypass -> A',dynamicallyValidated:false},
  routes: {
    central:'M 650 110 C 600 155 555 208 505 242 C 465 268 425 300 374 331 C 308 368 224 375 145 375',
    upper:'M 650 110 C 600 70 535 37 470 28 C 386 15 330 16 250 16 C 170 12 110 18 82 64 C 52 105 53 205 76 280 C 90 317 110 350 145 375',
    lower:'M 650 110 C 680 175 700 246 695 309 C 693 365 641 430 575 444 C 536 454 483 449 431 440 C 337 425 231 403 145 375'
  }
};
const routeColors={central:'#c16a5d',upper:'#b49843',lower:'#3b9ba7'};
function renderMap(scenario) {
  const id=`s${scenario}`;
  const zones = `<g class="wind-layer" clip-path="url(#${id}-mapclip)">
    <path d="M 508 240 L 374 344" fill="none" stroke="#c8e1ec" stroke-width="40" stroke-linecap="round"/>
    <g stroke="#3f6e87" stroke-width="2.4" marker-end="url(#${id}-windarrow)">
      <path d="M 497 242 l -37 29"/><path d="M 478 232 l -36 29"/>
      <path d="M 454 276 l -37 29"/><path d="M 435 266 l -36 29"/>
      <path d="M 412 309 l -25 20"/>
    </g>
    ${scenario===2?`<path d="M 617 277 A 107 107 0 0 1 460 422" fill="none" stroke="#ead6b4" stroke-width="64"/>
    <path d="M 617 277 A 107 107 0 0 1 460 422" fill="none" stroke="url(#${id}-hatch)" stroke-width="64"/>`:''}
  </g>`;
  const guides=Object.entries(design.routes).map(([name,d])=>`<path data-route="${name}" d="${d}" fill="none" stroke="${routeColors[name]}" stroke-width="2" stroke-dasharray="4 5"/>`).join('');
  const selected=['central',scenario===1?'lower':'upper'];
  const examples=selected.map(name=>`<path d="${design.routes[name]}" fill="none" stroke="#fffffff0" stroke-width="6"/><path d="${design.routes[name]}" fill="none" stroke="#16704f" stroke-width="3" stroke-dasharray="9 6"/>`).join('');
  const regions=design.regions.map(r=>`<g><rect x="${r.x}" y="${r.y}" width="${r.width}" height="${r.height}" fill="#eef2e8" stroke="#3c6252" stroke-width="1.8"/><text x="${r.x+18}" y="${r.y+30}" class="region-label">Monitoring region ${r.label}</text></g>`).join('');
  const circles=design.obstacles.map(o=>`<circle cx="${o.center[0]}" cy="${o.center[1]}" r="${o.radius}" fill="#6a7a82"/>`).join('');
  return `<svg viewBox="-5 -5 810 472" role="img" aria-labelledby="${id}-title ${id}-desc" xmlns="http://www.w3.org/2000/svg">
    <title id="${id}-title">Scenario ${scenario}, following your sketch</title>
    <desc id="${id}-desc">Three circular obstacles. The first two overlap in the upper-left to center; the third sits lower-right. Large rectangular monitoring regions occupy the lower-left and upper-right corners. The narrow diagonal central passage has high wind toward the lower-left. ${scenario===1?'Both outer bypasses have no wind.':'The lower bypass has high wind and is intended impassable; the upper bypass has no wind.'} Dashed green lines are illustrative paths only.</desc>
    <defs>
      <clipPath id="${id}-mapclip"><rect width="800" height="462"/></clipPath>
      <marker id="${id}-windarrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 1L9 5L0 9" fill="none" stroke="#3f6e87" stroke-width="1.8"/></marker>
      <pattern id="${id}-hatch" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(36)"><path d="M0 0V10" stroke="#b98946" stroke-width="1.3" opacity=".7"/></pattern>
    </defs>
    <rect width="800" height="462" fill="#fdfefd"/>
    ${zones}${regions}
    <g class="route-layer">${guides}</g>
    <g class="obstacles">${circles}</g>
    <g class="example-layer">${examples}<path d="M-7 -5L7 0L-7 5L-4 0Z" transform="translate(447 281) rotate(142)" fill="#16704f" stroke="white" stroke-width="1"/>
      <path d="M-7 -5L7 0L-7 5L-4 0Z" transform="translate(${scenario===1?'510 449':'233 16'}) rotate(${scenario===1?'0':'0'})" fill="#16704f" stroke="white" stroke-width="1"/>
      <circle cx="650" cy="110" r="6" fill="#16704f" stroke="white" stroke-width="2"/>
    </g>
    <g class="wind-layer zone-labels">
      <text x="375" y="65" text-anchor="middle" class="calm">No wind</text>
      <text x="394" y="312" text-anchor="middle" transform="rotate(-39 394 312)" class="central-label">High wind</text>
      ${scenario===1?'<text x="703" y="423" text-anchor="middle" class="calm">No wind</text>':'<text x="704" y="397" text-anchor="middle" class="strong-wind">High wind</text><text x="704" y="420" text-anchor="middle" class="blocked-label">Impassable*</text>'}
    </g>
    <rect width="800" height="462" fill="none" stroke="#667b72" stroke-width="1.5"/>
  </svg>`;
}
for(const scenario of [1,2])document.querySelector(`#map${scenario}`).innerHTML=renderMap(scenario);
function updateLayers(){
  for(const [control,layer] of [['wind','wind-layer'],['paths','example-layer'],['routes','route-layer']]) {
    const hidden=!document.querySelector(`#${control}`).checked;
    document.querySelectorAll(`.${layer}`).forEach(el=>el.style.display=hidden?'none':'');
  }
}
document.querySelectorAll('input').forEach(input=>input.addEventListener('change',updateLayers));
updateLayers();
