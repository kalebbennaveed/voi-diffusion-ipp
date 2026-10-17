# Fixed-wing survey visualization

Run the same server as the original demo:

```sh
JAX_PLATFORMS=cpu uv run --no-sync python -m voi_diffusion_ipp.viz.demo_server --port 8765
```

Open <http://127.0.0.1:8765/3d/>. The original <http://127.0.0.1:8765/> remains intact.
The default is the complete 120-second proposed mission, seed 8101, paired with the
wind-blind nominal planner using wind-aware execution safety. The other comparator
is wind-aware planning without anticipated learning. Only that comparison isolates
anticipation; these remain development recordings, not held-out validation.

## Mission and aircraft

The visual use case is repeated floodplain surveys to keep surface-water information
current. Two wetlands, reeds and survey gauges give context to the original A/B targets.
The wetlands are static scenery. The experiment models a generic environmental
quantity through uncertainty and clarity; it does not simulate hydrology, water-depth
measurements, camera imagery or flood prediction. The default terrain has no colored
ROI fill. Thin dashed outlines mark the original survey target boundaries.

A licensed ASK 21 Mi fixed-wing airframe replaces the earlier helicopter, as requested.
Its placement follows the exact recorded horizontal state. This is a **planar replay**:
fixed-wing airspeed limits, lift, coordinated turns and stall are not enforced. The
replay can include low speeds and sharp turns. The airframe and its bank/pitch are
visual details, not a validated flight model. No planner jobs or new missions are run.

The two concrete pump-station buildings stay within the original conservative
circular obstacle footprints. Roofs, doors, upper windows, annexes and river pipework
make them part of the floodplain setting. The full elevated mesh has a
maximum radial extent of 0.482 display units inside the recorded 0.5 envelope; display scale
is proportional to that envelope. Low maintenance paths and pipes extend toward
the river below flight altitude. Roof ridges rise above the fixed 0.58 display-unit
sensing altitude. The planner has no climb action. No geographical scale conversion is
applied to recorded telemetry. Alpine chalets and barns are dispersed across the wider valley and hillsides,
with irregular spacing, isolated farmsteads and a few small clusters. Narrow,
curving tracks connect them; there is no perimeter road or square housing layout.
Timber balconies, shutters, broad pitched roofs and foundations seated on local
slopes provide architectural context. Buildings are static scenery outside the
modeled flight domain; their geometry is batched by material to limit draw calls.
Trees are excluded from the lots and each curved track segment. The **Whole valley**
camera shows the broader settlement; **Overview** remains closer to the mission. The settlement does not add
obstacles, observations or task targets to the retained experiment.
Surrounding terrain and vegetation are scenery;
central vegetation is below the aircraft. Safety metrics use recorded physical
clearance, not airframe or terrain mesh intersections.

## Controls and diagnostics

- **Play mission**, pause, time scrubber and speed controls use a shared clock.
- **View** selects a large proposed view, large comparator view or side by side.
  **Full screen** retains the top monitoring dashboard and all playback controls.
- **Chase camera** follows behind the aircraft; **Sensor camera** expands a downward
  view from its recorded position when explicitly selected. The always-visible
  downward-camera inset was removed at the user’s request.
  Overview and Top view reset the external camera. Drag to orbit, scroll to zoom.
  `&camera=aircraft` opens the chase camera; `&camera=sensor` opens the downward view.
  Old `camera=helicopter` links remain accepted.
- The translucent red fill fades with sensing weight; the solid red contour shows 50% of the peak **information weight**, H²/R. A faint outer red contour
  shows 10%. For the recorded Gaussian H, the normalized weight is exp(-r²/sigma²),
  so contour radius is sigma sqrt(-ln(weight)). These are not hard sensing cutoffs.
  Survey readouts count original target cells inside the half-strength contour.
  Footprints are draped on the terrain for display; image occlusion is not added to
  the underlying sensing model. Sensor images are rendered, not experimental data.
- **Clarity deficit** is displayed above the scene, with current and accumulated
  values for both methods. **Survey freshness** shows mean regional clarity, the
  target, and which cells currently receive at least half-strength sensing.
- A separate **whole-field clarity map** is always visible at the top left of each
  scene. **Clarity / Deficit** switches both maps together. Clarity uses the recorded
  q values on a fixed 0–1 scale; deficit is max(target-q,0), with the same scale
  across both arms. Original A/B outlines, obstacles, aircraft position and the red
  50% sensing contour orient the map. It updates during playback and scrubbing,
  independently of the optional terrain overlay. The right corner remains wind.
- Corner **POAM wind maps** always use the real recorded posterior. Background is
  component standard deviation on a common 0–1 m/s² scale; arrows are posterior mean
  disturbance acceleration. The aircraft location and original obstacle/ROI bounds
  are shown. The landscape can independently show posterior or physical wind arrows.
- **Denoising** freezes at an executed replanning instant and steps through real MBD
  levels. Violet paths are proposals, white is the mean, method color is incumbent,
  red indicates rejection; fallback use is explicitly labeled. **Final result** shows
  full executed paths and final fields. Forecasts are conditional, not actual motion.
- Ground layers optionally show survey clarity, whole-field clarity, deficit or POAM
  uncertainty, using fixed scales across methods. Natural terrain is the default.

## Visible wind effect

Orange arrows at the aircraft show the reconstructed interval-mean disturbance;
blue arrows show the recorded control acceleration. They share scale 0.20 display
units per m/s², capped at 0.65. Moving streaks across the scene and survey-site windsocks use the physical reference
field to illustrate wind direction, not calibrated air velocity. They remain animated
when mission playback is paused. The Wind flow button toggles that animation. The wind-effect panel reports wind, control, drag, net
acceleration and ground speed. Assistance/opposition is the wind projection onto
recorded ground travel, not an inference about learning benefit.

For the recorded RK4 dynamics, v_dot = u - lambda*v + d(p), recover the disturbance as

```
d_mean = (v[k+1] - v[k])/dt - u[k] + lambda*(p[k+1] - p[k])/dt.
```

The position increment is the same RK4-weighted velocity integral, so drag is handled
without using a coarse nearest-grid truth value or treating net acceleration as wind.
Export precision limits the numerical reconstruction. It is used only in the replay
visualization and is never supplied to POAM or the planner. At the final frame, the
last recorded interval is shown. The bank/pitch response to control is illustrative.

## Reproduction and provenance

Existing `/api/replay/<id>` records are unchanged. Clarity samples, trapezoidal cost,
causal POAM boundaries and MBD candidate arrays are reused exactly. POAM updates only
at recorded observation boundaries. Region statistics use all cells within the
recorded rectangles. Planar-to-scene mapping is `(x,y) -> (x,height,-y)`.

Three.js 0.170.0, OrbitControls, GLTFLoader and BufferGeometryUtils are locally vendored
under `demo/3d/vendor/three/` with their MIT license and SHA-256 provenance. Poly Haven
terrain/foliage/rock textures are CC0. The current airframe is by helijah under CC BY
4.0, obtained unchanged from SDRangel's glider.glb and renamed aircraft.glb. The earlier
helicopter asset and its credits remain retained. Asset sources, sizes and hashes are
in `demo/3d/assets/provenance.json`; attribution is in `ATTRIBUTION.txt` and the footer.
No CDN or remote assets are requested at runtime. WebGL 2 and ES modules are required.
Rendering is capped at 30 fps and pixel ratio 1.5; hidden tabs do not render. The optional full sensor
view shares the main WebGL context; there is no additional inset render pass.

```sh
node --test demo/model.test.mjs demo/3d/data.test.mjs
JAX_PLATFORMS=cpu uv run --no-sync pytest -q tests/test_demo_routes.py
```

Numeric tests cover the sensing contours, target coverage, disturbance reconstruction
with drag, camera visibility, terrain footprint/altitude constraints, coordinate mapping,
causal POAM boundaries, cost/endpoint agreement and rejected/fallback forecast handling.
Route tests preserve the 2D interface and restrict serving to named local assets.

## Verification of the survey revision

The revised page was visually checked with the fixed-wing airframe, red sensing fill,
pump-station buildings and river pipework, wetlands, optional full sensor camera, wind streaks
and windsocks, top clarity dashboard, both posterior corner maps, and split view.
The wind animation toggle was checked independently of mission playback. Recorded
denoising completed all four levels, and the current-belief fallback/rejected plan
at 12 seconds remained explicitly labeled. Final costs remained 2.839 proposed and
3.499 wind-blind; the isolated anticipation comparison still showed the original
3.9% development difference. Browser warning/error logs were empty. The JavaScript
suite passed 21 tests, static routes passed 14 tests, and all 13 local asset hashes
and sizes were verified. This verifies the replay software, not fixed-wing flight
dynamics, flood prediction, or new experimental evidence for the planning method.

The final pump-station meshes were additionally checked vertex by vertex after their
world transforms: 926 vertices each, maximum radial extent 0.481567 against the 0.5
recorded radius, with tops at 0.88635 and 0.89597 above the 0.58 sensing altitude.
Low pipework and maintenance paths are scenery and were excluded from that elevated
structure check. The two original modeled circles and all recordings remain unchanged.
