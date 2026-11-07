# Stationary wind environment

The user's latest requirement is **time-independent wind**, not a uniform
nonzero velocity everywhere. The webpage loads one fixed vector field per
scenario. Particle integration visualizes that field without modifying it.

## Physical setup selected for the scenario design

- SI units; rectangular wind-tunnel style domain: 8 × 4.625 × 2 m.
- Three vertical circular columns, height 1.5 m. Their footprints follow the
  sketch, including the overlap of the first two columns. Exact coordinates
  are in `simulate_sketch_airflow.py` and each exported configuration.
- Isothermal air at 293.15 K; density 1.204 kg/m³; molecular kinematic
  viscosity 1.5061510649e−5 m²/s, obtained from Sutherland's law.
- Constant east inlet in direction (−1, −0.6, 0), normalized. A central
  Gaussian inlet-speed profile has peak 5 m/s; Scenario 2 adds a lower
  profile of peak 8 m/s. A sine variation over height vanishes at floor
  and roof. This is a prescribed external inflow, not an interior force
  painted onto route regions.
- West pressure outlet; no-slip floor, roof, and side walls.

These are designed physical parameters, not measurements inferred from the
sketch. The wind-tunnel geometry and inlet profile are explicit assumptions.

## CFD and the stationary field

The generator uses lbmpy 1.4.1, D3Q19 MRT lattice Boltzmann, and Smagorinsky
LES (Cs = 0.16). It is a finite-resolution, low-Mach, isothermal model of
3D flow. It is **not approximation-free, DNS, or experimentally validated**.
Molecular viscosity is not inflated. The LES closure models unresolved eddies.

Cylinder wall links use analytic intersections with the circular sides and
top planes. Bouzidi interpolation needs a second fluid node when q < 1/2.
At junctions where this is unavailable, the library-documented halfway
no-slip fallback is used, and every affected link is counted in metadata.
This explicitly recorded numerical treatment avoids invalid wall stencils.

The generator runs for 16 s under a constant inlet. The exporter defines the
stationary environment as the trapezoidal **vector** mean over 8–16 s at
exactly z = 0.75 m. Window endpoints and vertical slice height are interpolated.
The site never replays the transient solver sequence. A finite-window mean
does not establish an exact steady Navier–Stokes solution or a converged
statistical mean; sampling and mesh sensitivity are reported separately.

## Reproduction and checks

See the downloadable `airflow-source.zip` for the pinned environment and
compiler configuration. Run `simulate_sketch_airflow.py` for scenarios 1
and 2 on grids 64 and 128 with `--seconds 16 --sample .1`, then
`export_stationary_wind.py --nx 128` from the workspace root.

- `verify_flow_solver.py`: analytic Taylor–Green velocity decay and periodic
  mass conservation under grid refinement. This checks the solver, not the
  full obstacle case or its turbulence closure.
- `verify_stationary_adapter.cjs`: fixed values across animation times,
  spatial interpolation, SI/display conversion, exact zero preservation,
  and rejection of transient data.
- `stationary_validation.json`: matched-height/window grid sensitivity,
  8–12 versus 12–16 s sampling sensitivity, and boundary-link counts.

Older transient datasets are superseded; see `results/airflow/EXPERIMENT_STATUS.md`.
The 128-cell Scenario 2 computation failed its stability gate and was rejected.
The published preview retains both completed 64-cell fields and explicitly
reports this failure. It is not a validated final physical wind model.

## Robot and scenario scope

The robot remains deferred. Its requested model is ṗ = v, v̇ = u + d(p).
The CFD field is air velocity in m/s, not additive acceleration in m/s².
An explicit aerodynamic coupling would be required before any robot
feasibility or “impassable passage” conclusion. No controller, planner,
clarity dynamics, or inference model is changed by this wind visualization.
