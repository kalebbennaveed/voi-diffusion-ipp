"""Rebuild the narrative: generic formulation and method, concrete choices in appendix."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'output/presentation/wind_aware_mbd'
old=(ROOT/'tmp/slides/redo/previous/slides.qmd').read_text()
parts=re.split(r'(?m)^## ',old);header=parts[0];sections=['## '+s for s in parts[1:]]
def take(title):return next(s for s in sections if s.startswith('## '+title))
new=[]
def add(s):new.append(s.strip()+'\n\n')
for title in ['{.cover','An informative plan','Robot state','Presentation structure','Preliminaries','Clarity measures','Monitoring quality','Represent the wind']:
 s=take(title)
 s=s.replace('J=\\int_0^{T_N}', 'J=\\int_0^T').replace('$\\tau$ is forecast time; $T_N=10\\,\\mathrm{s}$ is the planning horizon.', '$\\tau\\in[0,T]$ is elapsed prediction time; $T>0$ is the planning horizon.')
 s=s.replace('The planar implementation is introduced later when defining the actual backup policy.', 'A particular vehicle model is not needed to state the method; simulation choices are collected in the appendix.')
 add(s)
add(r'''
## Why position alone is not enough {data-section="2 · Preliminaries"}

**To predict a future constraint violation, we must specify both the dynamics and the future control inputs.**

Consider a robot moving toward a wall. Let $p$ be its position, with the wall at $p=0$.

$$h(x)=p,\qquad h(x)\leq0\ \text{is the permitted side.}$$

Two robots at $p=-1$ have the same $h$, but a faster robot needs more distance to stop.

<div class="two-points"><div><b>A geometric constraint</b><br>$h(x)$ tests the current state.</div><div><b>A policy value</b><br>Tests the largest future value of $h$ when a specified control law is followed.</div></div>

<div class="key-line">That specified control law is the <b>design policy</b>, denoted $\pi_b$. It makes the future-motion test well defined.</div>

<div class="source">Policy-value construction: Knoedler et al., <i>Safety on the Fly</i>, Sec. III-A.</div>
''')
add(r'''
## Example: evaluate the policy “brake until stopped” {data-section="2 · Preliminaries"}

**Illustrative model:** $x=(p,v)$, $\dot p=v$, $\dot v=u$, initially $v\geq0$.
Use $u=-a_b$ until $v=0$, then $u=0$, where $a_b>0$ is braking acceleration.

$$V(x)=\max_{\tau\geq0}p(\tau)=p+\frac{v^2}{2a_b}$$

<img class="wide-figure short braking" src="assets/design_policy_example.svg" alt="Two robots start one metre before a wall: braking at one metre per second squared stops the slower robot before the wall and the faster robot beyond it" />

<div class="definition-grid"><div><b>Case A:</b> $p=-1$, $v=1$, $a_b=1$.<br>$V=-0.5$: braking stops before the wall.</div><div><b>Case B:</b> $p=-1$, $v=2$, $a_b=1$.<br>$V=1$: the braking trajectory crosses the wall.</div></div>

<div class="key-line">The value depends on the chosen policy. Wind, actuator limits or a different braking law change this prediction.</div>

::: {.notes}
Units are metres, seconds and metres per second squared. This is an analytic teaching example, not the planar simulation model and not a claim about all possible control policies. The maximum includes the stopping state and remains there afterward. A positive value means this specific design-policy trajectory violates h<=0; it does not prove that every possible policy fails.
:::
''')
add(r'''
## From the example to a robust policy value {data-section="2 · Preliminaries"}

Let $\phi_b(\tau;x,d)$ be the state reached from $x$ after time $\tau$ when following $\pi_b$ under disturbance signal $d$.

$$V_a(x)=\sup_{d\in\mathscr D}\;\sup_{\tau\geq0}
 h_a\!\left(\phi_b(\tau;x,d)\right)$$

<div class="definition-grid"><div>$h_a$: constraint function for constraint $a$; violation means $h_a>0$.<br>$\mathscr D$: disturbance signals considered in the source formulation.</div><div><b>Inner supremum:</b> largest constraint value over time.<br><b>Outer supremum:</b> largest such value over disturbances.</div></div>

<div class="two-points"><div><b>$V_a(x)\leq0$</b><br>All trajectories included in this test satisfy constraint $a$.</div><div><b>$V_a(x)>0$</b><br>At least one trajectory in the test violates the constraint under this policy.</div></div>

<div class="source">RPCBF = robust policy control barrier function. Knoedler et al., Eqs. (6), (10). Their numerical construction uses a finite horizon, sampled disturbances and spline maxima; guarantees require additional conditions.</div>
''')
add(r'''
## A policy value can guide the applied input {data-section="2 · Preliminaries"}

The task policy requests $u_{\mathrm{nom}}$. The execution controller chooses the input $u$ closest to it while meeting a value-rate condition:

$$\min_{u\in\mathcal U}\ \frac12\|u-u_{\mathrm{nom}}\|^2
\quad\text{subject to}\quad
\nabla V_a(x)^{\mathsf T}f_d(x,u,d)+\alpha V_a(x)\leq0.$$

The condition is imposed for every constraint $a$ and every disturbance $d$ included in the controller's test.

<div class="definition-grid"><div>$\mathcal U$: actuator input set.<br>$\nabla V_a$: sensitivity of the value to state.<br>$\alpha>0$: rate coefficient.</div><div>If $u_{\mathrm{nom}}$ satisfies the conditions, it is applied unchanged.<br>Otherwise, the controller selects a different input.</div></div>

<div class="key-line">This changes motion and sensing. Our planner predicts that controller response when evaluating monitoring performance.</div>
<div class="source">RPCBF uses this construction for safety filtering. Our finite, sampled execution model is inspired by it; we do not claim its safety guarantees for the planner.</div>
''')
add(r'''
## MBD example: optimize one path-height parameter {data-section="2 · Preliminaries"}

::: {.columns}
::: {.column width="48%"}
**Model-based diffusion (MBD)** uses model-evaluated costs to refine decision vectors. First consider a scalar decision $a$: the height of a path between a fixed start and goal.

$$r(s;a)=\begin{bmatrix}-2+4s\\4a\,s(1-s)\end{bmatrix},\qquad s\in[0,1]$$

Here $s$ indexes progress along the path; it is not physical time. The midpoint has height $a$.

:::
::: {.column width="52%"}
<img class="figure" src="assets/mbd_toy_setup.svg" alt="A straight path intersects a central obstacle; larger path heights bend around it" />
:::
:::

<div class="key-line">Minimize path length, subject to clearance from the obstacle. Each sampled value of $a$ produces one path to evaluate.</div>
<div class="source">Teaching example only. The monitoring method will instead optimize a nominal policy parameter vector.</div>
''')
add(r'''
## MBD: prefer lower-cost decisions, then smooth the density {data-section="2 · Preliminaries"}

For a decision vector $y$ with model cost $J(y)$, define

$$p_0(y)\propto\mathbf1_{\mathcal Y}(y)\exp\!\left(-\frac{J(y)}{\lambda}\right).$$

$\mathcal Y$ is the feasible decision domain; $\lambda>0$ is temperature. The indicator $\mathbf1_{\mathcal Y}$ is one inside that domain and zero outside.

**Add Gaussian noise to obtain smoother densities:**

$$y_i=\sqrt{\bar\alpha_i}\,y+\sqrt{1-\bar\alpha_i}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).$$

<div class="definition-grid"><div>$i$ is a diffusion level, not physical time.<br>$\bar\alpha_i=\prod_{j=1}^{i}\alpha_j$, with $0<\alpha_j<1$.<br>$I$: identity matrix.</div><div>MBD moves from a noisy vector toward lower-noise levels.<br>It estimates the required direction using model evaluations, without training a score network.</div></div>

<div class="source">Pan et al., <i>Model-Based Diffusion for Trajectory Optimization</i>, Eqs. (2), (3), (6)–(9). The target requires a finite, positive normalizing integral.</div>
''')
add(r'''
## One MBD iteration: sample, evaluate and weight {data-section="2 · Preliminaries"}

**1. Sample $K$ decision vectors around the current noisy vector $y_i$: **

$$y^{[k]}\sim\mathcal N\!\left(\frac{y_i}{\sqrt{\bar\alpha_i}},\frac{1-\bar\alpha_i}{\bar\alpha_i}I\right),
\qquad k=1,\ldots,K.$$

**2. Evaluate each decision with the model, then form normalized weights:**

$$w_k=\frac{p_0(y^{[k]})}{\sum_{j=1}^{K}p_0(y^{[j]})},
\qquad \bar y=\sum_{k=1}^{K}w_k y^{[k]}.$$

<div class="two-points"><div><b>In the path-height example</b><br>Lower length gives greater weight. Paths violating clearance receive zero weight.</div><div><b>The result</b><br>$\bar y$ is a weighted mean of decision vectors. It need not itself be feasible in a nonconvex environment.</div></div>

<div class="source">$[k]$ indexes samples; $\bar y$ is the weighted decision-vector mean. Pan et al., Eqs. (8), (9).</div>
''')
add(r'''
## Use the weighted mean to refine the noisy vector {data-section="2 · Preliminaries"}

The **score** is the direction of increasing log density at diffusion level $i$.
Its Monte Carlo estimate is

$$\widehat s_i=\frac{-y_i+\sqrt{\bar\alpha_i}\,\bar y}{1-\bar\alpha_i}.$$

MBD uses this estimate in the next-level update:

$$y_{i-1}=\frac{y_i+(1-\bar\alpha_i)\widehat s_i}{\sqrt{\alpha_i}}.$$

<div class="numbered-text"><p><b>Repeat:</b> sample → evaluate model costs → compute weights → refine.</p><p><b>Finish:</b> inspect evaluated decisions at the end of the reverse levels.</p></div>

<div class="key-line">The update occurs in decision space. The model supplies the cost; no derivative of the rollout cost is required.</div>
<div class="source">Pan et al., Monte Carlo score ascent (MCSA), Eq. (6), with the model-based score in Eq. (9). A finite sampling budget gives an approximation.</div>

::: {.notes}
The Gaussian proposal is written consistently with Eq. (8). The paper's printed Algorithm 2 has an alpha-bar index discrepancy in line 3. The teaching example uses Eq. (8) throughout. If every weight is zero, an implementation must retain/recover a valid iterate; it cannot normalize a zero denominator. No global optimality or feasibility of the weighted mean is asserted.
:::
''')
add(r'''
## Watch MBD refine paths in a simple environment {.replay data-section="2 · Preliminaries"}

<div class="animation"><img class="result-gif" src="assets/mbd_toy.gif" alt="Numerically generated MBD iterations: Gaussian path-height samples, path-length weights and score updates around a central obstacle" /><img class="print-poster" data-print-caption="Final iteration of the illustrative MBD example · Open the HTML presentation for all iterations" src="assets/mbd_toy_poster.png" alt="Final illustrative MBD iteration, showing weighted parameter mean and best evaluated feasible path" /></div>
<div class="caption">Blue: evaluated paths. Orange dashed: weighted parameter mean. Green: best evaluated feasible path. Click to restart.</div>

::: {.notes}
This is a real numerical MBD demonstration, not an animation of invented iterates and not a monitoring result. One scalar path-height parameter lies in [-2,2]. A radius 0.5 obstacle has required center clearance 0.6. Cost is discretized path length. Seed 19; eight reverse levels, 64 Gaussian samples per level; temperature 0.12; alpha-bar geometrically spaced from 0.98 to 0.16. Retaining the best evaluated feasible path is shown separately from the score iterate. The example illustrates the optimization mechanism without assuming convergence to the global optimum. All samples, costs, weights and iterates are saved in assets/mbd_toy_data.json.
:::
''')
add(take('Problem formulation'))
add(r'''
## Define the decision and the planning horizon {data-section="3 · Problem formulation"}

At time $t$, the planner knows the robot state $x_t$, clarity vector $q_t$, wind belief $\mathcal B_t^d$, environment geometry and target clarity $\bar q$.

**Decision:** choose parameters $\theta\in\Theta$ of a nominal control policy:

$$u_{\mathrm{nom}}(\tau)=\pi_\theta\big(x(\tau),\mathcal B^d(\tau),\tau\big),
\qquad 0\leq\tau\leq T.$$

<div class="definition-grid"><div>$\Theta$: policy parameter domain.<br>$T$: planning horizon.<br>$\tau$: elapsed time from the start of this prediction.</div><div>$u_{\mathrm{nom}}$: input requested by the task policy.<br>$u$: input applied by the execution controller.<br>$x(0)=x_t$, $q(0)=q_t$.</div></div>

<div class="key-line">The decision is a nominal policy. Its predicted motion and sensing determine the monitoring cost.</div>

::: {.notes}
A reference trajectory is one possible implementation of pi_theta, not an additional decision object required by the generic formulation. It is introduced only in the implementation appendix. The same parameter vector defines the policy's response to state and belief throughout a prediction.
State, input and clarity depend on theta through that policy; repeated theta subscripts are omitted for readability.
:::
''')
add(r'''
## Problem 1: plan for monitoring under uncertain wind {data-section="3 · Problem formulation"}

**Choose a nominal policy whose predicted execution delivers the lowest expected cumulative clarity deficit.**

$$\min_{\theta\in\Theta}J_{\mathrm{WA}}(\theta),\qquad
J_{\mathrm{WA}}(\theta)=\mathbb E\!\left[\int_0^T q_d(q(\tau))\,d\tau\right].$$

<div class="numbered-text"><p><b>Predict control:</b> the execution controller responds to the requested input using the current wind belief.</p><p><b>Predict motion:</b> the applied input and uncertain wind determine the robot trajectory.</p><p><b>Predict monitoring:</b> that trajectory determines sensing and the clarity vector $q$.</p></div>

$$\mathcal B^d(\tau)=\mathcal B_t^d\qquad\text{throughout this prediction.}$$

<div class="small-note">The expectation is over possible physical wind fields under the current belief. The belief is held fixed only in the prediction; real measurements still update it between planning calls.</div>
''')
add(r'''
## Problem 2: include what future observations will change {data-section="3 · Problem formulation"}

**Choose a nominal policy while anticipating how wind observations affect later control, motion and monitoring.**

$$\min_{\theta\in\Theta}J_{\mathrm{AL}}(\theta),\qquad
J_{\mathrm{AL}}(\theta)=\mathbb E\!\left[\int_0^T q_d(q(\tau))\,d\tau\right].$$

<div class="numbered-text"><p><b>Before an observation:</b> predict motion using the belief available at that time.</p><p><b>At an observation:</b> predict the measured wind at the robot's reached position, then update the belief.</p><p><b>After the observation:</b> use the updated belief to predict the continuation from the reached state.</p></div>

<div class="key-line">The cost is unchanged. The difference is how the future clarity trajectory $q(\tau)$ is predicted.</div>
<div class="small-note">The expectation now includes wind fields and measurement noise. Learning is useful only through its effect on subsequent monitoring; there is no separate uncertainty-reduction reward.</div>
''')
add(r'''
## Method: from policy parameters to delivered monitoring {data-section="4 · Method — architecture"}

<img class="architecture architecture-main" src="assets/architecture.svg" alt="Planning loop: read current state, sample nominal policy parameters, predict execution and monitoring, score and refine. An explicit decision repeats diffusion if levels remain, otherwise executes the first interval and updates measured state." />

::: {.notes}
The top row evaluates sampled policy parameter vectors, not physical commands sent to the robot. The decision below refinement has two exclusive exits: continue optimization while diffusion levels remain; execute only once the optimization has finished. The green return path is a different loop: new real measurements start the next planning call. The following slides expand the prediction and cost blocks in order.
:::
''')
add(r'''
## First distinguish the requested input from the applied input {data-section="4 · Method — execution prediction"}

For a given parameter vector $\theta$, the nominal policy requests

$$u_{\mathrm{nom}}=\pi_\theta(x,\mathcal B^d,\tau).$$

The execution controller chooses $u\in\mathcal U$ using that request, the current state and the wind belief. The robot then evolves according to

$$\dot x=f_d(x,u,d(\Psi(x))).$$

<div class="two-points"><div><b>Example</b><br>The task policy requests motion toward a monitored region. Near an obstacle, wind and current velocity may require braking or turning.</div><div><b>Planning consequence</b><br>The robot may arrive later or sense different cells. Predicting only the requested motion misses that cost.</div></div>

<div class="key-line">Next we construct the execution controller's wind-dependent constraint test, then use its chosen input in the motion prediction.</div>
<div class="small-note">Here $d$ is a possible physical wind field. The controller receives the belief, not privileged knowledge of that field.</div>
''')
add(r'''
## Predict future constraint values under a fixed policy {data-section="4 · Method — execution prediction"}

Use the constraint functions $h_a$ and a fixed design policy $\pi_b$ from the preliminaries. MBD optimizes $\theta$, the parameters of the nominal task policy $\pi_\theta$; $\pi_b$ is fixed.

Draw $L$ wind fields $\widetilde d^{[\ell]}$ from the belief. Starting at $x$, follow $\pi_b$ under each field. Denote the resulting state trajectory by $\phi_b^{[\ell]}(\sigma)$.

**Take the largest constraint value over time, then over fields:**

$$\widehat V_a(x;\mathcal B^d)=\max_{\ell=1,\ldots,L}\;\max_{0\leq\sigma\leq H}
 h_a\!\left(\phi_b^{[\ell]}(\sigma)\right).$$

::: {.columns}
::: {.column width="58%"}
<img class="figure small" src="assets/policy_value_traces.svg" alt="Illustrative constraint traces under three wind fields: first find each time maximum, then select the largest across fields" />
:::
::: {.column width="42%"}
$H$: design-policy evaluation horizon.<br>$\sigma$: time within that prediction.<br>$\ell$: wind-field index.

The belief changes the tested trajectories and their largest constraint value.
:::
:::
<div class="source">Illustrative traces. Finite sampled policy-value construction inspired by Knoedler et al.; spline maxima approximate the time maximum.</div>
''')
add(r'''
## Predict which input the execution controller will apply {data-section="4 · Method — execution prediction"}

At the predicted state, choose the input closest to the nominal request:

$$u=\underset{v\in\mathcal U}{\arg\min}\ \frac12\|v-u_{\mathrm{nom}}\|^2$$

subject to, for every constraint $a$,

$$\max_{\ell=1,\ldots,L}\nabla_x\widehat V_a^{\mathsf T}
 f_d(x,v,\widetilde d^{[\ell]}(\Psi(x)))+\alpha\widehat V_a\leq0.$$

<div class="definition-grid"><div>$v$ is the trial control input in this optimization; $u$ is its selected value.<br>The belief is held fixed when taking the state gradient $\nabla_x\widehat V_a$.</div><div><b>If the nominal input satisfies every inequality:</b> $u=u_{\mathrm{nom}}$.<br><b>Otherwise:</b> the controller changes the input to meet the tested conditions, when feasible.</div></div>

<div class="key-line">Wind affects both the value $\widehat V_a$ and the predicted rate of change. It can therefore change the applied input even for the same nominal request.</div>
<div class="small-note">Control-affine dynamics give a quadratic program. If the inequalities are infeasible, the execution model must reproduce the controller's constraint relaxation; the planner does not assert safety.</div>
''')
add(r'''
## Propagate the applied input, then compute sensing {data-section="4 · Method — execution prediction"}

The chosen input $u$ and the possible physical wind field $d$ determine the next robot state:

$$\dot x=f_d(x,u,d(\Psi(x))),\qquad x(0)=x_t.$$

That state determines sensing quality and the clarity evolution:

$$\dot q_j=\frac{C_j(x)^2}{R}(1-q_j)^2-Q_jq_j^2,
\qquad q(0)=q_t.$$

<div class="numbered-text"><p><b>At each prediction step:</b> recompute the nominal request and the execution controller's response at the reached state.</p><p><b>Accumulate monitoring cost:</b> integrate the mean clarity deficit $q_d(q)$ along this predicted motion.</p></div>

<div class="key-line">The constraint value is part of the execution model. The objective is monitoring quality, not the size of the controller's intervention.</div>
''')
add(r'''
## Wind-aware evaluation: average costs across possible fields {data-section="4 · Method — wind-aware objective"}

One possible physical field gives one predicted trajectory and one monitoring cost. Use $S$ coherent fields $d^{[1]},\ldots,d^{[S]}$ drawn from the current belief.

$$J_s(\theta)=\int_0^T q_d(q^{[s]}(\tau))\,d\tau,
\qquad \widehat J_{\mathrm{WA}}(\theta)=\frac1S\sum_{s=1}^{S}J_s(\theta).$$

<div class="definition-grid"><div>$s$ indexes a possible physical field.<br>$q^{[s]}$ is the clarity predicted in that field.<br>Every evaluation starts from the same $x_t,q_t$.</div><div>The physical field stays fixed along its trajectory.<br>The controller uses the current belief throughout: $\mathcal B^d(\tau)=\mathcal B_t^d$.</div></div>

<div class="key-line">A short nominal route can have a higher cost if wind and controller intervention reduce its delivered sensing.</div>
<div class="small-note">The $S$ physical-field predictions evaluate monitoring. The $L$ short design-policy predictions on the previous slides construct the controller's constraint test; these have different roles.</div>
''')
add(r'''
## Anticipated learning: update, then continue {data-section="4 · Method — anticipated learning"}

<img class="anticipation" src="assets/anticipated_flow.svg" alt="Common motion prediction up to a measurement update; continue from the same reached state with either the retained belief or the updated belief" />

<div class="key-line">A predicted observation changes only the continuation after it becomes available. The robot state and clarity are carried forward.</div>

::: {.notes}
This diagram isolates the difference for a matched parameter vector and possible physical field. It is a conceptual explanation, not an assertion that the updated continuation always has lower cost. The observation schedule is part of the sensing model; the generic method does not depend on the particular two-second cadence used in the simulations.
:::
''')
add(r'''
## Predict one observation and its belief update {data-section="4 · Method — anticipated learning"}

At a scheduled sensing time, the predicted robot reaches position $r_m$. In the chosen possible physical field $d$, generate

$$y_m^d=d(r_m)+\varepsilon_m^d,\qquad
\varepsilon_m^d\sim\mathcal N(0,R_d).$$

Use that location and measurement in the same POAM update used during execution:

$$\mathcal B^{d,+}=\operatorname{POAMUpdate}\!\left(\mathcal B^d;\,r_m,y_m^d\right).$$

<div class="definition-grid"><div>$m$: observation index.<br>$y_m^d$: predicted noisy wind observation.<br>$R_d$: measurement-noise covariance.</div><div>$\mathcal B^d$: belief before the observation.<br>$\mathcal B^{d,+}$: belief after the observation.<br>A batch is handled using all locations observed since the last update.</div></div>

<div class="key-line">Example: an observation can reveal stronger crosswind than expected. Later predicted inputs then respond to that updated estimate.</div>
<div class="small-note">The observation is generated from the possible field used for this trajectory. The controller learns about it only through the predicted measurement.</div>
''')
add(r'''
## What changes after a predicted observation? {data-section="4 · Method — anticipated learning"}

From the reached state $x$, recompute the nominal request with the updated belief:

$$u_{\mathrm{nom}}=\pi_\theta(x,\mathcal B^{d,+},\tau).$$

<div class="numbered-text"><p><b>Nominal policy:</b> wind compensation and other belief-dependent control decisions respond to $\mathcal B^{d,+}$.</p><p><b>Execution controller:</b> recompute the design-policy value and input inequalities from $\mathcal B^{d,+}$.</p><p><b>Continuation:</b> propagate the selected input under the same physical field, evolve clarity and predict later observations.</p></div>

<div class="key-line">Keep the same policy parameters $\theta$. The policy responds to new information; there is no nested MBD solve at each predicted update.</div>
<div class="small-note">The anticipated-learning cost averages these complete monitoring costs over physical fields and observation noise. A detour to learn is preferred only if its later benefit outweighs its monitoring cost.</div>
''')
add(r'''
## MBD now optimizes the complete monitoring prediction {data-section="4 · Method — optimization"}

The MBD decision is the nominal policy parameter vector: $y=\theta$.
A **sampled policy parameter vector** $\theta^{[k]}$ defines one policy whose cost is evaluated by the preceding prediction.

$$w_k\propto \mathbf1_{\Theta_{\mathrm{eval}}}(\theta^{[k]})
\exp\!\left(-\frac{\widehat J(\theta^{[k]})}{\lambda}\right).$$

Here $\Theta_{\mathrm{eval}}\subseteq\Theta$ contains evaluable policy parameters satisfying the specified numerical feasibility checks. Normalize the weights to sum to one.

$$\bar\theta=\sum_k w_k\theta^{[k]}.$$

<div class="two-points"><div><b>Wind-aware:</b> $\widehat J=\widehat J_{\mathrm{WA}}$ uses the fixed-belief execution prediction.</div><div><b>Anticipated learning:</b> $\widehat J=\widehat J_{\mathrm{AL}}$ updates the belief during the prediction.</div></div>

<div class="key-line">Use $\bar\theta$ in the MBD score update from the preliminaries. Average costs across physical fields before exponentiating.</div>
<div class="small-note">Weights act on nominal parameters. Executed trajectories supply the costs; they do not replace parameters in the weighted mean.</div>
''')
add(r'''
## Finish optimization before physical execution {data-section="4 · Method — complete algorithm"}

<div class="algorithm-wide"><ol>
<li><b>Read current information:</b> robot state, clarity, wind belief, geometry and target clarity.</li>
<li><b>At each diffusion level:</b> sample nominal policy parameter vectors.</li>
<li><b>Evaluate each vector:</b> predict applied inputs, motion and clarity in possible wind fields. For anticipated learning, update the predicted belief when observations become available.</li>
<li><b>Refine:</b> average monitoring costs, compute parameter weights and apply the MBD score update. Repeat only while diffusion levels remain.</li>
<li><b>Execute:</b> after the final level, select and check the lowest-cost evaluated vector; apply its policy for the next execution interval.</li>
<li><b>Replan:</b> assimilate real measurements and start a new planning call from the measured state and belief.</li>
</ol></div>

<div class="key-line">End of method. The following comparison concerns MBD; concrete vehicle models and numerical settings are in the appendix.</div>
''')
for title in ['Comparison with MBD','Algorithm comparison','Relationship to the MBD paper']:
 s=take(title).replace('wind-dependent input modification','the wind-dependent execution controller').replace('predicts tracking,','predicts tracking,')
 s=s.replace('Predict tracking, the wind-dependent execution controller, motion and clarity.', 'Predict the applied inputs, motion and clarity under wind.')
 s=s.replace('the evaluated vector $\\theta_{\\mathrm{init}}$', 'an evaluated initial policy parameter vector $\\theta_{\\mathrm{init}}$')
 add(s)
# Preserve archived results, provenance and scientific limitations.
start=next(i for i,s in enumerate(sections) if s.startswith('## Results {'))
end=next(i for i,s in enumerate(sections) if s.startswith('## Appendix: covariance'))
for s in sections[start:end]:
 s=s.replace('Belief updated along the forecast','Belief updated during prediction').replace('10 s</b> forecasts','10 s</b> planning horizon')
 add(s)
add(r'''
## Implementation appendix {.section-break data-section="Appendix · Implementation"}

<div class="section-number">A</div>
<div class="section-subtitle">Concrete simulation choices, kept separate from the generic method.</div>
<div class="section-agenda">Policy representation → planar controller → numerical budgets → covariance-only variant</div>
''')
for title in ['Parameterize the nominal control policy','Predict nominal tracking','Evaluate a design policy','How the wind belief','Update future references','Optimize the nominal policy parameters','Appendix: covariance','Appendix: sources and reproducibility']:
 try:s=take(title)
 except StopIteration:
  print('no section',title);continue
 s=re.sub(r'data-section="[^"]*"','data-section="Appendix · Implementation"',s,count=1)
 s=s.replace('## Parameterize the nominal control policy','## Implementation: the 18-dimensional policy representation')
 s=s.replace('## Predict nominal tracking under estimated wind','## Implementation: a planar tracking controller')
 s=s.replace('## Evaluate a design policy near motion constraints','## Implementation: the velocity-damping design policy')
 s=s.replace('## How the wind belief changes the control constraints','## Implementation: the planar input inequality')
 s=s.replace('## Update future references from the predicted belief','## Implementation: belief-dependent reference positions')
 s=s.replace('## Optimize the nominal policy parameters with MBD','## Implementation: a bounded target for parameter search')
 s=s.replace('$\\Theta=[-6,6]^{18}$: bounded nominal parameter domain','$\\Theta=[-6,6]^{18}$: policy parameter domain')
 s=s.replace('$\\alpha_B\\widehat V_a', '$\\alpha\\widehat V_a')
 s=s.replace('\\alpha_B', '\\alpha').replace('\\widetilde d^{(\\ell)}','\\widetilde d^{[\\ell]}')
 s=s.replace('The implementation uses $\\theta=', 'A reference trajectory specifies desired positions for the tracking controller. The implementation uses $\\theta=')
 s=s.replace('The same parameter vector $\\theta$ is retained after each predicted belief update. Its matrix $G$ maps belief features to a reference-position adjustment.', 'The matrix $G$ in the policy parameter vector maps wind-belief features to a reference-position adjustment.')
 add(s)
# Use one consistent term for prediction throughout visible prose and notes.
out=header+''.join(new)
out=out.replace('forecast-interval','prediction-interval').replace('forecast time','prediction time').replace('forecast intervals','prediction intervals').replace('forecast interval','prediction interval').replace('forecasts','predictions').replace('forecast','prediction')
(D/'slides.qmd').write_text(out)
print('slides',len(new))
