"""Targeted teaching revision. Preserve archived results and the prior narrative."""
from pathlib import Path
import json, html
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/presentation/wind_aware_mbd'
BASE=ROOT/'tmp/slides/interactive/previous'
text=(BASE/'slides.qmd').read_text()

def replace_slide(title, body):
    global text
    start=text.index('## '+title)
    end=text.find('\n## ',start+4)
    if end<0:end=len(text)
    text=text[:start]+body.strip()+'\n'+text[end:]

poam=r'''
## POAM uses spatial correlation to connect observations {data-section="2 · Preliminaries"}

For one wind component, start with a Gaussian-process prior:

$$d(r)\sim\mathcal{GP}\!\left(0,k_\eta(r,r')\right).$$

$k_\eta(r,r')$ is the prior covariance between locations $r,r'$; $\eta$ denotes kernel parameters, separate from the policy parameters $\theta$.

**The attentive kernel combines several spatial length scales:**

$$k_\eta(r,r')=\sigma_f^2\,\big[\bar w(r)^{\mathsf T}\bar w(r')\big]
\sum_{b=1}^{B}\bar w_b(r)\bar w_b(r')\,
\exp\!\left(-\frac{\|r-r'\|^2}{2\ell_b^2}\right).$$

<div class="definition-grid"><div>$B$: number of length scales.<br>$\ell_b>0$: spatial length scale of component $b$.<br>$\sigma_f^2$: prior variance scale.</div><div>$\bar w(r)=w_\eta(r)/\|w_\eta(r)\|_2$:<br>normalized, nonnegative neural-network weights.<br>The weights vary with location.</div></div>

<div class="key-line">An observation influences other positions according to their spatial correlation. POAM can represent different correlation lengths in different regions.</div>
<div class="source">Chen, Liu and Khardon, POAM, RSS 2024, Sec. III-B. Equations here describe one scalar wind component.</div>

## POAM predicts wind through a sparse Gaussian posterior {data-section="2 · Preliminaries"}

Choose $M$ representative positions $Z=(z_1,\ldots,z_M)$, called **inducing positions**. Approximate the wind values $d_Z$ there by

$$d_Z\sim\mathcal N(m_Z,S_Z),\qquad K=k_\eta(Z,Z),\qquad \beta(r)=K^{-1}k_\eta(Z,r).$$

<div class="definition-grid"><div>$m_Z\in\mathbb R^M$: inducing-value mean.<br>$S_Z\in\mathbb R^{M\times M}$: inducing-value covariance.</div><div>$K$: prior covariance at the inducing positions.<br>$\beta(r)$: interpolation coefficients for location $r$.</div></div>

**Predicted mean and spatial covariance:**

$$\mu_d(r)=\beta(r)^{\mathsf T}m_Z$$
$$\Sigma_d(r,r')=\underbrace{k_\eta(r,r')-\beta(r)^{\mathsf T}K\,\beta(r')}_{\text{conditional residual}}+\beta(r)^{\mathsf T}S_Z\beta(r').$$

<div class="key-line">Retain the residual: uncertainty away from inducing positions is not captured by $S_Z$ alone.</div>
<div class="source">POAM, Eq. (4), written using interpolation coefficients $\beta(r)$. The coefficients interpolate from the inducing positions to any query location.</div>

## How a new observation changes the POAM posterior {data-section="2 · Preliminaries"}

At $r_m$, measure $y_m^d$ with scalar noise variance $\sigma_n^2>0$. First hold the kernel and inducing positions fixed. Write $\beta_m=\beta(r_m)$.

$$g_m=\frac{S_Z\beta_m}{\sigma_n^2+\beta_m^{\mathsf T}S_Z\beta_m}$$

$$m_Z^+=m_Z+g_m\big(y_m^d-\beta_m^{\mathsf T}m_Z\big),\qquad
S_Z^+=S_Z-g_m \beta_m^{\mathsf T}S_Z.$$

<div class="definition-grid"><div><b>Mean update:</b> $g_m$ is a gain vector. The difference $y_m^d-\beta_m^{\mathsf T}m_Z$ is the measurement innovation.</div><div><b>Covariance update:</b> the observation reduces uncertainty in correlated inducing values. Substitute $m_Z^+,S_Z^+$ into the preceding prediction equations.</div></div>

<div class="key-line">A zero innovation leaves the mean unchanged. A nonzero innovation can change future wind compensation and predicted motion.</div>

<div class="small-note">These are the Gaussian-likelihood sparse variational updates for a fixed representation. Full online POAM can also relocate inducing positions and fit kernel parameters; it then transports accumulated statistics. Planning holds the kernel parameters fixed.</div>
<div class="source">Equivalent fixed-representation form of POAM Eqs. (7)–(8), (19)–(21). It is not exact conditioning of the full GP. Apply per wind component.</div>

::: {.notes}
Let \beta_m=K^{-1}k(Z,r_m). For fixed Z, kernel and observation variance, the Gaussian variational precision update is S_plus^{-1}=S^{-1}+\beta_m \beta_m^T/sigma_n^2 and S_plus^{-1}m_plus=S^{-1}m+\beta_m y/sigma_n^2. The displayed rank-one form follows from Woodbury. This is equivalent to the additive unweighted statistics used in the implementation when its representation is fixed. It must not be confused with exact conditioning of the entire sparse predictive process: its conditional residual is retained in prediction but not added to sigma_n^2 in this variational update. Relocation requires statistic transport, and subsequent kernel learning changes the representation. Hypothetical updates hold kernel hyperparameters fixed; inducing relocation remains a separate configuration choice. In the implemented independent-component wind model, R_d=sigma_n^2 I. Primary source: https://www.roboticsproceedings.org/rss20/p019.pdf .
:::
'''
text=text.replace('\n## Why position alone is not enough', '\n'+poam+'\n## Why position alone is not enough',1)

# Keep every original frame; replacing timed GIF playback does not change its data.
frame_dir=OUT/'assets/mbd_frames';frame_dir.mkdir(exist_ok=True)
gif=Image.open(OUT/'assets/mbd_toy.gif')
frames=[]
for i in range(gif.n_frames):
    gif.seek(i)
    name=f'assets/mbd_frames/frame-{i:02d}.png'
    gif.convert('RGB').save(OUT/name,optimize=True)
    frames.append(f'<img class="mbd-frame" src="{name}" alt="MBD iteration {i//3+1}, phase {i%3+1}"'+('' if i==0 else ' hidden')+' />')

def controls(maximum,detail=False):
    return f'''<div class="mbd-controls" role="group" aria-label="Animation playback controls">
<button type="button" data-action="reset">Restart</button><button type="button" data-action="prev">Previous</button><button type="button" data-action="play" aria-pressed="false">Play</button><button type="button" data-action="next">Next step</button>
<input type="range" min="0" max="{maximum}" value="0" step="1" aria-label="{'Algorithm step' if detail else 'Animation frame'}" />
<output aria-live="polite"></output></div>'''

overview='''## Watch MBD refine paths in a simple environment {.replay #watch-mbd-refine-paths-in-a-simple-environment data-section="2 · Preliminaries"}

<div class="mbd-player mbd-overview" data-player="overview">
<div class="mbd-frame-stack">'''+''.join(frames)+'''</div>
'''+controls(23)+'''
</div>
<img class="print-poster" data-print-caption="Final iteration of the illustrative MBD example · The HTML version has Play, Previous, Next step and a timeline" src="assets/mbd_toy_poster.png" alt="Final iteration of the original MBD demonstration" />
<div class="caption">Starts paused. Use Next step or the timeline to move through all eight iterations.<br>Blue: evaluated paths · Orange dashed: weighted parameter mean · Green: best evaluated feasible path.</div>

::: {.notes}
The original 24 animation frames are preserved exactly as decoded from mbd_toy.gif. The player starts paused, pauses on leaving the slide, and never advances the presentation itself. The underlying example uses eight reverse levels, 64 proposals per level, temperature 0.12, seed 19, a bounded height domain [-2,2] and center clearance at least 0.6 around a radius-0.5 obstacle. It is a numerical teaching example, not an additional monitoring experiment. All original numerical records remain in assets/mbd_toy_data.json. The following slide expands one iteration into eight algorithm steps.
:::
'''
replace_slide('Watch MBD refine paths in a simple environment',overview)

steps=[
 ('Read the current noisy vector',r'y_i,\ \bar\alpha_i,\ \alpha_i'),
 ('Sample 64 path-height values',r'a^{[k]}\sim\mathcal N(y_i/\sqrt{\bar\alpha_i},(1-\bar\alpha_i)/\bar\alpha_i)'),
 ('Construct each path',r'r(s;a^{[k]}),\quad s\in[0,1]'),
 ('Evaluate length and clearance',r'J_k=\text{path length};\quad\text{reject infeasible paths}'),
 ('Normalize cost weights',r'w_k\propto\mathbf1_{\mathrm{feasible}}\exp(-J_k/\lambda)'),
 ('Compute the weighted mean',r'\bar a=\sum_k w_k a^{[k]}'),
 ('Estimate the score',r'\widehat s_i=(-y_i+\sqrt{\bar\alpha_i}\bar a)/(1-\bar\alpha_i)'),
 ('Denoise to the next level',r'y_{i-1}=(y_i+(1-\bar\alpha_i)\widehat s_i)/\sqrt{\alpha_i}'),
]
rows=''.join(f'<li class="algorithm-step" data-step="{i}"><span>{i+1}</span><b>{name}</b></li>' for i,(name,_) in enumerate(steps))
equations=''.join(f'<div class="mbd-equation" data-step="{i}"'+('' if i==0 else ' hidden')+f'><span class="latex-display" data-latex="{html.escape(eq,quote=True)}"></span></div>' for i,(_,eq) in enumerate(steps))
detail='''
## One MBD iteration, step by step {.mbd-detail-slide #mbd-one-iteration data-section="2 · Preliminaries"}

<div class="mbd-player mbd-detail" data-player="detail">
<div class="mbd-iteration-choice"><label>Choose an iteration <select aria-label="MBD iteration">'''+''.join(f'<option value="{i}"'+(' selected' if i==7 else '')+f'>{i+1} of 8 · diffusion level {8-i}</option>' for i in range(8))+'''</select></label><span>Each iteration uses the same eight steps.</span></div>
<div class="mbd-detail-grid"><div><svg class="mbd-stage" viewBox="0 0 850 425" role="img" aria-label="Path model and parameter-space calculation for the selected algorithm step"></svg></div><ol class="mbd-algorithm">'''+rows+'''</ol></div>
<div class="mbd-explanation" aria-live="polite"></div>
'''+equations+controls(7,True)+'''
</div>
<div class="caption">The highlighted line is the current algorithm step. Denoising updates a decision vector; it is not robot motion.</div>

::: {.notes}
The selector defaults to iteration 8, the final iteration, where the proposals concentrate on a path around the obstacle. Any of the eight saved iterations can be inspected independently. Step 1 displays the current noisy vector; 2 draws Gaussian decision values; 3 constructs paths; 4 tests feasibility and evaluates path length; 5 normalizes cost weights; 6 forms the weighted mean; 7 computes the score; 8 computes the next lower-noise iterate. The model evaluations are deterministic values from the saved demonstration; playback does not fabricate or rerun optimization. At early levels the mean can be infeasible; the best evaluated feasible path remains a separate object. The final next-level step is selected for the static PDF export.
:::
'''
text=text.replace('\n## Problem formulation', '\n'+detail+'\n## Problem formulation',1)

constraints=r'''
## Shared motion constraints and execution controller {data-section="3 · Problem formulation"}

Both problems require admissible policy parameters, actuator inputs and predicted states:

$$\theta\in\Theta,\qquad u(\tau)\in\mathcal U,\qquad
h_a(x(\tau))\leq0\quad\text{for every constraint }a.$$

$h_a$ encodes an environment or state constraint. For example, outside a circular exclusion region, $h_a(x)=R_a^2-\|\Psi(x)-c_a\|^2$, with center $c_a$ and radius $R_a$.

**The controller from the preliminaries determines the applied input:**

$$u=\underset{v\in\mathcal U}{\arg\min}\ \frac12\|v-u_{\mathrm{nom}}\|^2
\quad\text{s.t.}\quad \nabla_x V_a^{\mathsf T}f_d(x,v,d)+\alpha V_a\leq0.$$

The value $V_a$ and tested wind values $d$ depend on the available wind belief. The inequality is imposed for each constraint and tested wind value. Here $v$ is the trial input; $u$ is the selected input.

<div class="key-line">The planner evaluates the motion produced by this controller. It cannot choose the nominal route and assume that route is executed.</div>
<div class="small-note">The numerical method uses finite field samples and time checks. If the controller's inequalities are infeasible, its prescribed relaxation must be included in prediction. These tests do not establish safety for every possible wind field.</div>

::: {.notes}
The state and input constraints state the intended admissibility requirements. The policy-value inequality is an execution-controller rule; it is not a replacement for checking the resulting trajectory against h_a. The ideal requirements are separate from their finite numerical enforcement. The implementation can relax controller value-rate inequalities, and recorded results retain and report those interventions. Controller feasibility is therefore not silently equated with satisfaction of all continuous-time geometric constraints. The subsequent method derives the finite sampled value and controller in detail. All numerical feasibility choices belong in the implementation appendix.
:::
'''
text=text.replace('\n## Problem 1:', '\n'+constraints+'\n## Problem 1:',1)

replace_slide('Problem 1:',r'''
## Problem 1: wind-aware planning with a fixed predicted belief {data-section="3 · Problem formulation"}

**Choose a nominal policy that minimizes the monitoring deficit of its predicted execution under uncertain wind.**

$$\min_{\theta\in\Theta}\ \mathbb E_{d\mid\mathcal B_t^d}
\!\left[\int_0^Tq_d(q(\tau))\,d\tau\right]$$

subject to the dynamics, initial conditions and admissibility requirements:

$$\begin{aligned}
\dot x&=f_d(x,u,d(\Psi(x))), & x(0)&=x_t,\\
\dot q_j&=\frac{C_j(x)^2}{R}(1-q_j)^2-Q_jq_j^2, & q(0)&=q_t,\\
u&\in\mathcal U, &h_a(x)&\leq0\quad\forall a.
\end{aligned}$$

The controller on the preceding slide chooses $u$ from the request $u_{\mathrm{nom}}=\pi_\theta(x,\mathcal B_t^d,\tau)$.

$$\boxed{\mathcal B^d(\tau)=\mathcal B_t^d\quad\text{for }0\leq\tau\leq T}$$

<div class="small-note">A possible physical field $d$ is drawn from the current belief. Requirements apply to predicted executions; numerical enforcement checks the sampled fields and times. Real measurements still update the belief between planning calls.</div>
''')

replace_slide('Problem 2:',r'''
## Problem 2: wind-aware planning with anticipated learning {data-section="3 · Problem formulation"}

**Choose the policy while predicting how future observations change later inputs, motion and monitoring.**

$$\min_{\theta\in\Theta}\ \mathbb E_{d,\varepsilon^d\mid\mathcal B_t^d}
\!\left[\int_0^Tq_d(q(\tau))\,d\tau\right]$$

**Retain all motion and sensing constraints from Problem 1:**

$$\dot x=f_d(x,u,d(\Psi(x))),\quad
\dot q_j=\frac{C_j(x)^2}{R}(1-q_j)^2-Q_jq_j^2,$$
$$x(0)=x_t,\quad q(0)=q_t,\quad u\in\mathcal U,\quad h_a(x)\leq0\ \ \forall a.$$

The same controller chooses $u$, now from $u_{\mathrm{nom}}=\pi_\theta(x,\mathcal B^d,\tau)$. At a sensing time, $r_m=\Psi(x)$ and

$$y_m^d=d(r_m)+\varepsilon_m^d,\qquad
\boxed{\mathcal B^{d,+}=\operatorname{POAMUpdate}(\mathcal B^d;r_m,y_m^d)}.$$

<div class="key-line">Continue from the reached $x,q$ using the updated belief. The objective and admissibility requirements are unchanged.</div>
<div class="small-note">$\varepsilon_m^d\sim\mathcal N(0,R_d)$ is measurement noise. The belief starts at $\mathcal B_t^d$ and is retained between updates. The expectation includes fields and observation noise; there is no separate learning reward.</div>
''')

# Keep inducing positions Z distinct from future observation locations.
start=text.index('## Appendix: covariance-only anticipation')
end=text.index('\n## ',start+4)
part=text[start:end].replace('$Z$',r'$\mathcal R$').replace('r,Z',r'r,\mathcal R').replace('Z,r',r'\mathcal R,r').replace('Z,Z',r'\mathcal R,\mathcal R').replace('R_Z',r'R_{\mathcal R}')
text=text[:start]+part+text[end:]
(OUT/'slides.qmd').write_text(text)
base_css=(BASE/'theme.css').read_text()
extra_css=(ROOT/'docs/presentation/interactive.css').read_text()
(OUT/'theme.css').write_text(base_css+'\n'+extra_css)
# Local inline data and JS also work when the whole deck is embedded in one file.
data=json.loads((OUT/'assets/mbd_toy_data.json').read_text())
js=(ROOT/'docs/presentation/interactive.js').read_text()
(OUT/'controls.html').write_text((BASE/'controls.html').read_text()+'\n<script>\nwindow.mbdTeachingData='+json.dumps(data,separators=(',',':'))+';\n'+js+'\n</script>\n')
print(json.dumps({'slides':text.count('\n## '),'preserved_frames':gif.n_frames}))
