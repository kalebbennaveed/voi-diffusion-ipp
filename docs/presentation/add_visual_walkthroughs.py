"""Add teaching visuals without changing the archived experiments or existing slides."""
from pathlib import Path
import html
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/presentation/wind_aware_mbd'
BASE = ROOT / 'tmp/slides/visuals/previous'

# A fixed attentive kernel, exactly the form introduced in the POAM preliminaries.
# These illustrative weights are specified, not claimed to have been learned.
def weights(x):
    w = np.stack([.15 + 1/(1+np.exp((x-5)/.8)), .15 + 1/(1+np.exp(-(x-5)/.8))], axis=1)
    return w / np.linalg.norm(w, axis=1)[:, None]

def kernel(x, y):
    a, b = weights(x), weights(y)
    k = sum(a[:, j, None] * b[None, :, j] * np.exp(-.5*((x[:, None]-y[None, :])/ell)**2)
            for j, ell in enumerate([2.4, .8]))
    return .55**2 * (a @ b.T) * k

r = np.linspace(0, 10, 201)
z = np.linspace(0, 10, 13)
K = kernel(z, z) + np.eye(len(z))*1e-9
beta = np.linalg.solve(K, kernel(z, r))
residual = np.diag(kernel(r, r)) - np.sum(beta*(K@beta), axis=0)
m, S = np.zeros(len(z)), K.copy()
observations = [(2.2, .65), (5.4, -.45), (8.1, .32)]
noise = .1**2
states = []
for i in range(4):
    mean = beta.T @ m
    var = residual + np.sum(beta*(S@beta), axis=0)
    states.append(dict(mean=mean.tolist(), std=np.sqrt(np.maximum(0, var)).tolist(),
                       inducing_mean=m.tolist(), inducing_std=np.sqrt(np.diag(S)).tolist(),
                       mean_std=float(np.sqrt(var).mean())))
    if i < 3:
        pos, value = observations[i]
        b = np.linalg.solve(K, kernel(z, np.array([pos])))[:, 0]
        gain = S@b / (noise+b@S@b)
        m = m + gain*(value-b@m)
        S = S - np.outer(gain, b@S)
        S = (S+S.T)/2
        assert np.linalg.eigvalsh(S).min() > -1e-9
assert all(np.max(np.array(states[i+1]['std'])-states[i]['std']) < 1e-9 for i in range(3))
data = dict(position=r.tolist(), inducing=z.tolist(), observations=observations, states=states,
            noise_std=.1, residual=residual.tolist(),
            description='Illustrative 1D, fixed-kernel and fixed-inducing-position sparse variational updates. Not an archived mission.',
            kernel=dict(length_scales=[2.4,.8], amplitude=.55, weights='normalized logistic weights; see generator'))
(OUT/'assets/poam_teaching_data.json').write_text(json.dumps(data, separators=(',', ':')))

def controls(n):
    return f'''<div class="walk-controls" role="group" aria-label="Walkthrough playback">
<button type="button" data-action="reset">Restart</button><button type="button" data-action="prev">Previous</button><button type="button" data-action="play" aria-pressed="false">Play</button><button type="button" data-action="next">Next step</button>
<input type="range" min="0" max="{n-1}" value="0" step="1" aria-label="Walkthrough step"/><output aria-live="polite"></output></div>'''

def player(kind, steps, equations, print_step=0):
    rows=''.join(f'<li class="walk-step"><span>{i+1}</span><b>{s}</b></li>' for i,s in enumerate(steps))
    eqs=''.join(f'<div class="walk-equation"'+(' hidden' if i else '')+f'><span class="latex-display" data-latex="{html.escape(e,quote=True)}"></span></div>' for i,e in enumerate(equations))
    return f'''<div class="walk-player" data-walk="{kind}" data-print-step="{print_step}">
<div class="walk-status"></div><div class="walk-grid"><svg class="walk-stage" viewBox="0 0 880 430" role="img" aria-label="{kind} visual walkthrough"></svg><ol class="walk-algorithm">{rows}</ol></div>
<div class="walk-explanation" aria-live="polite"></div>{eqs}{controls(len(steps))}</div>'''

poam = '''## Watch POAM learn a spatial wind model {.walk-slide #poam-visual data-section="2 · Preliminaries"}

'''+player('poam', ['Start from a spatial prior', 'Receive one wind observation', 'Update the sparse posterior', 'Predict at an unobserved position', 'Assimilate a second observation', 'Assimilate a third observation'],[
 r'd_Z\sim\mathcal N(m_Z,S_Z),\qquad m_Z=0,\quad S_Z=K',
 r'y_m^d=d(r_m)+\varepsilon_m^d,\qquad \varepsilon_m^d\sim\mathcal N(0,\sigma_n^2)',
 r'm_Z^+=m_Z+g_m(y_m^d-\beta_m^{\mathsf T}m_Z),\quad S_Z^+=S_Z-g_m\beta_m^{\mathsf T}S_Z',
 r'\mu_d(r)=\beta(r)^{\mathsf T}m_Z,\quad \sigma_d(r)=\sqrt{\Sigma_d(r,r)}',
 r'\mathcal B_2^d=\operatorname{POAMUpdate}(\mathcal B_1^d;r_2,y_2^d)',
 r'\mathcal B_3^d=\operatorname{POAMUpdate}(\mathcal B_2^d;r_3,y_3^d)'],5)+'''
<div class="caption">Numerical teaching example in one spatial dimension. Kernel and inducing positions are fixed; the conditional residual is retained. Wind values use normalized units.</div>

::: {.notes}
This is a computed sparse variational Gaussian update using the equations immediately preceding it. The 13 inducing positions cover [0,10]. The attentive kernel combines length scales 2.4 and 0.8 with specified, normalized logistic weights; the weights are illustrative, not learned from a real dataset. Observation noise standard deviation is 0.1. Three observed values are assimilated sequentially. Blue bands show pointwise mean plus/minus two predictive standard deviations. Purple circles are observations; diamonds denote inducing positions, which are not additional observations. The final state retains the conditional residual. With this fixed representation, covariance decreases with each observation; adaptive POAM can also move inducing positions or refit its kernel, so the pointwise monotonicity illustrated here is not a general claim for every online adaptation. All numerical arrays are saved in assets/poam_teaching_data.json. Starts paused; controls work independently of slide navigation.
:::
'''

rollout = '''## Follow one policy's predicted execution {.walk-slide #proposed-policy-walkthrough data-section="4 · Method — interactive explanation"}

'''+player('rollout', ['Begin at the measured state', 'Fix one sampled policy vector', 'Predict the applied input', 'Advance motion and clarity', 'Generate a predicted observation', 'Update the predicted wind belief', 'Continue with the same parameters', 'Complete the monitoring prediction'],[
 r'x(0)=x_t,\qquad q(0)=q_t,\qquad \mathcal B^d(0)=\mathcal B_t^d',
 r'\theta=\theta^{[k]},\quad d=d^{[s]},\qquad u_{\mathrm{nom}}=\pi_\theta(x,\mathcal B^d,\tau)',
 r'u=\arg\min_{v\in\mathcal U}\tfrac12\|v-u_{\mathrm{nom}}\|^2\quad\text{subject to the value-rate inequalities}',
 r'\dot x=f_d(x,u,d(\Psi(x))),\qquad \dot q_j=\frac{C_j(x)^2}{R}(1-q_j)^2-Q_jq_j^2',
 r'r_m=\Psi(x(\tau_m)),\qquad y_m^d=d(r_m)+\varepsilon_m^d',
 r'\mathcal B^{d,+}=\operatorname{POAMUpdate}(\mathcal B^d;r_m,y_m^d)',
 r'u_{\mathrm{nom}}=\pi_\theta(x,\mathcal B^{d,+},\tau)\quad\text{with the same }\theta',
 r'J_s(\theta)=\int_0^T q_d(q^{[s]}(\tau))\,d\tau'],6)+'''
<div class="caption">Schematic explanation of the algorithm, not a new mission result. Orange: nominal motion · Blue: predicted execution · Purple: predicted observation.</div>

::: {.notes}
The diagram intentionally does not invent numerical monitoring gains. Each path and belief profile is schematic. Step 2 fixes one sampled nominal policy parameter vector and one possible physical wind field. Step 3 invokes the controller optimization and sampled design-policy value defined on the preceding slides; the smaller internal wind-field set constructs the value-rate constraints. The physical field for this trajectory remains coherent. Steps 3–4 recur at every propagation step. At scheduled update times, steps 5–7 assimilate observations from the predicted reached positions and continue from the reached robot and clarity states, with the same policy parameters. Future measurement values are generated from this trajectory's fixed physical field, not from a new independent wind value at every location. The controller receives the predicted belief, not privileged access to that field. Frozen-belief continuation is shown only to explain the difference; no ordering of costs is asserted. The integral uses the whole trajectory, including the prefix before learning. Actual physical execution occurs only after MBD finishes, as shown on the next walkthrough slide.
:::
'''

cycle = '''## Turn predicted monitoring costs into the next action {.walk-slide #proposed-mbd-walkthrough data-section="4 · Method — interactive explanation"}

'''+player('cycle', ['Evaluate every sampled policy vector', 'Average across possible wind fields', 'Normalize the policy weights', 'Compute the score and denoise', 'Repeat while diffusion levels remain', 'Execute, observe and start a new call'],[
 r'\theta^{[1]},\ldots,\theta^{[K]}\quad\longrightarrow\quad J_s(\theta^{[k]})',
 r'\widehat J(\theta^{[k]})=\frac1S\sum_{s=1}^{S}J_s(\theta^{[k]})',
 r'w_k\propto\mathbf1_{\Theta_{\mathrm{eval}}}(\theta^{[k]})e^{-\widehat J(\theta^{[k]})/\lambda},\qquad \bar\theta=\sum_k w_k\theta^{[k]}',
 r'\widehat s_i=\frac{-y_i+\sqrt{\bar\alpha_i}\bar\theta}{1-\bar\alpha_i},\qquad y_{i-1}=\frac{y_i+(1-\bar\alpha_i)\widehat s_i}{\sqrt{\alpha_i}}',
 r'i>1:\ \text{sample again at level }i-1;\qquad i=1:\ \text{finish optimization}',
 r'\theta^*=\arg\min_{\theta\ \text{evaluated and checked}}\widehat J(\theta)'],4)+'''
<div class="caption">Schematic algorithm walkthrough. The weighted mean updates MBD parameters; the selected evaluated policy is used for physical execution.</div>

::: {.notes}
The first walkthrough expands the expensive policy evaluation used here. The diagram uses indexed labels rather than invented cost values or fictitious measured improvements. Costs are averaged before exponentiation. Reusing common sampled physical fields and observation-noise draws across policy evaluations gives a consistent sampled objective. Feasibility and parameter-domain checks determine zero weights; execution-model relaxation must be reproduced as described in the method. Each denoising update returns to parameter sampling only if reverse levels remain. After the final level, select and check the best evaluated policy; the parameter weighted mean is not automatically an admissible executed trajectory. Execute only the next interval, assimilate real observations, and replan from the newly measured state. The real measurement update is separate from hypothetical learning inside the planner.
:::
'''

text=(BASE/'slides.qmd').read_text()
text=text.replace('\n## Why position alone is not enough', '\n'+poam+'\n## Why position alone is not enough',1)
text=text.replace('\n## Finish optimization before physical execution', '\n'+rollout+'\n'+cycle+'\n## Finish optimization before physical execution',1)
(OUT/'slides.qmd').write_text(text)
(OUT/'theme.css').write_text((BASE/'theme.css').read_text()+'\n'+(ROOT/'docs/presentation/walkthrough.css').read_text())
(OUT/'controls.html').write_text((BASE/'controls.html').read_text()+
 '<script>window.poamTeachingData='+json.dumps(data,separators=(',',':'))+';</script>\n<script>\n'+
 (ROOT/'docs/presentation/walkthrough.js').read_text()+'\n</script>\n')
print('Added POAM, policy evaluation and complete MBD cycle walkthroughs: 63 slides.')
