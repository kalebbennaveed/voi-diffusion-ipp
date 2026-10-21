"""Clarify notation and expose every part of the saved MBD denoising update.

Run against the 63-slide snapshot; no optimization or mission is rerun.
"""
from pathlib import Path
import html
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/presentation/wind_aware_mbd'
BASE = ROOT / 'tmp/slides/denoising/previous'
s = (BASE / 'slides.qmd').read_text()

def section(title, body):
    global s
    start = s.index('## ' + title)
    end = s.find('\n## ', start + 3)
    if end < 0:
        end = len(s)
    s = s[:start] + body.strip() + '\n\n' + s[end:].lstrip('\n')

section('Represent the wind', r'''
## Represent the wind that the robot currently knows {.notation-slide data-section="2 · Preliminaries"}

At physical time $t$, the robot has noisy observations of the unknown physical wind field $\bar d$:

$$y_m^d=\bar d(r_m)+\varepsilon_m^d,\qquad
\varepsilon_m^d\sim\mathcal N(0,R_d).$$

<div class="definition-grid"><div>$m$: observation index; $r_m$: its position.<br>$y_m^d$: observed wind vector.<br>$\varepsilon_m^d$: zero-mean measurement noise.</div><div>$\mathcal N(\mu,\Sigma)$: Gaussian distribution with mean $\mu$ and covariance $\Sigma$.<br>$R_d$: measurement-noise covariance.</div></div>

The **wind belief** $\mathcal B_t^d$ summarizes what those observations imply:

$$\mathcal B_t^d=\{\mu_d(\cdot,t),\Sigma_d(\cdot,\cdot,t)\}.$$

<div class="definition-grid"><div>$\mu_d(r,t)$: predicted wind mean at position $r$.<br>$\Sigma_d(r,r',t)$: covariance between wind at $r$ and $r'$.</div><div>$r'$: a second position; $\cdot$: a position argument.<br>The label $d$ denotes wind; it is not a power.<br>Covariance expresses remaining uncertainty.</div></div>

<div class="key-line">POAM — Probabilistic Online Attentive Mapping — updates this spatial belief as observations arrive.</div>
<div class="source">Chen, Liu and Khardon, POAM, RSS 2024. The next slides explain one scalar wind component; vector components are modeled separately here.</div>
''')

section('POAM uses spatial correlation', r'''
## POAM uses spatial correlation to connect observations {.notation-slide data-section="2 · Preliminaries"}

For one scalar wind component, use a **Gaussian process (GP)**: any finite collection of wind values has a joint Gaussian distribution.

$$d(r)\sim\mathcal{GP}\!\left(0,k_\eta(r,r')\right).$$

<div class="definition-grid"><div>$d(r)$: uncertain wind component at $r$.<br>$0$: prior mean, before observations.<br>$k_\eta(r,r')$: prior covariance between positions.</div><div>$\eta$: kernel parameters.<br>The kernel controls how an observation at one position influences predictions elsewhere.</div></div>

The **attentive kernel** combines spatial correlation lengths:

$$k_\eta(r,r')=\sigma_f^2\,[\bar w(r)^{\mathsf T}\bar w(r')]
\sum_{b=1}^{B}\bar w_b(r)\bar w_b(r')\,
\exp\!\left(-\frac{\|r-r'\|_2^2}{2\ell_b^2}\right).$$

<div class="definition-grid"><div>$B$: number of kernel components; $b$: component index.<br>$\ell_b>0$: correlation length of component $b$.<br>$\sigma_f^2$: prior variance scale.</div><div>$\bar w_b(r)$: normalized component weight at $r$.<br>$\mathsf T$: transpose; $\|\cdot\|_2$: Euclidean norm.<br>Larger $\ell_b$ links more distant positions.</div></div>

<div class="source">Chen, Liu and Khardon, POAM, Sec. III-B. The following slide defines the location-dependent weights.</div>

## What the attentive weights mean {.notation-slide data-section="2 · Preliminaries"}

At each position, a neural network produces a nonnegative vector $w_\eta(r)\in\mathbb R^B$. Normalize it to obtain

$$\bar w(r)=\frac{w_\eta(r)}{\|w_\eta(r)\|_2},\qquad
\bar w_b(r)=\text{component }b\text{ of }\bar w(r).$$

<div class="two-points"><div><b>Which length scales apply here?</b><br>Large $\bar w_b(r)$ gives component $b$ more influence at position $r$.</div><div><b>How similar are the two regions?</b><br>$\bar w(r)^{\mathsf T}\bar w(r')$ reduces correlation when their weight vectors differ.</div></div>

**Example with two components:** one long correlation length describes smooth wind; a shorter one describes more rapid spatial variation. The weights can favor different components in different regions.

<div class="key-line">These are kernel weights over spatial length scales. Later, MBD uses different weights over sampled decisions.</div>
<div class="source">$\mathbb R^B$: vectors with $B$ real entries. $\eta$ includes the network and kernel parameters. The numerical visual uses specified illustrative weights.</div>
''')

section('POAM predicts wind', r'''
## POAM predicts wind through a sparse Gaussian posterior {.notation-slide data-section="2 · Preliminaries"}

Choose $M$ representative **inducing positions** $Z=(z_1,\ldots,z_M)$. They are model locations, not additional observations. Collect their wind values in $d_Z=[d(z_1),\ldots,d(z_M)]^{\mathsf T}$.

$$d_Z\sim\mathcal N(m_Z,S_Z),\qquad K_{ZZ}=k_\eta(Z,Z),\qquad
\beta(r)=K_{ZZ}^{-1}k_\eta(Z,r).$$

<div class="definition-grid"><div>$m_Z$: mean vector of the $M$ inducing values.<br>$S_Z$: their $M\times M$ posterior covariance.<br>$K_{ZZ}$: prior covariance matrix at $Z$.</div><div>$k_\eta(Z,r)$: vector of covariances from $Z$ to $r$.<br>$\beta(r)$: interpolation-coefficient vector.<br>$K_{ZZ}^{-1}$: matrix inverse (computed by a solve).</div></div>

$$\mu_d(r)=\beta(r)^{\mathsf T}m_Z$$
$$\Sigma_d(r,r')=\underbrace{k_\eta(r,r')-\beta(r)^{\mathsf T}K_{ZZ}\beta(r')}_{\text{conditional residual}}+\beta(r)^{\mathsf T}S_Z\beta(r').$$

<div class="key-line">The residual retains uncertainty that the finite inducing values cannot represent. Local wind standard deviation is $\sigma_d(r)=\sqrt{\Sigma_d(r,r)}$.</div>
<div class="source">POAM, Eq. (4). Time $t$ is suppressed on these slides: every posterior is the current one.</div>
''')

section('How a new observation', r'''
## How a new observation changes the POAM posterior {.notation-slide data-section="2 · Preliminaries"}

At $r_m$, observe $y_m^d$. For this scalar component, the measurement-noise variance is $\sigma_n^2>0$ (the scalar counterpart of $R_d$). Hold the kernel and inducing positions fixed.

$$\beta_m=\beta(r_m),\qquad
g_m=\frac{S_Z\beta_m}{\sigma_n^2+\beta_m^{\mathsf T}S_Z\beta_m}.$$

<div class="definition-grid"><div>$\beta_m$: interpolation coefficients at the observation.<br>$g_m$: gain vector, determining how much each inducing mean responds.</div><div>$y_m^d-\beta_m^{\mathsf T}m_Z$: <b>innovation</b>, the measured wind minus its predicted mean.<br>Superscript $+$ means after this observation.</div></div>

$$m_Z^+=m_Z+g_m\big(y_m^d-\beta_m^{\mathsf T}m_Z\big),\qquad
S_Z^+=S_Z-g_m\beta_m^{\mathsf T}S_Z.$$

<div class="two-points"><div><b>Mean</b><br>A nonzero innovation changes the wind estimate. A zero innovation leaves it unchanged.</div><div><b>Covariance</b><br>Uncertainty decreases in correlated inducing values. Recompute $\mu_d$ and $\Sigma_d$ from $m_Z^+,S_Z^+$.</div></div>

<div class="source">Fixed-representation sparse variational Gaussian update; POAM Eqs. (7)–(8), (19)–(21). Full online POAM can also refit its kernel and relocate inducing positions.</div>

::: {.notes}
This is the fixed-representation Gaussian-likelihood variational update, not exact conditioning of the entire predictive GP. The conditional residual remains in prediction and is not inserted into this variational gain denominator. With independent identically noisy wind components, R_d = sigma_n^2 I, where I is the identity. Kernel refitting and inducing relocation are separate changes of representation; pointwise variance monotonicity is only asserted with that representation fixed.
:::
''')

# Disambiguate the controller coefficient from the MBD diffusion coefficient.
s = s.replace(r'\alpha V_a', r'\kappa V_a').replace(r'\alpha\widehat V_a', r'\kappa\widehat V_a')
s = s.replace(r'$\alpha>0$: rate coefficient.', r'$\kappa>0$: controller value-rate coefficient.')

section('MBD: prefer lower-cost', r'''
## MBD: prefer lower-cost decisions, then smooth the density {.notation-slide data-section="2 · Preliminaries"}

Let $y$ denote a decision vector. In the path example, $y=a$ is a scalar path height. The model evaluates its cost $J(y)$.

$$p_0(y)\propto\mathbf1_{\mathcal Y}(y)\exp\!\left(-\frac{J(y)}{\lambda}\right).$$

<div class="definition-grid"><div>$p_0$: target probability density over decisions.<br>$\mathcal Y$: feasible decision domain.<br>$\mathbf1_{\mathcal Y}(y)$: 1 if $y\in\mathcal Y$; 0 otherwise.</div><div>$\lambda>0$: temperature, in cost units.<br>Smaller $\lambda$ favors lower costs more strongly.<br>$\propto$: proportional; density is normalized to integrate to 1.</div></div>

<div class="two-points"><div><b>Example</b><br>$J(a)$ is path length. $\mathcal Y$ contains heights in $[-2,2]$ whose paths satisfy the clearance requirement.</div><div><b>Why introduce noise?</b><br>The target can have separated high-density regions. Gaussian smoothing produces a sequence of less concentrated densities.</div></div>

<div class="key-line">MBD estimates the direction toward higher density using model evaluations, without training a score network.</div>
<div class="source">Pan et al., MBD, Eq. (2). A valid target requires a finite, positive normalizing integral.</div>

## MBD noise schedule: what alpha controls {.notation-slide data-section="2 · Preliminaries"}

Forward noise defines the densities that MBD will traverse in reverse:

$$y_i=\sqrt{\bar\alpha_i}\,y+\sqrt{1-\bar\alpha_i}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),\qquad
\bar\alpha_i=\prod_{j=1}^{i}\alpha_j.$$

<div class="definition-grid"><div>$y$: clean decision; $y_i$: noisy decision at level $i$.<br>$i=1,\ldots,N$: diffusion level; $N$: total levels.<br>$p_i$: probability density of $y_i$.</div><div>$0<\alpha_i<1$: signal-variance retention per forward step.<br>$\bar\alpha_i$: cumulative signal-variance retention.<br>$\bar\alpha_0=1$: the clean, zero-noise endpoint.</div></div>

<div class="definition-grid"><div>$\epsilon$: standard Gaussian noise vector.<br>$I$: identity matrix of the decision dimension.<br>$j$: index in the product defining $\bar\alpha_i$.</div><div>$\sqrt{\bar\alpha_i}$ scales the clean decision.<br>$\sqrt{1-\bar\alpha_i}$ scales the added noise.<br>Example: $\bar\alpha_i=0.25$ gives scales $0.5$ and $\sqrt{0.75}$.</div></div>

<div class="key-line">Forward: $0\rightarrow N$, more noise. MBD optimization: $N\rightarrow0$, less noise. Diffusion level is not physical time.</div>
<div class="source">Pan et al., Eqs. (3)–(4). The controller coefficient is denoted $\kappa$ here, separately from diffusion $\alpha_i$.</div>
''')

section('One MBD iteration: sample', r'''
## One MBD iteration: sample, evaluate and weight {.notation-slide data-section="2 · Preliminaries"}

At level $i$, draw $K$ decision vectors around the current noisy vector $y_i$:

$$y^{[k]}\sim\mathcal N\!\left(\frac{y_i}{\sqrt{\bar\alpha_i}},\frac{1-\bar\alpha_i}{\bar\alpha_i}I\right),
\qquad k=1,\ldots,K.$$

<div class="definition-grid"><div>$K$: number of sampled decisions per level.<br>$[k]$: sample index; $y^{[k]}$: one sampled decision.<br>These samples are in clean decision coordinates.</div><div>Proposal mean: $y_i/\sqrt{\bar\alpha_i}$.<br>Proposal covariance: $[(1-\bar\alpha_i)/\bar\alpha_i]I$.<br>In the scalar example, $y^{[k]}=a^{[k]}$.</div></div>

Evaluate each sampled decision with the model. Then normalize its target-density weight:

$$w_k=\frac{p_0(y^{[k]})}{\sum_{j=1}^{K}p_0(y^{[j]})},
\qquad \bar y=\sum_{k=1}^{K}w_k y^{[k]}.$$

<div class="definition-grid"><div>$w_k$: nonnegative cost weight, with $\sum_k w_k=1$.<br>$j$: summation index over the same $K$ samples.</div><div>$\bar y$: cost-weighted decision mean.<br>Infeasible decisions have zero weight. The mean itself need not be feasible.</div></div>

<div class="source">Pan et al., Eqs. (8)–(9). $K$ counts decisions; POAM's covariance matrix is $K_{ZZ}$. If all weights vanish, recovery is required before normalization.</div>
''')

section('Use the weighted mean', r'''
## Use the weighted mean to refine the noisy vector {.notation-slide data-section="2 · Preliminaries"}

The **score** $s_i(y_i)=\nabla_{y_i}\log p_i(y_i)$ points toward increasing log density. Here $\nabla_{y_i}$ means a gradient with respect to the noisy decision vector.

$$\widehat s_i=\frac{-y_i+\sqrt{\bar\alpha_i}\,\bar y}{1-\bar\alpha_i},\qquad
y_{i-1}=\frac{y_i+(1-\bar\alpha_i)\widehat s_i}{\sqrt{\alpha_i}}.$$

<div class="definition-grid"><div>$\widehat s_i$: score estimate from the $K$ model evaluations.<br>The hat denotes an estimate, not the exact score.<br>$y_{i-1}$: updated noisy decision at the next level.</div><div>First add the score correction to $y_i$.<br>Then divide by $\sqrt{\alpha_i}$ to change levels.<br>No derivative of the rollout cost is required.</div></div>

**Substituting the score reveals exactly what this update does:**

$$y_i+(1-\bar\alpha_i)\widehat s_i=\sqrt{\bar\alpha_i}\,\bar y
\quad\Longrightarrow\quad
\boxed{y_{i-1}=\sqrt{\bar\alpha_{i-1}}\,\bar y}.$$

<div class="key-line">The next sampling mean is $y_{i-1}/\sqrt{\bar\alpha_{i-1}}=\bar y$. Its covariance decreases to $[(1-\bar\alpha_{i-1})/\bar\alpha_{i-1}]I$.</div>
<div class="source">Pan et al., MCSA Eq. (6) and score Eq. (9). At $i=1$, $y_0=\bar y$ and reverse iterations end. This deterministic update does not guarantee a feasible mean.</div>

::: {.notes}
The equivalence follows by direct substitution and alpha_bar_i = alpha_i alpha_bar_(i-1). It applies to this MCSA update and its displayed score, not to every diffusion sampler. The paper's printed Algorithm 2 has an alpha-bar index discrepancy in line 3; the Gaussian proposal here consistently uses Eq. (8). The model-free forward-noise equation describes a density construction, not physical robot motion.
:::
''')

# Add on-step notation reminders to the POAM numerical visual, not only notes.
s = s.replace(r'S_Z=K"', r'S_Z=K_{ZZ}"')
s = s.replace('Numerical teaching example in one spatial dimension. Kernel and inducing positions are fixed; the conditional residual is retained. Wind values use normalized units.',
              r'$\mathcal B_n^d$: belief after $n$ observations; $\sigma_d(r)$: wind standard deviation. The numerical example fixes the kernel and inducing positions; wind uses normalized units.')

start=s.index('## One MBD iteration, step by step')
end=s.index('\n## ',start+3)
part=s[start:end]
part=part.replace('<option value="6">','<option value="6" selected>').replace('<option value="7" selected>','<option value="7">')
part=part.replace('<span>Each iteration uses the same eight steps.</span>', '<button type="button" data-action="continue">Continue to next iteration</button>')
part=part.replace('<b>Denoise to the next level</b>', '<b>Denoise: move → rescale → resample</b>')
eqs=[r'y_i\text{: noisy optimizer value},\quad \bar\alpha_i\text{: cumulative retention},\quad\alpha_i\text{: one-step retention}',
 r'a^{[k]}\sim\mathcal N(y_i/\sqrt{\bar\alpha_i},(1-\bar\alpha_i)/\bar\alpha_i)',
 r'r(s;a^{[k]}),\quad s\in[0,1]\text{ is path progress}',
 r'J_k=J(a^{[k]})=\text{path length};\quad\mathbf1_{\mathcal Y}=0\text{ for infeasible heights}',
 r'w_k\propto\mathbf1_{\mathcal Y}(a^{[k]})e^{-J_k/\lambda},\quad\lambda=0.12,\quad\sum_k w_k=1',
 r'\bar a=\sum_{k=1}^{K}w_k a^{[k]},\quad K=64\text{ sampled path heights}',
 r'\widehat s_i=(-y_i+\sqrt{\bar\alpha_i}\bar a)/(1-\bar\alpha_i)\quad\text{(score estimate)}',
 r'y_i+(1-\bar\alpha_i)\widehat s_i=\sqrt{\bar\alpha_i}\,\bar a',
 r'y_{i-1}=\frac{y_i+(1-\bar\alpha_i)\widehat s_i}{\sqrt{\alpha_i}}=\sqrt{\bar\alpha_{i-1}}\,\bar a',
 r'\text{Next mean}=\bar a,\qquad\text{next standard deviation}=\sqrt{(1-\bar\alpha_{i-1})/\bar\alpha_{i-1}}',
 r'i>1:\ \text{draw new heights and evaluate again};\qquad i=1:\ \text{end reverse iterations}']
rendered=''.join(f'<div class="mbd-equation" data-step="{i}"'+(' hidden' if i else '')+'><span class="latex-display" data-latex="'+html.escape(eq,quote=True)+'"></span></div>' for i,eq in enumerate(eqs))
part=re.sub(r'<div class="mbd-equation".*?(?=<div class="mbd-controls")',lambda _:rendered,part,flags=re.S)
part=part.replace('max="7"','max="10"').replace('aria-label="Algorithm step"','aria-label="Algorithm animation frame"')
part=part.replace('The highlighted line is the current algorithm step. Denoising updates a decision vector; it is not robot motion.', 'Step 8 has four frames: score correction, rescaling, next sampling distribution, and new paths. Nothing here is physical robot motion.')
part=part[:part.index('::: {.notes}')]+'''::: {.notes}
The player opens at iteration 7, level 2, to show the substantial narrowing before the last level. All eight saved iterations are selectable. The final algorithm line spans frames 8–11: score correction, change of noise scale, the next proposal, and the actual next saved sample bank. The next iteration button starts its full cost evaluation; it is enabled only after the transition and never after the final level. Gray path clouds in the last frame are prior proposals; blue paths are fresh proposals, not morphed versions of the same samples. All data come from the original saved teaching example. The noisy coordinate and clean path-height coordinate are labeled separately. The final level explicitly has no next Gaussian or sample bank. PDF export uses the before/after proposal view at level 2.
:::
'''
s=s[:start]+part+s[end:]
s=s.replace('The following slide expands one iteration into eight algorithm steps.', 'The following slide expands one iteration into eight algorithm steps, including four denoising frames.')
(OUT/'slides.qmd').write_text(s)

# Re-embed editable player sources into the original page wrapper.
c=(BASE/'controls.html').read_text()
for name in ['interactive.js','walkthrough.js']:
    old=(BASE/name).read_text()
    assert old in c, name
    c=c.replace(old,(ROOT/'docs/presentation'/name).read_text())
(OUT/'controls.html').write_text(c)
css=(BASE/'theme.css').read_text().replace((BASE/'interactive.css').read_text(),(ROOT/'docs/presentation/interactive.css').read_text())
css+='\n.reveal .notation-slide{font-size:29px;}\n.reveal .notation-slide .definition-grid{font-size:25px;line-height:1.35;}\n.reveal .notation-slide .key-line{font-size:27px;}\n'
(OUT/'theme.css').write_text(css)
print('Revised',s.count('\n## '),'slides; preserved archived numerical data.')
