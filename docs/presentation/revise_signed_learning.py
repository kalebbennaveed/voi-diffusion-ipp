"""Clarify chronological signed learning value in the existing Quarto deck."""
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'output/presentation/wind_aware_mbd'
BACK=ROOT/'tmp/slides/signed-learning/previous'
BACK.mkdir(parents=True,exist_ok=True)
for name in ['slides.qmd','theme.css','controls.html','README.txt']:
    if not (BACK/name).exists():shutil.copyfile(D/name,BACK/name)
s=(BACK/'slides.qmd').read_text()
s=s.replace('Original and reversed A/B environments; one additional environment.','Original and reversed A/B environments.')
s=re.sub(r'(?ms)^## Additional environment: an open passage.*?(?=^## )','',s)
s=s.replace('<li>The additional open-passage environment shows that this extension can cost more.</li>','<li>These two archived missions do not establish a general learning advantage.</li>')
s=s.replace('The anticipated-learning cost averages these complete monitoring costs over physical fields and observation noise. A detour to learn is preferred only if its later benefit outweighs its monitoring cost.','The anticipated-learning cost includes the initial motion, sensing and later continuation. The following slides express its benefit through matched predictions.')
new=r'''
## Two matched predictions define the value of learning {#paired-predictions data-section="4 · Method — anticipated learning"}

Fix one policy parameter vector $\theta$. Each draw $s$ contains a physical field and an observation-noise stream. Both predictions use that field and the same initial state and clarity.

$$J_s^{\mathrm{fixed}}=\int_0^T q_d(q_s^{\mathrm{fixed}}(\tau))\,d\tau,
\qquad J_s^{\mathrm{learn}}=\int_0^T q_d(q_s^{\mathrm{learn}}(\tau))\,d\tau.$$

<div class="definition-grid"><div><b>Fixed belief:</b> retain $\mathcal B_t^d$ throughout the prediction.</div><div><b>Anticipated learning:</b> assimilate observations when available and continue from the reached state and clarity.</div></div>

$$\Delta_s=J_s^{\mathrm{fixed}}-J_s^{\mathrm{learn}}.$$

<div class="two-points"><div>$\Delta_s>0$: learning reduces monitoring cost.</div><div>$\Delta_s<0$: learning increases monitoring cost.</div></div>
<div class="key-line">Both costs cover the same full horizon, including the motion used to obtain information.</div>

::: {.notes}
S counts the sampled field and observation-noise pairs. s indexes one pair. Each field stays fixed within both predictions. Match the noise stream and schedule, while generating each measurement at its own prediction's reached position. Controls respond to each branch's own state and available belief. The paths may diverge after the first learning update. Neither branch restarts after an observation. This comparison holds theta fixed and differs from comparing two independently optimized methods. The following slides suppress theta arguments for readability.
:::

## How often and how much does learning help? {#probability-and-magnitude data-section="4 · Method — anticipated learning"}

Among the $S$ matched predictions, let $n_+$ count those with $\Delta_s>0$.

$$\widehat p=\frac{n_+}{S},\qquad
\widehat g=\frac{1}{n_+}\sum_{s:\,\Delta_s>0}\Delta_s.$$

<div class="definition-grid"><div>$\widehat p$: estimated probability that learning helps.</div><div>$\widehat g$: mean cost reduction among the predictions where learning helps. Set it to zero if $n_+=0$.</div></div>

$$\widehat V^+=\widehat p\,\widehat g
=\frac1S\sum_{s=1}^{S}\max(\Delta_s,0).$$

<div class="key-line">Probability times conditional improvement gives the expected positive benefit.</div>
<div class="small-note">These quantities describe one policy. They do not yet account for outcomes where its learning-dependent execution performs worse.</div>

::: {.notes}
All quantities refer to the fixed policy theta from the preceding slide. The formula for g applies for n_+>0. The zero convention avoids division by zero. V+ has the same units as monitoring cost. The scalar probability p here differs from the MBD density p_0. Separately optimized mission costs cannot estimate this same-policy probability or conditional gain.
:::

## Signed learning value includes deterioration {#signed-learning-value data-section="4 · Method — anticipated learning"}

The expected deterioration is

$$\widehat D=\frac1S\sum_{s=1}^{S}\max(-\Delta_s,0).$$

Subtract it from the positive benefit:

$$\boxed{\widehat V^{\mathrm{net}}=\widehat p\,\widehat g-\widehat D
=\frac1S\sum_{s=1}^{S}\Delta_s}.$$

| Illustrative cost differences $\Delta_s$ | $+2$ | $+1$ | $-1$ | $0$ |
|---|---:|---:|---:|---:|
| Interpretation | Helps | Helps | Harms | No change |

<div class="example-line">Here $\widehat p=0.5$, $\widehat g=1.5$, $\widehat D=0.25$, so $\widehat V^{\mathrm{net}}=0.50$.</div>
<div class="key-line">A policy receives credit for beneficial outcomes and a penalty for deterioration.</div>

::: {.notes}
These four numbers are an arithmetic illustration, not A/B results. Vnet measures a signed difference in monitoring cost and differs from the constraint-policy value V_a in the execution controller. Ignoring deterioration gives 0.75 rather than 0.50. Zero differences are neither helpful nor harmful. Every matched prediction remains in the estimate.
:::

## Complete task cost with anticipated learning {#learning-cost-in-mbd data-section="4 · Method — anticipated learning"}

Average the two matched full-horizon costs:

$$\widehat J^{\mathrm{fixed}}=\frac1S\sum_s J_s^{\mathrm{fixed}},
\qquad \widehat J^{\mathrm{learn}}=\frac1S\sum_s J_s^{\mathrm{learn}}.$$

$$\boxed{\widehat J^{\mathrm{fixed}}-\widehat V^{\mathrm{net}}
=\widehat J^{\mathrm{learn}}}.$$

**MBD minimizes $\widehat J^{\mathrm{learn}}$.** This includes the information-gathering motion and its later payoff.

| Illustrative policy | $\widehat J^{\mathrm{fixed}}$ | $\widehat V^{\mathrm{net}}$ | $\widehat J^{\mathrm{learn}}$ |
|---|---:|---:|---:|
| Direct route | 10 | 1 | **9** |
| Detour to learn | 14 | 4 | 10 |

<div class="key-line">Choose the direct route here. The detour has a larger learning value but a higher complete task cost.</div>
<div class="small-note">Do not subtract learning value again from $\widehat J^{\mathrm{learn}}$. That would count the benefit twice.</div>

::: {.notes}
The table is illustrative. For matching sample sets, Jfixed - Vnet equals Jlearn exactly. This decomposes the chronological learning objective without introducing a second reward or tuning coefficient. MBD can evaluate Jlearn directly and compute Jfixed when attribution is needed. A detour wins only when its complete predicted monitoring cost is lower. The horizon must cover enough later execution to capture that payoff.
:::

'''
s=s.replace('## MBD now optimizes the complete monitoring prediction',new+'## MBD now optimizes the complete monitoring prediction')
s=s.replace('<b>Wind-aware:</b> $\\widehat J=\\widehat J_{\\mathrm{WA}}$ uses the fixed-belief execution prediction.','<b>Wind-aware:</b> $\\widehat J=\\widehat J^{\\mathrm{fixed}}$ uses the fixed-belief execution prediction.')
s=s.replace('<b>Anticipated learning:</b> $\\widehat J=\\widehat J_{\\mathrm{AL}}$ updates the belief during the prediction.','<b>Anticipated learning:</b> $\\widehat J=\\widehat J^{\\mathrm{learn}}$ includes chronological belief updates.')
s=s.replace('Average cumulative clarity deficits across the fixed physical-field draws.','Average complete learning-updated costs: $\\widehat J^{\\mathrm{learn}}=\\widehat J^{\\mathrm{fixed}}-\\widehat V^{\\mathrm{net}}$.')
s=s.replace('<b>Refine:</b> average monitoring costs, compute parameter weights','<b>Refine:</b> average complete learning-updated monitoring costs, compute parameter weights')
s=s.replace('Archived development runs; these predate the revised bounded MBD target.','Archived development runs; these predate the revised bounded MBD target. The signed-value equations decompose the chronological objective; no new mission is implied.')
s=s.replace('The selected archived evidence remains limited.','The selected archived evidence remains limited. Between-method differences are not the same-policy signed learning value defined in the method.')
s=s.replace('data-latex="J_s(\\theta)=\\int_0^T q_d(q^{[s]}(\\tau))\\,d\\tau"','data-latex="J_s^{\\mathrm{learn}}=\\int_0^T q_d(q_s^{\\mathrm{learn}}(\\tau))\\,d\\tau,\\qquad \\Delta_s=J_s^{\\mathrm{fixed}}-J_s^{\\mathrm{learn}}"')
s=s.replace(r'\widehat J_{\mathrm{WA}}',r'\widehat J^{\mathrm{fixed}}')
(D/'slides.qmd').write_text(s)
css=(BACK/'theme.css').read_text()+'''\n#brand-logo{width:clamp(64px,12vw,190px);}
#deck-footer{font-size:clamp(9px,.95vw,15px);padding-left:18px;}
.snapshot .reveal .result-gif,.snapshot .reveal .mbd-overview,.snapshot .reveal .controls,.snapshot .reveal .walk-controls,.snapshot .reveal .mbd-controls,.snapshot .reveal .mbd-iteration-choice{display:none!important;}
.snapshot .reveal .print-poster{display:block!important;width:1500px;max-width:100%;max-height:625px;object-fit:contain;margin:0;}
.reveal .example-line{font-size:28px;line-height:1.4;margin:24px 0;}
.reveal #learning-cost-in-mbd table{margin:16px 0;font-size:25px;}
.reveal #learning-cost-in-mbd table th{font-size:25px;padding:12px 16px;}
.reveal #learning-cost-in-mbd table td{padding:12px 16px;}
.reveal #learning-cost-in-mbd .key-line{font-size:27px;margin-top:18px;}
.reveal #signed-learning-value table{margin:20px 0;}
'''
(D/'theme.css').write_text(css)
p=ROOT/'docs/presentation/walkthrough.js'
w=p.read_text().replace("'Whole-trajectory cost'","'Full cost with learning'").replace("'Area = cumulative deficit.'","'Area = full monitoring cost.'")
w=w.replace('Repeat motion propagation and scheduled belief updates to the horizon. Integrate clarity deficit over the entire trajectory, including the interval before learning.','Integrate the full learning-updated monitoring cost, including motion before learning. Its matched fixed-belief cost gives the signed learning-value diagnostic. MBD uses the learning-updated cost directly.')
w=w.replace('Average the complete monitoring costs for each policy. This happens before exponentiation in the MBD weight calculation.','Average the full learning-updated monitoring costs. This equals fixed-belief cost minus signed learning value. Do not subtract another learning reward.')
w=w.replace("location.search.includes('print-pdf')?Number(root.dataset.printStep):0","(location.search.includes('print-pdf')||location.search.includes('snapshot'))?Number(root.dataset.printStep):0")
p.write_text(w)
c=(BACK/'controls.html').read_text();start=c.rfind('<script>');end=c.find('</script>',start)
assert 'window.walkPlayers' in c[start:end]
c=c[:start]+'<script>\n'+w+'\n'+c[end:]
c=c.replace("detail&&location.search.includes('print-pdf')?9:0","detail&&(location.search.includes('print-pdf')||location.search.includes('snapshot'))?9:0")
c=c.replace('<script>\n(function(){', '<script>\nif(location.search.includes(\"snapshot\"))document.documentElement.classList.add(\"snapshot\");\n(function(){',1)
(D/'controls.html').write_text(c)
print('Revised',len(re.findall(r'^## ',s,re.M)),'slides')
