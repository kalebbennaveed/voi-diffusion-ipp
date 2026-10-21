"""Revise only the review deck. Preserve all previous method slides in its archive."""
from pathlib import Path
import re, json, hashlib
ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'output/presentation/wind_aware_mbd'
BACK=ROOT/'tmp/slides/optimistic-review/previous'
s=(BACK/'slides.qmd').read_text()
prefix,archive=s.split('## Problem formulation',1)
archive='## Problem formulation'+archive
prefix=prefix.replace('and anticipating useful wind learning','with optimistic planning and real online learning')
a=prefix.index('::: {.notes}');b=prefix.index(':::',a+4)+3
prefix=prefix[:a]+'''::: {.notes}
Method proposal for review, 20 October 2026. The first contribution is disturbance-aware prediction of delivered monitoring. The revision introduces optimism over performance outcomes and importance-corrected proposal adaptation. It holds the posterior fixed inside predictions and learns from real measurements. The new method has not been implemented or evaluated. All previous value-of-learning equations and their associated results are retained in the final Archived method section.
:::'''+prefix[b:]
prefix=prefix.replace('<b>2. Anticipated learning</b><br>Predict where observations can improve those future execution predictions.','<b>2. Optimistic planning with online learning</b><br>Try favorable plausible alternatives and learn from actual execution.')
prefix=prefix.replace('The second is whether future observations will change subsequent execution. Uncertainty reduction is not the task objective.','The second is whether optimism encourages useful early experience that improves later monitoring. This is a hypothesis for the revised method. Uncertainty reduction is not the task objective.')
prefix=prefix.replace('Wind-aware planning; wind-aware planning with anticipated learning.','Disturbance-aware monitoring and optimistic performance preference.')
prefix=prefix.replace('Architecture, execution prediction, belief updates and optimization.','Execution prediction, optimistic costs, adapted sampling and real learning.')
prefix=prefix.replace('The retained score estimator and the changed rollout objective.','The target, importance correction and retained reverse update.')
prefix=prefix.replace('<strong>Results</strong><br>Original and reversed A/B environments.','<strong>Planned evaluation</strong><br>Baselines and ablations; prior method and results in the final archive.')
# Reserve eta for the optimistic fraction, and use the source revision's design-policy symbol.
prefix=prefix.replace(r'\eta',r'\vartheta').replace(r'\pi_b',r'\pi_D').replace(r'\phi_b',r'\phi_D')
# This QP is explanatory and is not quoted as the source paper's robust optimization order.
prefix=prefix.replace('RPCBF uses this construction for safety filtering. Our finite, sampled execution model is inspired by it; we do not claim its safety guarantees for the planner.','RPCBF motivates the policy-value construction. The common-input inequalities shown here specify our sampled adaptation; the original paper states a separate robust condition.')
# Retain original content and manual players, with explicit archive labeling on every slide.
archive=re.sub(r'(?m)^## (.*?)(\{[^\n]*\})$',lambda m:'## '+m[1]+m[2][:-1]+' .archived-method data-archived="true"}',archive)
archive=re.sub(r'data-section="[^"]*"','data-section="Archived method"',archive)
archive=archive.replace('## Problem formulation {','## Archived problem formulation {',1)
separator='''## Archived method {.section-break #archived-method data-section="Archived method"}

<div class="section-number">A</div>
<div class="section-subtitle">Previous anticipated-learning and value-of-learning formulation.</div>
<div class="section-agenda">Original equations, interactive explanations, A/B results and implementation notes retained for reference</div>

::: {.notes}
Everything after this divider belongs to the earlier method. Its hypothetical posterior updates and signed value-of-learning decomposition are not used by the revised optimistic method. Its recorded A/B results have not been recomputed and must not be attributed to the new proposal. The word proposed inside these archived figures and animations denotes the earlier anticipated-learning planner. Numerical implementation details here are historical only.
:::

'''
new=(ROOT/'docs/presentation/optimistic_method.qmd').read_text()
(D/'slides.qmd').write_text(prefix+new+'\n'+separator+archive)
css=(BACK/'theme.css').read_text()+'\n'+(ROOT/'docs/presentation/optimistic.css').read_text()
(D/'theme.css').write_text(css)
controls=(BACK/'controls.html').read_text()
controls=controls.replace("const slide=Reveal.getCurrentSlide();document.getElementById('current-section').textContent=slide.dataset.section||'Method';", "const slide=Reveal.getCurrentSlide();document.getElementById('current-section').textContent=slide.dataset.section||'Method';document.body.classList.toggle('show-archive-label',slide.dataset.archived==='true'||slide.id==='archived-method');")
controls+='\n<div id="archive-label">Archived method · prior formulation and results</div>\n<script>\n'+(ROOT/'docs/presentation/optimistic.js').read_text()+'\n</script>\n'
(D/'controls.html').write_text(controls)
content=(D/'slides.qmd').read_text()
manifest={'revision':'20261020-optimistic-review','status':'method proposal only; no planner or simulation implementation',
'source_pdf':'/Users/kalebbennaveed/Downloads/Paper Pitch Development.pdf',
'source_pdf_sha256':hashlib.sha256(Path('/Users/kalebbennaveed/Downloads/Paper Pitch Development.pdf').read_bytes()).hexdigest(),
'slides':len(re.findall(r'^## ',content,re.M)),
'new_main_slides':len(re.findall(r'^## ',prefix+new,re.M)),
'archived_slides':len(re.findall(r'^## ',archive,re.M)),
'new_results':False,'archived_results_modified':False,'public_deployment':'not part of this review stage',
'notation_changes':{'POAM kernel parameters':'vartheta, to reserve eta for optimism','design policy':'pi_D','proposal density':'q_i^prop, distinct from clarity q','controller value-rate function':'linear kappa V, kappa>0'},
'design_choices_for_review':['Closed design-policy evaluation interval includes terminal point.','The wind-blind baseline omits the added constraint filter; the second blind baseline uses the shared runtime filter.','Fading eta schedule is an optional ablation, not part of the supplied core method.']}
(D/'revision_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
