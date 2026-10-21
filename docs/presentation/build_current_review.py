"""Assemble the algorithm-first review in the existing Quarto deck.

The complete slides.qmd is versioned and remains directly editable. This helper
replaces only the current main sections and preserves the preliminaries and
the prior anticipated-learning archive. It reads measured timing JSON; no
timing values are inferred from work counts.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import shutil

ROOT=Path(__file__).resolve().parents[2]
DECK=ROOT/'output/presentation/wind_aware_mbd'
AUTHOR=ROOT/'docs/presentation'
PROFILE=ROOT/'docs/compute_review/current45_dynamic.json'
REVISION='20261021-method-compute-review'


def main():
    text=(DECK/'slides.qmd').read_text()
    prefix=text[:text.index('## Problem formulation {.section-break #optimistic-problem')]
    archive=text[text.index('## Archived method {.section-break #archived-method'):]
    prefix=re.sub(r'::: \{\.notes\}\n.*?\n:::', '''::: {.notes}
Current implementation review, 21 October 2026. This deck separates the generic method from numerical choices and code. The current score is favorable-tail monitoring plus a mean soft-constraint penalty. Actual observations update the disturbance belief; predicted beliefs remain fixed. The final archive contains the earlier anticipated-learning method and historical results.
:::''',prefix,count=1,flags=re.S)
    prefix=prefix.replace('<strong>Planned evaluation</strong><br>Baselines and ablations; prior method and results in the final archive.','<strong>Implementation and compute time</strong><br>Exact settings, measured costs, expensive components, and source files.')
    prefix=prefix.replace('Execution prediction, optimistic costs, adapted sampling and real learning.','Algorithm first, then execution prediction, scoring, sampling, and real learning.')
    method=(AUTHOR/'current_method.qmd').read_text()
    implementation=(AUTHOR/'current_implementation.qmd').read_text()
    d=json.loads(PROFILE.read_text()); t=d['timings']
    assert d['split_matches_fused']
    def val(name,scale=1,key='median_seconds',digits=2):return f'{t[name][key]*scale:.{digits}f}'
    replacements={
        'MAX_SIGMA':f"{math.sqrt((math.cos(.008/1.008*math.pi/2)/math.cos((d['config']['cosine_end']+.008)/1.008*math.pi/2))**2-1):.3f}",
        'COMPLETE_MED':val('complete_planning_call'),'COMPLETE_MAX':val('complete_planning_call',key='max_seconds'),
        'OPT_MED':val('optimizer_fused'),'OPT_MAX':val('optimizer_fused',key='max_seconds'),
        'RRT_MED':val('rrt'),'RRT_MAX':val('rrt',key='max_seconds'),
        'FIELD_MED_MS':val('one_posterior_field_bank',1000),'FIELD_MAX_MS':val('one_posterior_field_bank',1000,key='max_seconds'),
        'TRAIN_MED':val('training_separate'),'SELECT_MED':val('selection_separate'),
        'INITIAL_MED':val('initial_evaluations_separate'),'BATCH_MED':val('one_diffusion_batch_separate'),
        'FILTERED_MS':val('one_trajectory_four_fields_filtered',1000),'UNFILTERED_MS':val('one_trajectory_four_fields_unfiltered',1000),
        'FILTER_RATIO':f"{t['one_trajectory_four_fields_filtered']['median_seconds']/t['one_trajectory_four_fields_unfiltered']['median_seconds']:.1f}",
        'VALUE_US':val('constraint_values',1e6,digits=1),'DERIV_US':val('constraint_values_and_derivatives',1e6,digits=1),
        'FILTER_US':val('one_filter_call',1e6,digits=1),
        'UPDATE_MS':val('one_poam_update',1000),'EXEC_MS':val('one_second_nonlinear_execution',1000),
        'COLD_OPT':val('optimizer_fused',key='first_seconds'),
    }
    for name,value in replacements.items():implementation=implementation.replace('{{'+name+'}}',value)
    assert '{{' not in implementation
    updated=prefix+method+'\n'+implementation+'\n'+archive
    (DECK/'slides.qmd').write_text(updated)
    css=(DECK/'theme.css').read_text().split('/* Current-method revision within the established Quarto template. */')[0]
    (DECK/'theme.css').write_text(css+'\n'+(AUTHOR/'current.css').read_text())
    controls=(DECK/'controls.html').read_text()
    marker='/* Presentation-only arithmetic and walkthrough. Does not run the monitoring planner. */'
    index=controls.index(marker)
    controls=controls[:index]+(AUTHOR/'optimistic.js').read_text()+'\n</script>\n'
    (DECK/'controls.html').write_text(controls)
    manifest=dict(revision=REVISION,status='Current method, implementation, and measured compute-time review',
        slides=len(re.findall(r'^## ',updated,re.M)),archived_slides=44,
        timing_profile=str(PROFILE.relative_to(ROOT)),timing_profile_sha256=hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
        current_configuration=d['config'],profiling_record=d['record'],profiling_mission_time=d['mission_time'],
        runtime_tuning='Stopped at user request. No 10 s / 2 s result claimed.',
        archived_results_modified=False,model_source_hashes=d['source_hashes'])
    (DECK/'revision_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(revision=REVISION,slides=manifest['slides'],timing_replacements=replacements),indent=2))


if __name__=='__main__':main()
