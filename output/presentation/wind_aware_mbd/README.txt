DISTURBANCE-AWARE INFORMATIVE PLANNING
Quarto Reveal.js presentation, 118 slides, revision 20261021-method-compute-review

CURRENT REVIEW
The generic method is presented algorithm first, followed by each component.
Implementation and computation are separate sections. The earlier anticipated-
learning method and its historical results remain in the final archive.

SEQUENCE
1-4       Introduction
5-24      Preliminaries, including manual POAM and MBD animations
25-28     Problem formulation
29-47     Current method: algorithm first, components and manual explanation
48-50     Comparison with MBD
51-64     Detailed implementation
65-73     Compute measurements, work counts, bottlenecks and source map
74        Archived method divider
75-118    Prior anticipated-learning method and historical results

COMPUTE
The complete warm planning call is 10.75 s median at the measured state.
Search, fresh-field comparison, RRT, field sampling, controller values and
state derivatives have separate measured timings and clearly stated scope.
No 10-second-horizon / two-second-runtime result is claimed.

FILES
slides.html: animated deck with local assets alongside it.
slides-standalone.html: generated self-contained copy, kept locally.
slides.pdf: static review copy; use HTML for manual animations.
slides.qmd: complete editable Quarto source.
docs/compute_review/ at repository root: exact configuration, raw timings, saved
profiling state and reproduction commands (see repository-relative path below).

REBUILD
From repository root:
  .venv/bin/python docs/presentation/build_current_review.py
From this deck directory:
  quarto render slides.qmd
The complete source can also be edited and rendered directly.
Detailed reproduction: docs/compute_review/README.md at repository root.

PUBLIC COPY
https://kalebbennaveed.github.io/dis-aware-ipp/update-slides/
Only the existing unlisted slide directory is published. No homepage,
navigation, sitemap or website configuration changes are required.

VALIDATION
validation.json records equation, image, layout and manual-control checks.
revision_manifest.json identifies the current configuration and measured data.
