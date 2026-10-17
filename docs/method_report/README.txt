Methods report source and provenance

Final report: output/pdf/voi_diffusion_ipp_method.pdf

Rebuild from repository root:
  .venv/bin/python docs/method_report/make_assets.py
  <Python with ReportLab, Pillow and pypdf> docs/method_report/build_report.py

The first command uses the repository NumPy/Matplotlib environment. The second
was run with the bundled Codex document Python runtime. No JAX calculation or
mission is needed. Assets are regenerated in tmp/pdfs/method_report/.

provenance.json records source HEAD, source/data hashes and figure inputs.
validation.json records PDF quality checks. Equations describe implemented
methods; results are retained development data, not new experiments.
