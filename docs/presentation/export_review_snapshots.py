"""Export the visually checked slide snapshots as the static review PDF."""
from pathlib import Path
import json
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
DECK = ROOT / 'output/presentation/wind_aware_mbd'
QA = ROOT / 'tmp/slides/optimistic-review/qa'
report = json.loads((QA / 'report.json').read_text())
assert not report['errors'] and not report['math']['errors']
assert all(not row['images'] and not row['overflow'] for row in report['report'])
target = DECK / 'slides.pdf'
c = Canvas(str(target), pagesize=(1200, 675), pageCompression=1)
c.setTitle('Disturbance-aware informative planning: current method and compute review')
c.setSubject('Algorithm, component equations, implementation and measured compute time; previous value-of-learning method archived')
for i, slide in enumerate(report['slides'], 1):
    png = QA / f'{i:03d}.png'
    assert png.is_file(), png
    c.drawImage(ImageReader(str(png)), 0, 0, width=1200, height=675)
    c.bookmarkPage(slide['id'])
    c.addOutlineEntry(f"{i}. {slide['title'] or 'Title'}", slide['id'], 0)
    c.showPage()
c.save()
pdf = PdfReader(target)
assert len(pdf.pages) == len(report['slides'])
assert all(float(p.mediabox.width) == 1200 and float(p.mediabox.height) == 675 for p in pdf.pages)
print(json.dumps({'pdf':str(target), 'pages':len(pdf.pages), 'bytes':target.stat().st_size}))
