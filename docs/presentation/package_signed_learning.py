"""Prepare the unlisted slide package locally, without touching the website checkout."""
from html.parser import HTMLParser
from pathlib import Path
import hashlib
import json
import re
import shutil

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'output/presentation/wind_aware_mbd'
DEST=ROOT/'output/presentation/unlisted_signed_learning'
DEST.mkdir(parents=True,exist_ok=True)
s=(SRC/'slides.html').read_text()
s=s.replace('<head>','<head>\n<meta name="robots" content="noindex, nofollow, noarchive">\n<meta name="googlebot" content="noindex, nofollow, noarchive">',1)
s=re.sub(r'(<img\b[^>]*class="result-gif"[^>]*?)\bsrc="([^"]+)"',r'\1data-src="\2" data-original-src="\2"',s)
s=s.replace("img.dataset.originalSrc||img.getAttribute('src')", "img.dataset.originalSrc||img.getAttribute('src')||img.dataset.src")
(DEST/'index.html').write_text(s)
shutil.copyfile(SRC/'theme.css',DEST/'theme.css')
shutil.copytree(SRC/'slides_files',DEST/'slides_files',dirs_exist_ok=True)
shutil.copytree(SRC/'assets/katex/dist',DEST/'assets/katex/dist',dirs_exist_ok=True)
class Assets(HTMLParser):
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ('src','href','data-src','data-original-src') and v and v.startswith('assets/'):
    p=SRC/v
    if p.is_file():
     out=DEST/v;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
Assets().feed(s)
assert len(re.findall(r'class="result-gif"',s))==2
assert 'assets/open.gif' not in s
manifest={'destination':'dis-aware-ipp/update-slides/','robots':'noindex, nofollow, noarchive','scope':'Only the existing unlisted slide directory. No homepage, navigation, sitemap or website configuration changes.','slides':68,'simulation_environments':['original A/B','reversed A/B'],'files':{str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.rglob('*')) if p.is_file()}}
(ROOT/'tmp/slides/signed-learning/publication_plan.json').write_text(json.dumps(manifest,indent=2))
print('Prepared',len(manifest['files']),'files locally. Website unchanged.')
