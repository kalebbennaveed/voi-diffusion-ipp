"""Stage only the self-contained slide directory in an existing Pages checkout."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib, json, re, shutil, subprocess

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'output/presentation/wind_aware_mbd'
CHECKOUT=Path('/tmp/ipp-slides-publish')
DEST=CHECKOUT/'dis-aware-ipp/update-slides'
DEST.mkdir(parents=True,exist_ok=True)
document=(SRC/'slides.html').read_text()
assert '<meta name="robots"' not in document
document=document.replace('<head>','<head>\n<meta name="robots" content="noindex, nofollow, noarchive">\n<meta name="googlebot" content="noindex, nofollow, noarchive">',1)
revision=json.loads((SRC/'revision_manifest.json').read_text())['revision']
document=document.replace('href="theme.css"',f'href="theme.css?revision={revision}"')
# Reveal loads only the active mission animation. This avoids fetching 80 MB on entry.
document=re.sub(r'(<img\b[^>]*class="result-gif"[^>]*?)\bsrc="([^"]+)"',r'\1data-src="\2" data-original-src="\2"',document)
document=document.replace("img.dataset.originalSrc||img.getAttribute('src')", "img.dataset.originalSrc||img.getAttribute('src')||img.dataset.src")
# No Jekyll front matter: this is a static deck, not a site page or sitemap entry.
(DEST/'index.html').write_text(document)
shutil.copy2(SRC/'theme.css',DEST/'theme.css')
shutil.copytree(SRC/'slides_files',DEST/'slides_files',dirs_exist_ok=True)
shutil.copytree(SRC/'assets/katex/dist',DEST/'assets/katex/dist',dirs_exist_ok=True)

class Assets(HTMLParser):
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('src','href','data-src','data-original-src') and value and value.startswith('assets/'):
                source=SRC/value
                if source.is_file():
                    target=DEST/value;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
Assets().feed(document)
# No source repository, run directories, source archives or large standalone duplicate.
assert not (DEST/'slides-standalone.html').exists()
assert all(p.stat().st_size<100*1024**2 for p in DEST.rglob('*') if p.is_file())
changes=subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=CHECKOUT,text=True).splitlines()
assert all(line[3:].startswith('dis-aware-ipp/update-slides/') for line in changes),changes
manifest={
 'url':'https://kalebbennaveed.github.io/dis-aware-ipp/update-slides/',
 'repository':'kalebbennaveed/kalebbennaveed.github.io',
 'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip(),
 'scope':'Only dis-aware-ipp/update-slides/. No existing site files, configuration, navigation or sitemap changed.',
 'robots':'noindex, nofollow, noarchive',
 'public_not_access_controlled':True,
 'files':{str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.rglob('*')) if p.is_file()}}
(ROOT/'tmp/slides/publication_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'files':len(manifest['files']),'bytes':sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()),'scope':manifest['scope']}))
