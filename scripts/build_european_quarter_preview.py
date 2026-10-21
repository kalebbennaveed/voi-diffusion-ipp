"""Copy the standalone environment review to the existing local demo server."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
source = root / "docs/european_quarter_preview"
destination = root / "output/all_methods_demo/european-quarter"
shutil.copytree(source, destination, dirs_exist_ok=True)
print(destination)
