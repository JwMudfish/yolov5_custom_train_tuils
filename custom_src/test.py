from pathlib import Path
import sys

FILE = Path(__file__).resolve()
print(FILE)
ROOT = FILE.parents[0]  # YOLOv5 root directory
print(ROOT)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = ROOT.relative_to(Path.cwd())  # relative

print(FILE)
