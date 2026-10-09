#!/usr/bin/env python3

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tienkung_demo.multi_evt2_runner import main


if __name__ == "__main__":
    main()
