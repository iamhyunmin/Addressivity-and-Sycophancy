
#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COMPUTE_SCRIPT = SCRIPT_DIR.parent / "compute_mean_diff_direct.py"


def main() -> None:
    if not COMPUTE_SCRIPT.exists():
        raise SystemExit(f"Missing: {COMPUTE_SCRIPT}")


    cmd = [sys.executable, str(COMPUTE_SCRIPT)] + sys.argv[1:]
    print(f"Running: {' '.join(cmd[:3])} ...")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
