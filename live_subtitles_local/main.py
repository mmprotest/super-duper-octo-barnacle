from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    app_path = Path(__file__).resolve().parent / "app" / "ui_streamlit.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
