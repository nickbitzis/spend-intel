"""Capture README screenshots with headless Chrome. Start the server first: make run.

Usage: uv run python scripts/screenshot.py --port 8849
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
]
DOCS = Path(__file__).resolve().parents[1] / "docs"


def chrome() -> str:
    for c in CHROME_PATHS:
        if Path(c).exists() or shutil.which(c):
            return c
    raise SystemExit("Chrome not found")


def shoot(url: str, out: Path, width: int, height: int, dark: bool = False) -> None:
    args = [
        chrome(),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--window-size={width},{height}",
        "--virtual-time-budget=8000",
        f"--screenshot={out}",
    ]
    if dark:
        args += ["--force-dark-mode", "--blink-settings=preferredColorScheme=0"]
    else:
        args.append("--blink-settings=preferredColorScheme=1")
    subprocess.run([*args, url], check=True, capture_output=True, timeout=60)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8849)
    ap.add_argument("--out", type=Path, default=DOCS)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    a.out.mkdir(exist_ok=True)
    shoot(f"{base}/#/summary", a.out / "screenshot.png", 1400, 1180)
    shoot(f"{base}/#/opportunities", a.out / "opportunities.png", 1400, 1000, dark=True)
    shoot(f"{base}/#/suppliers", a.out / "suppliers.png", 1400, 1000)
    shoot(f"{base}/#/categories/l2/MR", a.out / "categories.png", 1400, 900)


if __name__ == "__main__":
    main()
