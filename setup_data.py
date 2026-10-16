"""Download the Tiny Shakespeare corpus to data/input.txt.

Usage:  python setup_data.py
"""

import urllib.request
from pathlib import Path

# Karpathy's char-rnn copy of Tiny Shakespeare (~1.1 MB, ~40k lines of plays).
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT = Path(__file__).parent / "data" / "input.txt"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"{OUT} already exists ({OUT.stat().st_size:,} bytes), skipping download.")
        return
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=60) as resp:
        data = resp.read()
    # Decode and write back as UTF-8 so the file is valid UTF-8 regardless of the source encoding.
    OUT.write_text(data.decode("utf-8"), encoding="utf-8")
    print(f"Saved {len(data):,} bytes to {OUT}")


if __name__ == "__main__":
    main()
