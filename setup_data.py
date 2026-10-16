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
    data.decode("utf-8")  # fail loudly now if the download isn't valid UTF-8
    # Write the bytes untouched: write_text would turn every "
" into "
" on Windows.
    OUT.write_bytes(data)
    print(f"Saved {len(data):,} bytes to {OUT}")


if __name__ == "__main__":
    main()
