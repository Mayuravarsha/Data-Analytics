"""Write data/vehicles_sample.csv.gz: 30,000 random raw listings (fixed seed).

Used by the tests and CI so they run without the 1.4 GB LFS file.
    python scripts/make_sample.py path/to/vehicles.csv
"""

import sys

from usedcars.data import RAW_COLUMNS, SAMPLE, read_raw

raw = read_raw(sys.argv[1] if len(sys.argv) > 1 else None)
raw.sample(30_000, random_state=0)[RAW_COLUMNS].to_csv(SAMPLE, index=False, compression="gzip")
print(f"wrote {SAMPLE} ({SAMPLE.stat().st_size / 1e6:.1f} MB)")
