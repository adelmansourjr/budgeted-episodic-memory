"""Compare a fresh run with every frozen action sequence, ignoring timings."""
import argparse
from itertools import zip_longest
import json
from pathlib import Path

from analyse import open_rows

VERIFIED = Path(__file__).resolve().parent/"results/final/rows.jsonl.gz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fresh", type=Path)
    args = parser.parse_args()
    with open_rows(VERIFIED) as reference, open_rows(args.fresh) as candidate:
        records = 0
        for records, (a,b) in enumerate(zip_longest(reference,candidate),1):
            assert a is not None and b is not None, "Different record count"
            x,y = json.loads(a), json.loads(b)
            for key in ("cpu_ns","setup_cpu_ns"):
                x.pop(key)
                y.pop(key)
            assert x == y, f"Reproduction mismatch in row {records}: {x['id']}, {x['method']}"
    assert records == 131208
    print(f"Verified {records:,} exact freshly reproduced selections and counters.")


if __name__ == "__main__":
    main()
