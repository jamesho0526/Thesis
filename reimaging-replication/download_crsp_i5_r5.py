"""Download the CRSP daily fields needed for the I5/R5 replication.

The paper's CRSP data are licensed and require a WRDS account. This script
does not bypass that requirement; it downloads only after the user supplies
their own WRDS credentials.
"""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path


SQL = """
SELECT
    d.permno,
    d.date,
    d.openprc,
    d.askhi AS highprc,
    d.bidlo AS lowprc,
    d.prc AS closeprc,
    d.ret,
    d.vol,
    d.shrout,
    d.cfacpr,
    d.cfacshr,
    n.shrcd,
    n.exchcd
FROM crsp.dsf AS d
JOIN crsp.dsenames AS n
  ON d.permno = n.permno
 AND d.date BETWEEN n.namedt AND COALESCE(n.nameendt, DATE '9999-12-31')
WHERE d.date BETWEEN %(raw_start)s AND %(sample_end)s
  AND n.shrcd IN (10, 11)
  AND n.exchcd IN (1, 2, 3)
ORDER BY d.date, d.permno
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CRSP daily data for the paper's I5/R5 sample."
    )
    parser.add_argument("--sample-start", default="1993-01-01")
    parser.add_argument("--sample-end", default="2019-12-31")
    parser.add_argument(
        "--raw-start",
        default="1992-12-01",
        help="Warm-up date for the first 5-day image; keep before sample-start.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/crsp_i5_r5/crsp_daily_raw.csv.gz"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.raw_start > args.sample_start:
        raise ValueError("--raw-start must be on or before --sample-start")

    query = SQL.strip()
    if args.dry_run:
        print(query)
        print(
            f"\nParameters: raw_start={args.raw_start}, "
            f"sample_start={args.sample_start}, sample_end={args.sample_end}"
        )
        return

    import pandas as pd  # noqa: F401 - required by WRDS/pandas result handling

    try:
        import wrds
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'wrds'. Install with: py -m pip install wrds pandas"
        ) from exc

    username = os.environ.get("WRDS_USERNAME") or input("WRDS username: ")
    password = os.environ.get("WRDS_PASSWORD")
    if password is None:
        password = getpass.getpass("WRDS password: ")

    conn = wrds.Connection(wrds_username=username, wrds_password=password)
    try:
        frame = conn.raw_sql(
            query,
            params={
                "raw_start": args.raw_start,
                "sample_end": args.sample_end,
            },
            date_cols=["date"],
        )
    finally:
        conn.close()

    if frame.empty:
        raise RuntimeError("WRDS returned no rows; check credentials and date filters.")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, compression="gzip")
    print(f"Wrote {len(frame):,} rows to {output}")
    print(
        f"Rows include warm-up data from {args.raw_start} through {args.sample_end}; "
        f"filter the image/label date to {args.sample_start} through {args.sample_end}."
    )


if __name__ == "__main__":
    main()
