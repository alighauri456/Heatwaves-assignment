"""Sequential, resumable processing of CSIRO yearly EHF files.

Run: python scripts/process_csiro_all_years.py
Inspect filenames only: python scripts/process_csiro_all_years.py --inspect-only
Dependencies are the same as process_csiro_single_year.py.

Each year runs the validated single-year script in a separate process: its
dynamic dimension handling, finite-value calculations and Natural Earth
Australia mask are reused without modification. CSV column names remain
positive_ehf_days (heatwave_days), max_ehf, mean_positive_ehf and
longest_positive_ehf_run_days (longest_run), plus latitude, longitude and year.
The mask source is Natural Earth 1:10m Admin 0 Countries, ADMIN=Australia:
https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/

Available dates may cover only part of a year; coverage is reported explicitly.
Runs touching the first/last available date may be truncated by file boundaries.
Runs are never joined across files. No official severe/extreme categories are
derived. Child-process exit releases all per-year arrays and NetCDF resources.

Completed outputs are skipped. A CSV is promoted from .partial to its final
name only after a successful child exit and streaming validation. A stale
.partial file is overwritten on retry, never mistaken for a completed year.
Errors are logged to stderr; other years continue. Exit status is 1 on failure.
"""

import argparse
from collections import Counter
import csv
import logging
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"AU_daily_EHF_epsg4326_0p11deg_(\d{4})\.nc", re.IGNORECASE)
FIELDS = ["year", "latitude", "longitude", "positive_ehf_days", "max_ehf",
          "mean_positive_ehf", "longest_positive_ehf_run_days"]


def inventory(directory):
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".nc")
    identified, unmatched = [], []
    for path in files:
        match = PATTERN.fullmatch(path.name)
        # Include alternate filenames with a trailing year, but report the mismatch.
        year_match = match or re.search(r"(?<!\d)(\d{4})(?!\d)(?=[^0-9]*\.nc$)", path.name, re.I)
        if not match:
            unmatched.append(path.name)
        identified.append((int(year_match.group(1)) if year_match else None, path))
    identified.sort(key=lambda item: (item[0] is None, item[0] or 0, item[1].name))
    counts = Counter(year for year, _ in identified if year is not None)
    missing = sorted(set(range(min(counts), max(counts) + 1)) - counts.keys()) if counts else []
    duplicates = sorted(year for year, count in counts.items() if count > 1)
    print(f"NetCDF files: {len(files)}", flush=True)
    print(f"Filename year range: {min(counts) if counts else 'none'} to {max(counts) if counts else 'none'}")
    print(f"Missing filename years: {missing or 'none'}")
    print(f"Duplicated filename years: {duplicates or 'none'}")
    print(f"Nonstandard filenames: {unmatched or 'none'}", flush=True)
    return identified, set(duplicates)


def inspect_dates(path, filename_year):
    """Read only coordinates, closing the file before the worker opens it."""
    import numpy as np
    from netCDF4 import Dataset, num2date
    from process_csiro_single_year import coordinates, numeric_values

    with Dataset(path, "r") as dataset:
        coords, _ = coordinates(dataset, dataset.variables["EHF"])
        times = numeric_values(coords["time"])
        if not times.size or not np.isfinite(times).all():
            raise ValueError("Empty/non-finite time coordinates")
        dates = num2date(np.sort(times), coords["time"].units,
                        calendar=getattr(coords["time"], "calendar", "standard"))
        if {date.year for date in dates} != {filename_year}:
            raise ValueError("Time-coordinate year does not match filename year")
        gaps = [(dates[i] - dates[i - 1]).total_seconds() / 86400 for i in range(1, len(dates))]
        if any(gap <= 0 or abs(gap - round(gap)) > 1e-8 for gap in gaps):
            raise ValueError("Expected distinct daily timestamps at whole-day intervals")
        first, last = dates[0], dates[-1]
        complete = ((first.month, first.day) == (1, 1)
                    and last.month == 12 and last.day == last.daysinmonth
                    and all(abs(gap - 1) < 1e-8 for gap in gaps))
        print(f"Coverage: {first} to {last}; {len(dates)} days; "
              f"{'complete calendar year' if complete else 'WARNING: incomplete calendar year'}", flush=True)


def csv_rows(path, year):
    """Validate output structure in constant memory and return its row count."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected CSV columns in {path}")
        count = 0
        for row in reader:
            if row.get("year") != str(year) or None in row or any(v is None for v in row.values()):
                raise ValueError(f"Invalid row/year in {path}")
            count += 1
        return count


def run_batch(files, duplicates, output_dir, block_size):
    outcomes = {"successful": [], "skipped": [], "failed": []}
    for index, (year, source) in enumerate(files, 1):
        started = time.perf_counter()
        label = year if year is not None else source.name
        print(f"\n[{index}/{len(files)}] {source.name}", flush=True)
        try:
            if year is None:
                raise ValueError("Cannot identify a year from filename")
            if year in duplicates:
                raise ValueError("Duplicate input year: resolve the conflicting files before processing")
            output = output_dir / f"csiro_grid_{year}.csv"
            if output.exists():
                rows = csv_rows(output, year)
                print(f"SKIPPED {year}: {rows:,} rows; {output.stat().st_size:,} bytes; "
                      f"{time.perf_counter() - started:.2f}s (existing output)", flush=True)
                outcomes["skipped"].append(label)
                continue
            inspect_dates(source, year)
            output_dir.mkdir(parents=True, exist_ok=True)
            partial = output.with_suffix(".csv.partial")
            # Synchronous worker: never more than one EHF file is processed at once.
            subprocess.run([sys.executable, "-B", "-u", str(ROOT / "scripts/process_csiro_single_year.py"),
                            "--input", str(source.resolve()), "--output", str(partial.resolve()),
                            "--latitude-block-size", str(block_size)], check=True)
            rows = csv_rows(partial, year)
            partial.replace(output)
            print(f"SUCCESS {year}: {rows:,} rows; {output.stat().st_size:,} bytes; "
                  f"{time.perf_counter() - started:.2f}s; {output}", flush=True)
            outcomes["successful"].append(label)
        except Exception as error:
            logging.error("FAILED %s after %.2fs: %s: %s", label,
                          time.perf_counter() - started, type(error).__name__, error)
            outcomes["failed"].append(label)
    print("\nBatch summary", flush=True)
    for status, years in outcomes.items():
        unique = list(dict.fromkeys(years))
        print(f"{status.capitalize()}: {len(unique)} — {', '.join(map(str, unique)) or 'none'}", flush=True)
    return 1 if outcomes["failed"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "raw/csiro")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/csiro_yearly")
    parser.add_argument("--latitude-block-size", type=int, default=8)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.latitude_block_size < 1:
        parser.error("--latitude-block-size must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    files, duplicates = inventory(args.input_dir)
    if args.inspect_only:
        return 0
    if not files:
        logging.error("No NetCDF files found")
        return 1
    return run_batch(files, duplicates, args.output_dir, args.latitude_block_size)


if __name__ == "__main__":
    sys.exit(main())
