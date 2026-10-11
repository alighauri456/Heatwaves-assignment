"""Build national, decade-grid and extreme-year summaries from yearly CSVs.

Dependency: numpy. Run: python scripts/build_csiro_summaries.py
Expected years default to 1979 through 2025 inclusive; change with --start-year
and --end-year. Missing, duplicate or unexpected years cause validation failure.
Yearly sources are read only and never overwritten.

National statistics give every land grid cell equal weight (not area/population
weighting). mean_positive_ehf is the mean of nonblank cell positive-day means,
not a pooled daily EHF mean. Decade statistics give each available year equal
weight at each cell; blank positive-EHF means are excluded, never made zero.
Decade max_ehf is the maximum annual maximum. 1979 is excluded from decades.
The 2020s currently cover 2020-2025, not a full decade. No annualisation is
applied: partial years remain partial. Yearly CSVs do not retain date coverage,
so completeness within a year cannot be established from these inputs alone.
1979 is known to be incomplete; check source coverage for other endpoint years.
Longest runs touching file boundaries may be truncated. These are index
statistics, not official severe/extreme heatwave categories.

Extreme-year ranks are independent descending competition ranks (1, 1, 3).
Outputs preserve float64 precision. Only one year's columns plus one decade's
per-cell accumulators are held in memory. Grid equality is checked by numeric
coordinates, allowing input row order to vary. Outputs are staged as .partial
files so validation failure does not replace completed summaries.
"""

import argparse
from collections import Counter
import csv
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT_COLUMNS = ["year", "latitude", "longitude", "positive_ehf_days",
                 "max_ehf", "mean_positive_ehf", "longest_positive_ehf_run_days"]
NATIONAL_COLUMNS = ["year", "mean_heatwave_days", "median_heatwave_days",
                    "mean_longest_run", "max_longest_run", "mean_positive_ehf", "max_ehf"]
DECADE_COLUMNS = ["latitude", "longitude", "decade", "mean_heatwave_days",
                  "mean_longest_run", "mean_positive_ehf", "max_ehf"]
EXTREME_COLUMNS = ["year", "rank_mean_heatwave_days", "mean_heatwave_days",
                   "rank_max_ehf", "max_ehf"]


def find_years(directory, start, end):
    files = {}
    for path in directory.glob("csiro_grid_*.csv"):
        match = re.fullmatch(r"csiro_grid_(\d{4})\.csv", path.name)
        if not match:
            raise ValueError(f"Unexpected yearly filename: {path.name}")
        year = int(match.group(1))
        if year in files:
            raise ValueError(f"Duplicate year: {year}")
        files[year] = path
    expected = set(range(start, end + 1))
    if set(files) != expected:
        raise ValueError(f"Year validation failed: missing={sorted(expected - files.keys())}, "
                         f"unexpected={sorted(files.keys() - expected)}")
    return sorted(files.items())


def read_year(path, year):
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    if len(header) != len(set(header)) or not set(INPUT_COLUMNS) <= set(header):
        raise ValueError(f"Invalid columns in {path}")
    # Load only needed numeric columns; blanks are represented as NaN.
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64,
                         usecols=[header.index(c) for c in INPUT_COLUMNS], ndmin=2)
    if data.size == 0 or data.shape[1] != 7:
        raise ValueError(f"Empty or malformed yearly CSV: {path}")
    if not np.isfinite(data[:, [0, 1, 2, 3, 6]]).all() or not np.all(data[:, 0] == year):
        raise ValueError(f"Invalid year, coordinates or counts: {path}")
    if np.isinf(data).any():
        raise ValueError(f"Infinite values in {path}")
    days, runs = data[:, 3], data[:, 6]
    if np.any(days < 0) or np.any(runs < 0) or np.any(runs > days) or np.any(days != np.floor(days)) or np.any(runs != np.floor(runs)):
        raise ValueError(f"Invalid day counts or runs: {path}")
    if not np.array_equal(np.isnan(data[:, 5]), days == 0):
        raise ValueError(f"Positive-EHF mean missingness disagrees with day count: {path}")
    data = data[np.lexsort((data[:, 2], data[:, 1]))]
    if np.any(np.all(data[1:, 1:3] == data[:-1, 1:3], axis=1)):
        raise ValueError(f"Duplicate grid coordinates: {path}")
    return data


def finite_mean(values):
    good = np.isfinite(values)
    return float(values[good].mean()) if good.any() else np.nan


def finite_max(values):
    good = np.isfinite(values)
    return float(values[good].max()) if good.any() else np.nan


def output_row(writer, row):
    writer.writerow(["" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v for v in row])


def descending_ranks(values):
    counts = Counter(v for v in values if np.isfinite(v))
    ranks, position = {}, 1
    for value in sorted(counts, reverse=True):
        ranks[value] = position
        position += counts[value]
    return [ranks.get(v, "") for v in values]


def build(input_dir, output_dir, start=1979, end=2025):
    files = find_years(input_dir, start, end)
    print(f"Validated all {len(files)} expected years: {start}-{end}", flush=True)
    print("WARNING: statistics use available dates without annualisation; 1979 is incomplete.")
    print("Yearly CSVs do not establish within-year completeness; verify endpoint-year source coverage.")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / name for name in
               ("csiro_national_annual.csv", "csiro_decade_grid.csv", "csiro_extreme_years.csv")]
    partials = [p.with_suffix(".csv.partial") for p in outputs]
    national, grid, decade, years_in_decade = [], None, None, []
    sums = valid_counts = maxima = None
    decade_rows = 0
    with partials[1].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(DECADE_COLUMNS)

        def flush_decade():
            if decade is None:
                return 0
            print(f"{decade}s: {len(years_in_decade)} years ({years_in_decade[0]}-{years_in_decade[-1]})"
                  + ("; INCOMPLETE DECADE" if len(years_in_decade) != 10 else ""), flush=True)
            means = np.full_like(sums, np.nan)
            np.divide(sums, valid_counts, out=means, where=valid_counts > 0)
            for i, (lat, lon) in enumerate(grid):
                output_row(writer, [lat, lon, f"{decade}s", *means[i], maxima[i]])
            return len(grid)

        for year, path in files:
            data = read_year(path, year)
            if grid is None:
                grid = data[:, 1:3].copy()
            elif not np.array_equal(grid, data[:, 1:3]):
                raise ValueError(f"Grid cells differ from reference year: {path}")
            days, runs, positive, maximum = data[:, 3], data[:, 6], data[:, 5], data[:, 4]
            national.append([year, float(days.mean()), float(np.median(days)), float(runs.mean()),
                             int(runs.max()), finite_mean(positive), finite_max(maximum)])
            if 1980 <= year <= 2029:
                current = year // 10 * 10
                if current != decade:
                    decade_rows += flush_decade()
                    decade, years_in_decade = current, []
                    sums = np.zeros((len(grid), 3), dtype=np.float64)
                    valid_counts = np.zeros((len(grid), 3), dtype=np.int32)
                    maxima = np.full(len(grid), np.nan)
                years_in_decade.append(year)
                for index, values in enumerate((days, runs, positive)):
                    good = np.isfinite(values)
                    np.add(sums[:, index], values, out=sums[:, index], where=good)
                    valid_counts[:, index] += good
                np.fmax(maxima, maximum, out=maxima)
            print(f"Read {year}: {len(data):,} grid cells", flush=True)
            del data, days, runs, positive, maximum
        decade_rows += flush_decade()
    with partials[0].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(NATIONAL_COLUMNS)
        for row in national:
            output_row(writer, row)
    day_ranks = descending_ranks([row[1] for row in national])
    ehf_ranks = descending_ranks([row[6] for row in national])
    with partials[2].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXTREME_COLUMNS)
        for row, day_rank, ehf_rank in sorted(zip(national, day_ranks, ehf_ranks), key=lambda item: (item[1], item[0][0])):
            output_row(writer, [row[0], day_rank, row[1], ehf_rank, row[6]])
    for temporary, output, count in zip(partials, outputs, [len(national), decade_rows, len(national)]):
        temporary.replace(output)
        print(f"{output}: {count:,} rows; {output.stat().st_size:,} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/csiro_yearly")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--start-year", type=int, default=1979)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()
    if args.end_year < args.start_year:
        parser.error("--end-year must be at least --start-year")
    build(args.input_dir, args.output_dir, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
