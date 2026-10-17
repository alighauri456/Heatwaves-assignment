"""Build compact monthly heatwave counts offline; never overwrite annual data.

Requires numpy, netCDF4, shapely and pyshp (as for the existing daily processor).
Run: python scripts/build_csiro_monthly_activity.py
Each cell has equal weight. A valid cell has finite EHF on every available
date in the month. Missing dates are not imputed or annualised. Months with
no dates/valid cells have blank values, not zero. Coverage is saved separately.
"""

import calendar
import csv
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date
from shapely import intersects_xy

from process_csiro_single_year import australian_land_boundary, coordinates, numeric_values

ROOT = Path(__file__).resolve().parents[1]


def monthly_counts(values):
    """Return positive-day counts and full-observation validity for each cell."""
    values = np.ma.asarray(values, dtype=np.float64).filled(np.nan)
    finite = np.isfinite(values)
    return np.sum(finite & (values > 0), axis=2), np.all(finite, axis=2)


def build():
    land = australian_land_boundary()
    rows, coverage = [], []
    for year in range(1980, 2026):
        path = ROOT / f"raw/csiro/AU_daily_EHF_epsg4326_0p11deg_{year}.nc"
        with Dataset(path) as dataset:
            ehf = dataset.variables["EHF"]
            coords, dimensions = coordinates(dataset, ehf)
            lat, lon, times = [numeric_values(coords[k]) for k in ("lat", "lon", "time")]
            if not all(np.isfinite(a).all() and a.size for a in (lat, lon, times)):
                raise ValueError(f"Invalid coordinates: {year}")
            dates = num2date(times, coords["time"].units,
                             calendar=getattr(coords["time"], "calendar", "standard"))
            day_keys = [(d.year, d.month, d.day) for d in dates]
            if len(set(day_keys)) != len(dates) or {d.year for d in dates} != {year}:
                raise ValueError(f"Duplicate daily dates or wrong year: {year}")
            axes = [ehf.dimensions.index(d) for d in dimensions]
            month_indices = [np.array([i for i, d in enumerate(dates) if d.month == m])
                             for m in range(1, 13)]
            totals, cells = np.zeros(12, dtype=np.int64), np.zeros(12, dtype=np.int64)
            land_cells = 0
            for start in range(0, len(lat), 8):
                stop = min(start + 8, len(lat))
                mask = intersects_xy(land, lon[None, :], lat[start:stop, None])
                land_cells += int(mask.sum())
                if not mask.any():
                    continue
                selection = [slice(None)] * ehf.ndim
                selection[axes[0]] = slice(start, stop)
                values = np.transpose(ehf[tuple(selection)], axes)
                for m, indices in enumerate(month_indices):
                    if not indices.size:
                        continue
                    counts, valid = monthly_counts(values[:, :, indices])
                    valid &= mask
                    totals[m] += int(counts[valid].sum())
                    cells[m] += int(valid.sum())
            for m, indices in enumerate(month_indices):
                month = m + 1
                expected = calendar.monthrange(year, month)[1]
                value = float(totals[m] / cells[m]) if cells[m] else None
                if value is not None and not 0 <= value <= len(indices):
                    raise ValueError(f"Invalid monthly count: {year}-{month}")
                rows.append([year, month, calendar.month_abbr[month],
                             f"{value:.4f}" if value is not None else ""])
                coverage.append(dict(year=year, month=month, available_days=len(indices),
                                     calendar_days=expected, valid_land_cells=int(cells[m]),
                                     land_cells=land_cells, complete_month=len(indices) == expected))
            print(f"{year}: {len(dates)} dates; {land_cells:,} land cells; "
                  f"monthly sum {sum(float(r[3]) for r in rows[-12:] if r[3]):.4f}", flush=True)
    # Same mask and complete cells should reproduce the existing annual counts.
    with (ROOT / "data/csiro_national_annual.csv").open() as handle:
        annual = {int(r['year']): float(r['mean_heatwave_days']) for r in csv.DictReader(handle)}
    reconciled = []
    for year in range(1980, 2026):
        parts = [r for r in coverage if r['year'] == year and r['available_days']]
        if parts and all(r['valid_land_cells'] == r['land_cells'] for r in parts):
            total = sum(float(r[3]) for r in rows if r[0] == year and r[3])
            if abs(total - annual[year]) > 0.00061:
                raise ValueError(f"Monthly sum differs from existing annual mean: {year}")
            reconciled.append(year)
    output = ROOT / "data/csiro_monthly_heatwave_activity.csv"
    with output.with_suffix('.csv.partial').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['year', 'month', 'month_name', 'mean_heatwave_days'])
        writer.writerows(rows)
    output.with_suffix('.csv.partial').replace(output)
    metadata = dict(
        source='CSIRO daily EHF; offline aggregation of local yearly NetCDF files',
        years='1980–2025; incomplete 1979 excluded',
        definition='Count available EHF > 0 days per cell/month, then equal-cell mean',
        valid_cell='Australian land grid centre with finite EHF on every available date in that month',
        mask='Existing Natural Earth 1:10m Australia mask; no area or population weighting',
        missing='No dates or valid cells: blank mean, not zero; no imputation or annualisation',
        precision='CSV rounded to four decimal places',
        annual_reconciliation_years=reconciled, coverage=coverage)
    (ROOT / 'data/csiro_monthly_heatwave_activity_metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f"Saved {len(rows)} rows; {output.stat().st_size:,} bytes; "
          f"{len(reconciled)} years reconcile with existing annual data", flush=True)


if __name__ == '__main__':
    build()
