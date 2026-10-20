"""Offline histogram density of all positive land-grid daily EHF, 1980–2025.

Dependencies and land mask match build_csiro_monthly_activity.py. No sampling;
0.5 °C² bins, shared Gaussian bandwidth 1 °C², reflected at zero. The browser
loads only the compact processed CSV. Existing files are never overwritten.
"""
import csv
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date
from shapely import intersects_xy
from process_csiro_single_year import australian_land_boundary, coordinates, numeric_values

ROOT = Path(__file__).resolve().parents[1]
PERIODS = ['1980s', '1990s', '2000s', '2010s', '2020–2025']
STEP = 0.5
BANDWIDTH = 1.0


def main():
    land = australian_land_boundary()
    with (ROOT / 'data/csiro_national_annual.csv').open() as handle:
        reference = {int(r['year']): r for r in csv.DictReader(handle)}
    upper = np.ceil(max(float(reference[y]['max_ehf']) for y in range(1980, 2026)))
    edges = np.arange(0, upper + STEP, STEP)
    counts = {p: np.zeros(len(edges)-1, dtype=np.int64) for p in PERIODS}
    coverage = []
    for year in range(1980, 2026):
        period = f'{year//10*10}s' if year < 2020 else PERIODS[-1]
        with Dataset(ROOT / f'raw/CSIRO/AU_daily_EHF_epsg4326_0p11deg_{year}.nc') as dataset:
            ehf = dataset.variables['EHF']
            coords, dimensions = coordinates(dataset, ehf)
            lat, lon, times = [numeric_values(coords[k]) for k in ('lat', 'lon', 'time')]
            if not all(a.size and np.isfinite(a).all() for a in (lat, lon, times)):
                raise ValueError(f'Invalid coordinates in {year}')
            dates = num2date(times, coords['time'].units, calendar=getattr(coords['time'], 'calendar', 'standard'))
            keys = [(d.year, d.month, d.day) for d in dates]
            if len(set(keys)) != len(keys) or {k[0] for k in keys} != {year}:
                raise ValueError(f'Invalid daily dates in {year}')
            axes = [ehf.dimensions.index(d) for d in dimensions]
            positive_count = finite_count = land_cells = 0
            annual_hist = np.zeros(len(edges)-1, dtype=np.int64)
            for start in range(0, len(lat), 8):
                stop = min(start+8, len(lat))
                mask = intersects_xy(land, lon[None, :], lat[start:stop, None])
                land_cells += int(mask.sum())
                if not mask.any():
                    continue
                selection = [slice(None)] * ehf.ndim
                selection[axes[0]] = slice(start, stop)
                values = np.ma.asarray(np.transpose(ehf[tuple(selection)], axes), dtype=np.float64).filled(np.nan)[mask, :]
                finite = np.isfinite(values)
                positive = values[finite & (values > 0)]
                if positive.size and positive.max() >= upper:
                    raise ValueError('Intensity exceeds shared histogram range')
                annual_hist += np.histogram(positive, bins=edges)[0]
                positive_count += len(positive)
                finite_count += int(finite.sum())
            assert annual_hist.sum() == positive_count
            # Reconcile daily input with the existing equal-cell annual counts.
            expected = float(reference[year]['mean_heatwave_days']) * land_cells
            assert abs(positive_count - expected) < 0.01, (year, positive_count, expected)
            counts[period] += annual_hist
            coverage.append(dict(year=year, available_dates=len(dates), first_date='%04d-%02d-%02d' % min(keys),
                                 last_date='%04d-%02d-%02d' % max(keys), land_cells=land_cells,
                                 finite_grid_cell_days=finite_count, positive_grid_cell_days=positive_count))
            print(f'{year}: {positive_count:,} positive cell-days; reconciled', flush=True)
    # An explicitly descriptive benchmark, not an official intensity category.
    index = int(np.searchsorted(np.cumsum(counts['1980s']), counts['1980s'].sum()*.9))
    threshold = float(edges[index+1])
    summaries = {}
    offsets = np.arange(-8, 9)
    kernel = np.exp(-.5*(offsets*STEP/BANDWIDTH)**2)
    kernel /= kernel.sum()
    rows = []
    for period in PERIODS:
        hist = counts[period]
        n = int(hist.sum())
        probability = hist/n
        smooth = np.convolve(np.pad(probability, (8, 8), mode='symmetric'), kernel, mode='valid')
        density = smooth/(smooth.sum()*STEP)
        assert abs(density.sum()*STEP-1) < 1e-12
        summary = dict(positive_grid_cell_days=n,
                       approximate_median_ehf=float(edges[np.searchsorted(np.cumsum(hist), n*.5)]+STEP/2),
                       share_at_or_above_historical_benchmark=float(hist[index+1:].sum()/n))
        summaries[period] = summary
        for i, value in enumerate(density):
            rows.append([period, f'{edges[i]+STEP/2:.2f}', f'{value:.9f}', int(hist[i])])
    output = ROOT / 'data/csiro_intensity_decade_density.csv'
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['period', 'ehf', 'density', 'bin_count'])
        writer.writerows(rows)
    metadata = dict(source='Local CSIRO daily EHF, aggregated offline; no NetCDF loaded by webpage.',
                    unit='°C²', years='1980–2025; 1979 excluded', mask='Existing Natural Earth 1:10m Australian land grid-centre mask.',
                    weighting='Pool all available finite positive grid-cell-day observations within each period; equal observation weight, not area/population weighting or equal-year weighting.',
                    bin_width=STEP, bin_range=[0, float(upper)], smoothing='Shared Gaussian bandwidth 1 °C², truncated at four bandwidths; symmetric reflection at the histogram boundaries (0 and 185 °C²); discrete density renormalised to unit area.',
                    benchmark_ehf=threshold, benchmark='Upper edge of the histogram bin containing the 1980s 90th percentile; descriptive threshold, not an official severe/extreme category.',
                    validation='Every annual positive cell-day count reconciles with existing national annual heatwave-day mean; all bin counts conserved; each density integrates to one.',
                    caveats=['Only EHF > 0 is included; ridge height is relative density, not frequency across all days.',
                             '2020–2025 has six available years; 2025 ends on 29 August. No imputation or annualisation.',
                             'The pooled distribution weights years with more positive cell-days more heavily.',
                             'Bin aggregation and smoothing approximate shape; full observed tail retained, no clipping.',
                             'EHF is relative to local conditions and is expressed in °C², not air temperature.'],
                    summaries=summaries, coverage=coverage)
    (ROOT / 'data/csiro_intensity_decade_density_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(dict(benchmark_ehf=threshold, summaries=summaries), indent=2))
    print(f'{len(rows)} rows; {output.stat().st_size} bytes')


if __name__ == '__main__':
    main()
