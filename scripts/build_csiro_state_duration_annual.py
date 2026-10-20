"""Build annual equal-cell state duration means, using the choropleth methodology.

Requires numpy and shapely>=2; uses cached full ABS boundaries. No existing
datasets are rewritten. Runs use available dates without annualisation.
"""
import csv
import json
from pathlib import Path

import numpy as np
from shapely import intersects_xy, make_valid, prepare
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]


def main():
    boundary = ROOT / 'raw/boundaries/abs_states_2021_full.geojson'
    document = json.loads(boundary.read_text(encoding='utf-8'))
    features = sorted((f for f in document['features'] if f['geometry'] is not None),
                      key=lambda f: f['properties']['state_code_2021'])
    geometries = [make_valid(shape(f['geometry'])) for f in features]
    for geometry in geometries:
        prepare(geometry)
    reference = assignment = None
    rows = []
    ties = 0
    for year in range(1980, 2026):
        data = np.genfromtxt(ROOT / f'data/csiro_yearly/csiro_grid_{year}.csv',
                             delimiter=',', names=True, dtype=float,
                             usecols=('year', 'latitude', 'longitude',
                                      'longest_positive_ehf_run_days'))
        data.sort(order=['latitude', 'longitude'])
        coords = np.column_stack((data['longitude'], data['latitude']))
        if reference is None:
            reference = coords.copy()
            assignment = np.full(len(data), -1, dtype=np.int16)
            for i, geometry in enumerate(geometries):
                inside = intersects_xy(geometry, coords[:, 0], coords[:, 1])
                ties += int(np.count_nonzero(inside & (assignment >= 0)))
                assignment[inside & (assignment < 0)] = i
        elif not np.array_equal(coords, reference):
            raise ValueError(f'Grid changed in {year}')
        if not np.all(data['year'] == year):
            raise ValueError(f'CSV year mismatch in {year}')
        for i, feature in enumerate(features):
            runs = data['longest_positive_ehf_run_days'][assignment == i]
            if not len(runs):
                continue
            if not np.isfinite(runs).all() or (runs < 0).any():
                raise ValueError(f'Invalid duration in {year}')
            rows.append(dict(year=year, state=feature['properties']['state_name_2021'],
                             annual_longest_run=round(float(runs.mean()), 6)))
        print(f'{year}: complete', flush=True)
    assert len(rows) == 46 * 8
    # Independently check the overlap against the published choropleth values.
    existing = list(csv.DictReader((ROOT / 'data/csiro_state_duration_2020_2025.csv').open()))
    for state in existing:
        values = [r['annual_longest_run'] for r in rows
                  if r['state'] == state['state'] and r['year'] >= 2020]
        if values:
            assert abs(float(np.mean(values)) - float(state['mean_longest_run'])) < 0.000002
    output = ROOT / 'data/csiro_state_duration_annual.csv'
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['year', 'state', 'annual_longest_run'])
        writer.writeheader()
        writer.writerows(rows)
    summaries = {}
    for name in sorted({r['state'] for r in rows}):
        values = [r['annual_longest_run'] for r in rows if r['state'] == name]
        q1, median, q3 = np.quantile(values, [.25, .5, .75])
        summaries[name] = dict(years=len(values), median=float(median),
                               q1=float(q1), q3=float(q3), iqr=float(q3-q1))
    metadata = dict(years=list(range(1980, 2026)), excluded_years=[1979],
                    aggregation='Equal-cell mean of annual longest positive-EHF runs within each state; each annual state mean is one boxplot observation.',
                    boundary_source='https://geo.abs.gov.au/arcgis/rest/services/ASGS2021/STE/MapServer/0',
                    boundary_license='ABS CC BY 4.0', grid_cells=len(reference),
                    unmatched_cells=int(np.count_nonzero(assignment < 0)), border_ties=ties,
                    cells_by_state={f['properties']['state_name_2021']: int(np.count_nonzero(assignment == i)) for i, f in enumerate(features)},
                    caveats=['1979 excluded because incomplete.', '2025 ends on 29 August; no annualisation.',
                             'Annual file boundaries may split runs.', 'Unmatched coastal cells excluded; border ties use lowest ABS state code.',
                             'Other Territories has no represented cells and is omitted, not assigned zero.',
                             'Distributions describe variation between annual state means, not between individual events or grid cells.'],
                    validation='2020–2025 annual means reproduce existing choropleth means within 0.000002 days.',
                    summaries=summaries)
    (ROOT / 'data/csiro_state_duration_annual_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(summaries, indent=2))
    print(f'{len(rows)} rows; {output.stat().st_size} bytes')


if __name__ == '__main__':
    main()
