"""Compare equal-cell annual state heatwave days, then equal-year period means.

Requires numpy, shapely>=2 and cached full ABS 2021 boundaries. Existing inputs
are read only. Run with the same dependencies as build_csiro_duration_map.py.
"""
import csv
import json
from pathlib import Path

import numpy as np
from shapely import intersects_xy, make_valid, prepare
from shapely.geometry import shape

from build_csiro_summaries import read_year

ROOT = Path(__file__).resolve().parents[1]
PERIODS = {'1980–1989': range(1980, 1990), '2016–2025': range(2016, 2026)}


def main():
    document = json.loads((ROOT / 'raw/boundaries/abs_states_2021_full.geojson').read_text(encoding='utf-8'))
    features = sorted((f for f in document['features'] if f['geometry'] is not None),
                      key=lambda f: f['properties']['state_code_2021'])
    geometries = [make_valid(shape(f['geometry'])) for f in features]
    for geometry in geometries:
        prepare(geometry)
    reference = assignment = None
    annual = {}
    ties = 0
    for years in PERIODS.values():
        for year in years:
            data = read_year(ROOT / f'data/csiro_yearly/csiro_grid_{year}.csv', year)
            coords = data[:, 1:3]  # latitude, longitude; sorted and validated
            if reference is None:
                reference = coords.copy()
                assignment = np.full(len(data), -1, dtype=np.int16)
                for i, geometry in enumerate(geometries):
                    inside = intersects_xy(geometry, coords[:, 1], coords[:, 0])
                    ties += int(np.count_nonzero(inside & (assignment >= 0)))
                    assignment[inside & (assignment < 0)] = i
            elif not np.array_equal(coords, reference):
                raise ValueError(f'Grid changed in {year}')
            annual[year] = {f['properties']['state_name_2021']: float(data[assignment == i, 3].mean())
                            for i, f in enumerate(features) if np.any(assignment == i)}
            print(f'{year}: complete', flush=True)
    names = list(annual[1980])
    assert len(names) == 8
    means = {name: {period: float(np.mean([annual[y][name] for y in years]))
                    for period, years in PERIODS.items()} for name in names}
    changes = {name: means[name]['2016–2025'] - means[name]['1980–1989'] for name in names}
    order = sorted(names, key=lambda name: (-abs(changes[name]), name))
    # Validate recent overlap against the existing state's published frequency.
    with (ROOT / 'data/csiro_state_duration_2020_2025.csv').open() as handle:
        for row in csv.DictReader(handle):
            if row['state'] in names:
                actual = np.mean([annual[y][row['state']] for y in range(2020, 2026)])
                assert abs(actual - float(row['mean_heatwave_days'])) < 0.000001
    rows = [dict(state=name, period=period, mean_annual_heatwave_days=round(means[name][period], 6))
            for name in order for period in PERIODS]
    output = ROOT / 'data/csiro_state_frequency_change.csv'
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['state', 'period', 'mean_annual_heatwave_days'])
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(periods={p: list(years) for p, years in PERIODS.items()},
                    measure='Days with positive EHF per year, not separate heatwave events.',
                    aggregation='Equal-cell annual state means, then equal-year mean over ten years per period; no area or population weighting.',
                    boundary_source='https://geo.abs.gov.au/arcgis/rest/services/ASGS2021/STE/MapServer/0',
                    boundary_license='ABS CC BY 4.0', grid_cells=len(reference),
                    unmatched_cells=int(np.count_nonzero(assignment < 0)), border_ties=ties,
                    cells_by_state={f['properties']['state_name_2021']: int(np.count_nonzero(assignment == i)) for i, f in enumerate(features)},
                    validation='2020–2025 state means reproduce published mean_heatwave_days within 0.000001 days; all inputs checked for duplicate coordinates, finite counts, matching grids and years.',
                    caveats=['1979 excluded.', '2025 ends on 29 August; partial-year counts are not annualised.',
                             'Unmatched coastal cells excluded; border ties use lowest ABS state code.',
                             'Other Territories has no represented cells and is omitted, not assigned zero.'],
                    sort='Descending absolute magnitude of recent minus historical mean.',
                    summaries={name: dict(means[name], change_days=changes[name]) for name in order})
    (ROOT / 'data/csiro_state_frequency_change_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata['summaries'], indent=2))
    print(f'{len(rows)} rows; {output.stat().st_size} bytes')


if __name__ == '__main__':
    main()
