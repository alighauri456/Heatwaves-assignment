"""Build the 2020-2025 state duration choropleth data. Requires numpy, shapely>=2.

ABS ASGS Edition 3 (2021) full State/Territory polygons, CC BY 4.0.
Join WGS84 grid centres against full polygons; simplify only map geometry.
Equal-cell annual state means are averaged equally over six years (not area
weighted). Max is the largest annual cell run. Unmatched coastal cells are
excluded, never snapped. Border ties use lowest ABS state code. Annual runs
can be truncated at file boundaries; partial years are not annualised.
"""
import csv
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from shapely import intersects_xy, prepare, make_valid
from shapely.geometry import shape, mapping, box
from shapely.geometry.polygon import orient

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://geo.abs.gov.au/arcgis/rest/services/ASGS2021/STE/MapServer/0'
YEARS = range(2020, 2026)


def main():
    cache = ROOT / 'raw/boundaries/abs_states_2021_full.geojson'
    if not cache.exists():
        query = urlencode(dict(where='1=1', outFields='state_code_2021,state_name_2021',
                               outSR=4326, returnGeometry='true', f='geojson'))
        with urlopen(SOURCE + '/query?' + query, timeout=180) as response:
            document = json.load(response)
        if 'features' not in document or document.get('exceededTransferLimit'):
            raise ValueError('ABS response is incomplete')
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(document, separators=(',', ':')), encoding='utf-8')
    document = json.loads(cache.read_text(encoding='utf-8'))
    # Exclude non-spatial ABS categories such as Outside Australia.
    features = sorted((f for f in document['features'] if f['geometry'] is not None),
                      key=lambda f: f['properties']['state_code_2021'])
    geometries = [make_valid(shape(f['geometry'])) for f in features]
    for geometry in geometries:
        prepare(geometry)
    totals = np.zeros((len(features), 2))
    maxima = np.zeros(len(features))
    reference = assignment = None
    unmatched = ties = 0
    for year in YEARS:
        path = ROOT / f'data/csiro_yearly/csiro_grid_{year}.csv'
        data = np.genfromtxt(path, delimiter=',', names=True, dtype=float,
                            usecols=('year', 'latitude', 'longitude', 'positive_ehf_days',
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
            unmatched = int(np.count_nonzero(assignment < 0))
        elif not np.array_equal(coords, reference):
            raise ValueError(f'Grid changed in {year}')
        if not np.all(data['year'] == year):
            raise ValueError('CSV year mismatch')
        for i in range(len(features)):
            selected = assignment == i
            if not selected.any():
                continue
            runs = data['longest_positive_ehf_run_days'][selected]
            days = data['positive_ehf_days'][selected]
            if not np.isfinite(runs).all() or not np.isfinite(days).all():
                raise ValueError('Missing/non-finite duration or frequency')
            totals[i] += [runs.mean(), days.mean()]
            maxima[i] = max(maxima[i], runs.max())
        print(f'{year}: {len(data):,} cells; {np.count_nonzero(assignment >= 0):,} joined')
        del data
    rows, display = [], []
    for i, feature in enumerate(features):
        count = int(np.count_nonzero(assignment == i))
        name = feature['properties']['state_name_2021']
        row = dict(state_code=feature['properties']['state_code_2021'], state=name,
                   mean_longest_run=round(float(totals[i, 0] / 6), 6) if count else None,
                   max_longest_run=int(maxima[i]) if count else None,
                   mean_heatwave_days=round(float(totals[i, 1] / 6), 6) if count else None,
                   grid_cells=count, years=6)
        rows.append(row)
        # Clip remote territories to the study extent and preserve polygon holes.
        geometry = geometries[i].intersection(box(112, -44, 155, -9)).simplify(0.01, preserve_topology=True)
        if geometry.is_empty:
            continue
        polygons = list(geometry.geoms) if geometry.geom_type == 'MultiPolygon' else [geometry]
        # D3/Vega expects clockwise exterior rings for small spherical polygons.
        # Remove subpixel islets from display only, never from the spatial join.
        coordinates = [[[[round(x, 4), round(y, 4)] for x, y in ring]
                        for ring in mapping(orient(p, sign=-1))['coordinates']]
                       for p in polygons if p.geom_type == 'Polygon' and p.area >= 0.001]
        display.append(dict(type='Feature', properties=row,
                            geometry=dict(type='MultiPolygon', coordinates=coordinates)))
    out = ROOT / 'data/csiro_state_duration_2020_2025.geojson'
    out.write_text(json.dumps(dict(type='FeatureCollection', features=display), separators=(',', ':'), allow_nan=False), encoding='utf-8')
    csv_path = ROOT / 'data/csiro_state_duration_2020_2025.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(boundary_source=SOURCE, boundary_license='ABS CC BY 4.0', years=list(YEARS),
                  grid_cells=len(reference), joined_cells=int(np.count_nonzero(assignment >= 0)),
                  unmatched_cells=unmatched, border_ties=ties, states=rows,
                  aggregation='Equal-cell annual state means, then equal-year mean; max over cells and years.',
                  caveats=['2025 may be partial; no annualisation.', 'Runs may be truncated at annual file boundaries.',
                           'Unmatched Natural Earth land-mask cells excluded from ABS join.',
                           'Display simplified by 0.01 degree, rounded to 4 decimals; islets below 0.001 square degree omitted from display only. Full geometry used for join.',
                           'Other Territories may have no represented grid centres.'])
    (ROOT / 'data/csiro_state_duration_metadata.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Map data: {out.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
