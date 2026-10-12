"""Match GeoNames city coordinates to CSIRO land centres (haversine km).

Coordinates: https://www.geonames.org/search.html?country=AU and
https://www.geonames.org/au/largest-cities-in-australia.html ; Alice Springs:
https://www.geonames.org/2077895/alice-springs.html (GeoNames CC BY).
Mean positive EHF is pooled over positive days using annual counts as weights.
Annual days and longest runs are equally averaged over 2020-2025. Partial
years are not annualised; annual boundary runs may be truncated. These eight
representative locations cannot establish Australia's national intensity peak.
Uses Python standard library only; inputs and existing maps remain unchanged.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CITIES = [
    ('Perth', -31.95224, 115.861397),
    ('Adelaide', -34.928661, 138.598633),
    ('Melbourne', -37.8139965641595, 144.963322877884),
    ('Hobart', -42.87936056387943, 147.32940673828125),
    ('Sydney', -33.868, 151.207),
    ('Brisbane', -27.4679357321901, 153.028092384338),
    ('Darwin', -12.46113366159021, 130.84184646606445),
    ('Alice Springs', -23.69748, 133.88362),
]


def distance(lat1, lon1, lat2, lon2):
    a, b = math.radians(lat1), math.radians(lat2)
    h = math.sin((b-a)/2)**2 + math.cos(a)*math.cos(b)*math.sin(math.radians(lon2-lon1)/2)**2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1, h)))


def valid(row):
    keys = ['max_ehf', 'positive_ehf_days', 'longest_positive_ehf_run_days']
    return all(row[k] and math.isfinite(float(row[k])) for k in keys) and (
        float(row['positive_ehf_days']) == 0 or
        (row['mean_positive_ehf'] and math.isfinite(float(row['mean_positive_ehf']))))


def main():
    nearest = [(math.inf, None)] * len(CITIES)
    with (ROOT/'data/csiro_yearly/csiro_grid_2020.csv').open() as f:
        for row in csv.DictReader(f):
            if not valid(row):
                continue
            lat, lon = float(row['latitude']), float(row['longitude'])
            for i, (_, city_lat, city_lon) in enumerate(CITIES):
                km = distance(city_lat, city_lon, lat, lon)
                if km < nearest[i][0]:
                    nearest[i] = km, (lat, lon)
    if any(km > 20 for km, _ in nearest):
        raise ValueError('A city has no reliable land-grid match within 20 km')
    matches = {coords: i for i, (_, coords) in enumerate(nearest)}
    values = [[] for _ in CITIES]
    for year in range(2020, 2026):
        seen = set()
        with (ROOT/f'data/csiro_yearly/csiro_grid_{year}.csv').open() as f:
            for row in csv.DictReader(f):
                coords = float(row['latitude']), float(row['longitude'])
                if coords not in matches:
                    continue
                if coords in seen or int(row['year']) != year or not valid(row):
                    raise ValueError(f'Invalid matched cell in {year}: {coords}')
                seen.add(coords)
                values[matches[coords]].append(row)
        if seen != set(matches):
            raise ValueError(f'Missing matched cells in {year}')
    output = []
    for i, (name, city_lat, city_lon) in enumerate(CITIES):
        rows = values[i]
        days = sum(float(r['positive_ehf_days']) for r in rows)
        mean = sum(float(r['mean_positive_ehf']) * float(r['positive_ehf_days'])
                   for r in rows if float(r['positive_ehf_days']) > 0) / days if days else None
        km, (lat, lon) = nearest[i]
        output.append(dict(location=name, latitude=lat, longitude=lon,
                           city_latitude=city_lat, city_longitude=city_lon,
                           match_distance_km=round(km, 3),
                           mean_positive_ehf=round(mean, 6) if mean is not None else None,
                           max_ehf=max(float(r['max_ehf']) for r in rows),
                           mean_heatwave_days=round(days/6, 6),
                           mean_longest_run=round(sum(float(r['longest_positive_ehf_run_days']) for r in rows)/6, 6),
                           years=6))
    path = ROOT/'data/csiro_city_intensity_2020_2025.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps(output, indent=2))
    print(f'{path}: {len(output)} rows, {path.stat().st_size} bytes')


if __name__ == '__main__':
    main()
