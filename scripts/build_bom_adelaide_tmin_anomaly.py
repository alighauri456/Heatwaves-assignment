"""Complete-calendar-year Adelaide Tmin means and fixed early baseline.

Standard library only. Retains only years with every daily value present;
baseline is the equal-year average of complete years within 1910–1939.
Existing inputs and visualisation data are never rewritten.
"""
import calendar
import csv
import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'raw/BOM/tmin.023000.daily.csv'
    daily, station = {}, None
    with source.open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if not row['date']:
                station = dict(number=row['site number'], name=row['site name'])
                continue
            date = dt.date.fromisoformat(row['date'])
            if date in daily:
                raise ValueError(f'Duplicate date {date}')
            raw = row['minimum temperature (degC)'].strip()
            value = float(raw) if raw else None
            if value is not None and (not math.isfinite(value) or not -80 <= value <= 65):
                raise ValueError(f'Invalid temperature {date}: {raw}')
            daily[date] = value
    annual, coverage = {}, []
    for year in range(min(daily).year, max(daily).year+1):
        expected = 366 if calendar.isleap(year) else 365
        dates = [dt.date(year, 1, 1) + dt.timedelta(days=i) for i in range(expected)]
        missing = [str(d) for d in dates if daily.get(d) is None]
        values = [daily[d] for d in dates if daily.get(d) is not None]
        complete = not missing
        coverage.append(dict(year=year, expected_days=expected, observation_count=len(values),
                             included=complete, missing_dates=missing,
                             absent_dates=[str(d) for d in dates if d not in daily]))
        if complete:
            annual[year] = statistics.mean(values)
    baseline_years = [y for y in annual if 1910 <= y <= 1939]
    if len(baseline_years) < 20:
        raise ValueError('Fewer than 20 complete reference years; review baseline before changing it')
    baseline = statistics.mean(annual[y] for y in baseline_years)
    anomalies = {y: value-baseline for y, value in annual.items()}
    # Target four bands, rounded upward to a readable quarter-degree increment.
    band_size = math.ceil(max(abs(v) for v in anomalies.values()) / 4 * 4) / 4
    bands = math.ceil(max(abs(v) for v in anomalies.values()) / band_size)
    output = ROOT / 'data/bom_adelaide_tmin_anomaly.csv'
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['year', 'annual_mean_tmin', 'baseline_tmin', 'anomaly', 'observation_count'])
        writer.writeheader()
        for year, value in annual.items():
            writer.writerow(dict(year=year, annual_mean_tmin=f'{value:.9f}', baseline_tmin=f'{baseline:.9f}',
                                 anomaly=f'{anomalies[year]:.9f}', observation_count=366 if calendar.isleap(year) else 365))
    recent_years = [y for y in annual if 2000 <= y <= 2024]
    recent = statistics.mean(annual[y] for y in recent_years)
    metadata = dict(source='BOM ACORN-SAT, supplied raw/BOM/tmin.023000.daily.csv', station=station,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    source_coverage=[str(min(daily)), str(max(daily))],
                    coverage_threshold='100% of expected calendar days, including leap day; missing and absent dates both disqualify a year.',
                    included_years=list(annual), excluded_years=[r['year'] for r in coverage if not r['included']],
                    baseline_period='1910–1939', baseline_years=baseline_years, baseline_tmin=baseline,
                    baseline_method='Equal-year mean of the 20 complete-year daily Tmin means in the fixed early reference window. Not a claim of 30 complete years.',
                    annual_method='Arithmetic mean of daily minimum temperature in complete calendar years only; no imputation.',
                    unit='°C temperature anomaly, not EHF', anomaly_range=[min(anomalies.values()), max(anomalies.values())],
                    band_size=band_size, band_count=bands,
                    horizon_method='For each sign s and band k: clamp(s * anomaly - k * band_size, 0, band_size). Overlay lighter low bands first, then darker higher bands on a common zero baseline. Annual horizontal segments do not bridge excluded years. Sum of signed folded band heights reconstructs the original anomaly.',
                    precision='Nine decimals in CSV; derived from unrounded daily/annual values.',
                    recent_period='2000–2024', recent_years=recent_years, recent_mean_tmin=recent,
                    recent_minus_baseline=recent-baseline,
                    caveats=['31 incomplete years excluded; gaps remain visible.',
                             'Baseline uses 20 complete years within 1910–1939; the 2000–2024 comparison uses 21 complete years.',
                             'No fitted temporal trend or attribution is inferred from the period comparison.',
                             'Minimum daily temperature provides overnight-temperature context, not an EHF heatwave classification.'],
                    coverage=coverage)
    (ROOT / 'data/bom_adelaide_tmin_anomaly_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: metadata[k] for k in ['source_coverage', 'baseline_years', 'baseline_tmin', 'anomaly_range', 'band_size', 'band_count', 'recent_years', 'recent_mean_tmin', 'recent_minus_baseline']}, indent=2))
    print(f'{len(annual)} complete years; {output.stat().st_size} bytes')


if __name__ == '__main__':
    main()
