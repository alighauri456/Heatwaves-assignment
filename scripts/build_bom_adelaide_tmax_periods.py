"""Retain every valid daily BOM ACORN-SAT Adelaide Tmax in selected periods.

Standard library only. Compact CSV uses period start year and original daily
temperature, preserving observation multiplicity without duplicating dates.
Missing values are omitted, never filled; metadata retains coverage and gaps.
"""
import csv
import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERIODS = [(1910, 1929), (1940, 1959), (1970, 1989), (2000, 2019), (2020, 2024)]


def quantile(values, p):
    ordered = sorted(values)
    index = (len(ordered)-1)*p
    lower = int(index)
    return ordered[lower] + (ordered[min(lower+1, len(ordered)-1)]-ordered[lower])*(index-lower)


def main():
    source = ROOT / 'raw/BOM/tmax.023000.daily.csv'
    observations = {}
    station = None
    with source.open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if not row['date']:
                station = dict(number=row['site number'], name=row['site name'])
                continue
            date = dt.date.fromisoformat(row['date'])
            if date in observations:
                raise ValueError(f'Duplicate date: {date}')
            value = row['maximum temperature (degC)'].strip()
            temperature = float(value) if value else None
            if temperature is not None and (not math.isfinite(temperature) or not -80 <= temperature <= 65):
                raise ValueError(f'Invalid temperature: {date}, {value}')
            observations[date] = temperature
    rows, summaries = [], {}
    for first, last in PERIODS:
        start, end = dt.date(first, 1, 1), dt.date(last, 12, 31)
        dates = [start + dt.timedelta(days=i) for i in range((end-start).days+1)]
        missing = [str(d) for d in dates if observations.get(d) is None]
        valid = [(d, observations[d]) for d in dates if observations.get(d) is not None]
        values = [v for _, v in valid]
        if len(values)/len(dates) < .99:
            raise ValueError(f'Insufficient coverage in {first}–{last}; review periods before changing them')
        # Keep the input's recorded precision; no sampling or aggregation.
        rows.extend([first, str(v)] for _, v in valid)
        summaries[f'{first}–{last}'] = dict(expected_days=len(dates), valid_days=len(values),
                                          missing_days=len(missing), missing_dates=missing,
                                          median=statistics.median(values), mean=statistics.mean(values),
                                          p95=quantile(values, .95), maximum=max(values), minimum=min(values),
                                          days_at_least_35=sum(v >= 35 for v in values),
                                          share_at_least_35=sum(v >= 35 for v in values)/len(values))
    output = ROOT / 'data/bom_adelaide_tmax_periods.csv'
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['period', 'tmax'])
        writer.writerows(rows)
    metadata = dict(source='BOM ACORN-SAT, local tmax.023000.daily.csv', station=station,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    source_coverage=[str(min(observations)), str(max(observations))],
                    source_dated_rows=len(observations), unit='°C',
                    csv_fields=dict(period='Start year of the selected period; see period labels in chart.', tmax='Original daily maximum temperature in °C. Every valid daily record retained, including repeated values.'),
                    selected_daily_rows=len(rows), sampling='None. No binning, rounding or annual/monthly aggregation.',
                    density='Vega-Lite Gaussian density transform; shared 1 °C bandwidth, extent [0, 55], 221 steps; probability density, normalised independently per period.',
                    caveats=['Missing temperatures and absent dates omitted without imputation.',
                             'Final period has five years; other periods have twenty. Violin widths show relative density, not observation counts.',
                             'All seasons pooled; gaps may affect seasonal representation.',
                             'Daily maximum temperature is independent long-term context, not a CSIRO EHF heatwave classification.'],
                    summaries=summaries)
    (ROOT / 'data/bom_adelaide_tmax_periods_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f'{len(rows)} daily records; {output.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
