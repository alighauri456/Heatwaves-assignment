"""Retain every second position on each axis of the original 0.11-degree grid."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data/csiro_decade_grid.csv"
destination = ROOT / "data/csiro_decade_grid_every_second.csv"
original_hash = hashlib.sha256(source.read_bytes()).hexdigest()

with source.open(newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    rows = list(reader)

# Use lattice positions, not row numbers: missing land positions must not
# change the alternating selection. Rounding absorbs source float precision.
lat_origin = min(float(row["latitude"]) for row in rows)
lon_origin = min(float(row["longitude"]) for row in rows)


def grid_index(value, origin):
    position = (float(value) - origin) / 0.11
    index = round(position)
    assert abs(position - index) < 0.001, "Coordinate is off the expected grid"
    return index


selected = [
    row for row in rows
    if grid_index(row["latitude"], lat_origin) % 2 == 0
    and grid_index(row["longitude"], lon_origin) % 2 == 0
]
with destination.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(selected)

spec = json.loads((ROOT / "charts/heatwave_frequency_map.json").read_text(encoding="utf-8-sig"))
spec["data"]["url"] = "data/csiro_decade_grid_every_second.csv"
spec["description"] += (
    " TEST COPY: retains even-indexed positions along both original grid axes,"
    " anchored at the minimum latitude and longitude. Values and 0.11-degree"
    " cell footprints are unchanged; omitted cells remain blank."
)
(ROOT / "charts/heatwave_frequency_map_reduced_test.json").write_text(
    json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
print(json.dumps({
    "original_bytes": source.stat().st_size,
    "reduced_bytes": destination.stat().st_size,
    "original_rows": len(rows),
    "reduced_rows": len(selected),
    "original_by_decade": dict(Counter(row["decade"] for row in rows)),
    "reduced_by_decade": dict(Counter(row["decade"] for row in selected)),
    "grid_origin": [lat_origin, lon_origin],
    "original_sha256": original_hash,
}, indent=2))
