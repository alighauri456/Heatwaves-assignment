"""Build a 2020s-only 2x2 land-grid mean and a separate Canvas test map."""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/csiro_decade_grid.csv"
ORIGINAL_MAP = ROOT / "charts/heatwave_frequency_map.json"
OUTPUT = ROOT / "data/csiro_2020s_grid_2x2.csv"
REPORT = ROOT / "outputs/map-aggregation-comparison"
protected = [SOURCE, ORIGINAL_MAP, ROOT / "index.html", ROOT / "css/styles.css"]
hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}

with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
    rows = [r for r in csv.DictReader(handle) if r["decade"] == "2020s"]

# Restore the regular lattice despite small float errors in the source CSV.
step = Decimal("0.11")
lat_origin = Decimal(str(round(min(float(r["latitude"]) for r in rows), 2)))
lon_origin = Decimal(str(round(min(float(r["longitude"]) for r in rows), 2)))


def index(value, origin):
    position = (float(value) - float(origin)) / float(step)
    result = round(position)
    assert abs(position - result) < 0.001
    return result


groups = defaultdict(list)
positions = set()
for row in rows:
    lat_index = index(row["latitude"], lat_origin)
    lon_index = index(row["longitude"], lon_origin)
    assert (lat_index, lon_index) not in positions
    positions.add((lat_index, lon_index))
    groups[(lat_index // 2, lon_index // 2)].append(row)

aggregated = []
for (i, j), members in sorted(groups.items()):
    latitudes = [float(r["latitude"]) for r in members]
    longitudes = [float(r["longitude"]) for r in members]
    values = [float(r["mean_heatwave_days"]) for r in members]
    # Exact decimal edges share boundaries. Always draw the whole 2x2 block,
    # even at coasts where only 1-3 of its source positions contain land.
    south = lat_origin + (2 * i) * step - step / 2
    west = lon_origin + (2 * j) * step - step / 2
    aggregated.append({
        "decade": "2020s",
        "mean_heatwave_days": math.fsum(values) / len(values),
        "min_latitude": min(latitudes),
        "max_latitude": max(latitudes),
        "min_longitude": min(longitudes),
        "max_longitude": max(longitudes),
        "south": str(south), "north": str(south + 2 * step),
        "west": str(west), "east": str(west + 2 * step),
        "source_cell_count": len(members),
    })
    assert 1 <= len(members) <= 4
    assert all(float(south) <= x <= float(south + 2 * step) for x in latitudes)
    assert all(float(west) <= x <= float(west + 2 * step) for x in longitudes)

with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(aggregated[0]))
    writer.writeheader()
    writer.writerows(aggregated)

assert sum(r["source_cell_count"] for r in aggregated) == len(rows)
assert math.isclose(
    math.fsum(r["mean_heatwave_days"] * r["source_cell_count"] for r in aggregated),
    math.fsum(float(r["mean_heatwave_days"]) for r in rows), rel_tol=1e-12,
)
by_block = dict(zip(sorted(groups), aggregated))
for (i, j), block in by_block.items():
    if (i + 1, j) in by_block:
        assert block["north"] == by_block[(i + 1, j)]["south"]
    if (i, j + 1) in by_block:
        assert block["east"] == by_block[(i, j + 1)]["west"]

spec = json.loads(ORIGINAL_MAP.read_text(encoding="utf-8-sig"))
spec["description"] = (
    "TEST: mean of available land cells within each 2x2 block of the original "
    "0.11-degree grid, 2020–2025. Full 0.22-degree block footprints share edges; "
    "there is no interpolation. Coastal blocks may contain only 1–3 land cells "
    "and their rectangles extend over the rest of the block. min/max latitude "
    "and longitude describe source cell centres, not rectangle edges."
)
spec["usermeta"] = {"embedOptions": {"renderer": "canvas", "actions": False}}
spec["data"]["url"] = "data/csiro_2020s_grid_2x2.csv"
spec["data"]["format"]["parse"] = {
    field: "string" if field == "decade" else "number" for field in aggregated[0]
}
spec["transform"] = [{
    "calculate": "{'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[datum.west, datum.south], [datum.west, datum.north], [datum.east, datum.north], [datum.east, datum.south], [datum.west, datum.south]]]}}",
    "as": "grid_cell",
}]
# Keep colours comparable to the original rather than rescaling after averaging.
spec["encoding"]["color"]["scale"]["domain"] = [
    0, max(float(r["mean_heatwave_days"]) for r in rows)
]
# A fine matching stroke covers Canvas anti-alias seams along shared edges.
spec["mark"] = {
    "type": "geoshape", "strokeWidth": 0.5,
    "stroke": {"expr": "scale('color', datum.mean_heatwave_days)"},
}
spec["encoding"]["color"]["legend"]["type"] = "gradient"
spec["encoding"]["tooltip"] = [
    {"field": "decade", "type": "nominal", "title": "Decade"},
    {"field": "mean_heatwave_days", "type": "quantitative", "title": "Block average heatwave days per year", "format": ".1f"},
    {"field": "source_cell_count", "type": "quantitative", "title": "Land cells averaged", "format": "d"},
] + [
    {"field": field, "type": "quantitative", "title": title, "format": ".3f"}
    for field, title in [("south", "South edge (°)"), ("north", "North edge (°)"),
                         ("west", "West edge (°)"), ("east", "East edge (°)")]
]
(ROOT / "charts/heatwave_frequency_map_reduced_test.json").write_text(
    json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in hashes.items())
REPORT.mkdir(parents=True, exist_ok=True)
summary = {
    "original_csv_bytes": SOURCE.stat().st_size,
    "aggregated_csv_bytes": OUTPUT.stat().st_size,
    "original_2020s_cells": len(rows), "aggregated_blocks": len(aggregated),
    "blocks_by_source_cell_count": dict(Counter(r["source_cell_count"] for r in aggregated)),
    "mean_range": [min(r["mean_heatwave_days"] for r in aggregated), max(r["mean_heatwave_days"] for r in aggregated)],
    "checks": "Each original cell used exactly once; weighted sum preserved; adjacent rectangle edges match; original files unchanged.",
    "protected_sha256": {str(p.relative_to(ROOT)): digest for p, digest in hashes.items()},
}
(REPORT / "data_checks.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
