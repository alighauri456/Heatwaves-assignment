"""Summarise one year of daily CSIRO EHF on Australian land.

Dependencies: python -m pip install numpy netCDF4 shapely pyshp
Run from any directory: python scripts/process_csiro_single_year.py
Optional arguments: --input PATH --output PATH --latitude-block-size 8

Counts describe the available dates, not necessarily a complete calendar year.
Positive runs are EHF-index runs, not official heatwave event classifications.
Runs touching the first or last file date may be truncated by file boundaries.
Missing/non-finite values and missing calendar dates break consecutive runs.
No severe/extreme categories are derived.

Land boundary: Natural Earth, 1:10 million Admin 0 Countries (public domain).
https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/
The published ZIP is cached in raw/boundaries on first use. Its VERSION.txt
identifies the downloaded release. Select ADMIN=Australia in WGS84 (EPSG:4326).
Keep grid centres inside or on the boundary, including represented islands;
this is not a fractional land-area mask. The generalised coastline can omit
small islands and affect near-coast cells. No territorial waters are included.
"""

import argparse
import csv
import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from netCDF4 import Dataset, num2date
import shapefile
from shapely import intersects_xy, prepare
from shapely.geometry import shape


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
BOUNDARY_PATH = ROOT / "raw/boundaries/ne_10m_admin_0_countries.zip"


def australian_land_boundary():
    """Load published country polygons, retaining polygon holes and islands."""
    if not BOUNDARY_PATH.exists():
        with urlopen(BOUNDARY_URL, timeout=120) as response:
            payload = response.read()
        # Validate the download before caching it; never extract ZIP paths.
        with ZipFile(io.BytesIO(payload)) as archive:
            if archive.testzip() is not None:
                raise ValueError("Invalid Natural Earth boundary archive")
        BOUNDARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        BOUNDARY_PATH.write_bytes(payload)
    stem = "ne_10m_admin_0_countries"
    with ZipFile(BOUNDARY_PATH) as archive:
        version = archive.read(f"{stem}.VERSION.txt").decode().strip()
        with shapefile.Reader(shp=io.BytesIO(archive.read(f"{stem}.shp")),
                              shx=io.BytesIO(archive.read(f"{stem}.shx")),
                              dbf=io.BytesIO(archive.read(f"{stem}.dbf")),
                              encoding="utf-8") as reader:
            matches = [shape(record.shape.__geo_interface__)
                       for record in reader.iterShapeRecords()
                       if record.record.as_dict().get("ADMIN") == "Australia"]
    if len(matches) != 1 or matches[0].is_empty or not matches[0].is_valid:
        raise ValueError("Expected one valid Australian country geometry")
    geometry = matches[0]
    prepare(geometry)
    print(f"Land boundary: Natural Earth 1:10m Admin 0 Countries, version {version}")
    return geometry


def coordinate_role(name, variable):
    """Use CF metadata, with coordinate names as fallback for sparse metadata."""
    standard = str(getattr(variable, "standard_name", "")).lower()
    axis = str(getattr(variable, "axis", "")).upper()
    units = str(getattr(variable, "units", "")).lower()
    if standard == "time" or axis == "T" or " since " in units:
        return "time"
    if standard == "latitude" or units in {"degrees_north", "degree_north", "degrees_n", "degree_n"}:
        return "lat"
    if standard == "longitude" or units in {"degrees_east", "degree_east", "degrees_e", "degree_e"}:
        return "lon"
    return {"time": "time", "lat": "lat", "latitude": "lat",
            "lon": "lon", "longitude": "lon"}.get(name.lower())


def coordinates(dataset, ehf):
    result = {}
    for name, variable in dataset.variables.items():
        if variable.ndim != 1 or variable.dimensions[0] not in ehf.dimensions:
            continue
        role = coordinate_role(name, variable)
        if role:
            if role in result:
                raise ValueError(f"Ambiguous {role} coordinates")
            result[role] = variable
    if set(result) != {"time", "lat", "lon"}:
        raise ValueError("Need identifiable one-dimensional time, latitude and longitude coordinates")
    dimensions = [result[role].dimensions[0] for role in ("lat", "lon", "time")]
    if ehf.ndim != 3 or len(set(dimensions)) != 3 or set(dimensions) != set(ehf.dimensions):
        raise ValueError(f"Unsupported EHF dimensions: {ehf.dimensions}")
    return result, dimensions


def numeric_values(variable):
    return np.ma.asarray(variable[:], dtype=np.float64).filled(np.nan)


def summarise_block(values, consecutive):
    """Reduce a (latitude, longitude, time) block with only 2-D work arrays."""
    shape = values.shape[:2]
    counts = np.zeros(shape, dtype=np.int32)
    total = np.zeros(shape, dtype=np.float64)
    maximum = np.full(shape, -np.inf)
    current = np.zeros(shape, dtype=np.int32)
    longest = np.zeros(shape, dtype=np.int32)
    for index in range(values.shape[2]):
        day = np.ma.asarray(values[:, :, index], dtype=np.float64).filled(np.nan)
        finite = np.isfinite(day)
        positive = finite & (day > 0)
        counts += positive
        np.add(total, day, out=total, where=positive)
        np.maximum(maximum, day, out=maximum, where=finite)
        if not consecutive[index]:
            current.fill(0)
        current += 1
        current[~positive] = 0
        np.maximum(longest, current, out=longest)
    maximum[~np.isfinite(maximum)] = np.nan
    mean = np.full(shape, np.nan)
    np.divide(total, counts, out=mean, where=counts > 0)
    return counts, maximum, mean, longest


def process(input_path, output_path, block_size=8):
    if block_size < 1:
        raise ValueError("Latitude block size must be positive")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files")
    land = australian_land_boundary()
    with Dataset(input_path, "r") as dataset:
        ehf = dataset.variables["EHF"]
        coords, dimensions = coordinates(dataset, ehf)
        latitude = numeric_values(coords["lat"])
        longitude = numeric_values(coords["lon"])
        times = numeric_values(coords["time"])
        if any(not a.size or not np.isfinite(a).all() for a in (latitude, longitude, times)):
            raise ValueError("Coordinates must be nonempty and finite")
        time_variable = coords["time"]
        dates = num2date(times, units=time_variable.units,
                        calendar=getattr(time_variable, "calendar", "standard"))
        years = {date.year for date in dates}
        if len(years) != 1:
            raise ValueError("Expected dates from exactly one year")
        year = years.pop()
        # Read time in ascending order, even when the stored axis is descending.
        order = np.argsort(times)
        dates = dates[order]
        gaps = np.array([(dates[i] - dates[i - 1]).total_seconds() / 86400
                         for i in range(1, len(dates))])
        if np.any(gaps <= 0) or not np.allclose(gaps, np.round(gaps), rtol=0, atol=1e-8):
            raise ValueError("Expected unique daily timestamps with whole-day spacing")
        consecutive = np.r_[False, np.isclose(gaps, 1, rtol=0, atol=1e-8)]
        time_selection = slice(None) if np.array_equal(order, np.arange(len(times))) else order
        axes = [ehf.dimensions.index(dimension) for dimension in dimensions]
        print(f"EHF dimensions: {dict(zip(ehf.dimensions, ehf.shape))}")
        print(f"Available dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")
        before_mask = len(latitude) * len(longitude)
        print(f"Grid cells before masking: {before_mask:,}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = positive_cells = 0
        lat_min = lon_min = np.inf
        lat_max = lon_max = -np.inf
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["year", "latitude", "longitude", "positive_ehf_days",
                             "max_ehf", "mean_positive_ehf", "longest_positive_ehf_run_days"])
            for start in range(0, len(latitude), block_size):
                stop = min(start + block_size, len(latitude))
                # Broadcast only a small 2-D block; Shapely uses x=lon, y=lat.
                on_land = intersects_xy(land, longitude[None, :], latitude[start:stop, None])
                if not on_land.any():
                    continue
                selection = [slice(None)] * ehf.ndim
                selection[axes[0]] = slice(start, stop)
                selection[axes[2]] = time_selection
                # NetCDF reads only this spatial block; transpose is a view.
                values = np.transpose(ehf[tuple(selection)], axes)
                counts, maximum, mean, longest = summarise_block(values, consecutive)
                positive_cells += int(np.count_nonzero((counts > 0) & on_land))
                for i, lat in enumerate(latitude[start:stop]):
                    for j, lon in enumerate(longitude):
                        if not on_land[i, j]:
                            continue
                        # Blank intensity fields mean unavailable/undefined, not zero.
                        writer.writerow([year, lat, lon, int(counts[i, j]),
                                         float(maximum[i, j]) if np.isfinite(maximum[i, j]) else "",
                                         float(mean[i, j]) if np.isfinite(mean[i, j]) else "",
                                         int(longest[i, j])])
                        rows += 1
                        lat_min, lat_max = min(lat_min, lat), max(lat_max, lat)
                        lon_min, lon_max = min(lon_min, lon), max(lon_max, lon)
        print(f"Output: {output_path}")
        print(f"Rows: {rows:,}")
        print(f"Grid cells after masking: {rows:,}")
        print(f"Grid cells removed: {before_mask - rows:,}")
        if rows:
            print(f"Latitude: {lat_min:g} to {lat_max:g}")
            print(f"Longitude: {lon_min:g} to {lon_max:g}")
        print(f"Cells with at least one positive-EHF day: {positive_cells:,}")
        print(f"Output size: {output_path.stat().st_size:,} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "raw/csiro/AU_daily_EHF_epsg4326_0p11deg_1979.nc")
    parser.add_argument("--output", type=Path, default=ROOT / "data/csiro_grid_1979.csv")
    parser.add_argument("--latitude-block-size", type=int, default=8)
    args = parser.parse_args()
    process(args.input, args.output, args.latitude_block_size)


if __name__ == "__main__":
    main()
