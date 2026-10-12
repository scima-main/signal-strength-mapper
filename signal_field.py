#!/usr/bin/env python3
"""
signal_field.py -- turn a signal.json trace log into a spatial signal field.

Reads records of the form produced by the Termux logger:

    [{"timestamp": "...", "signal": {"type":..,"dbm":..,"rsrp":..,"rssi":..},
      "location": {"latitude":..,"longitude":..,"accuracy":..,"provider":..}}, ...]

and evaluates, for every coordinate of a grid covering a given region, a value
for one or more signal channels (dbm / rsrp / rssi / ...).

------------------------------------------------------------------------
HOW `accuracy` IS TREATED (the "probability density wave function")
------------------------------------------------------------------------
Termux/Android hand `accuracy` straight through from the platform Location
object, where it is documented as "the estimated horizontal accuracy of this
location, in metres, defined as the radius of 68% confidence". In other words
the reported fix is a 2-D sample whose true position lies within `accuracy`
metres 68% of the time. That is exactly one standard deviation of a circular
Gaussian, so

    sigma_position = accuracy                      (k68 = 1.0, the default)

and each observation contributes a 2-D Gaussian probability density centred
on its reported fix:

    p_i(x) = 1 / (2*pi*s_i^2) * exp(-|x - x_i|^2 / (2*s_i^2))

    s_i^2 = (accuracy_scale * accuracy_i)^2 + bandwidth^2

The extra `bandwidth` term is a spatial-correlation floor: it stops a single
hyper-accurate fix from becoming a delta spike and says "the radio environment
is smooth over roughly this many metres". With bandwidth=0 you get the literal
per-fix density sum.

Those densities are the weights that fuse the noisy signal samples into a
posterior field. Two normalisations make them usable:

  1. peak-normalise, w_i(x) = p_i(x) * 2*pi*s_i^2 = exp(-d_i^2 / 2 s_i^2).
     This cancels in every weighted statistic but keeps sd/CI in dB rather
     than in 1/m^2.
  2. divide by the strongest kernel at that cell, so weights live in (0, 1]
     and "how much evidence is here" no longer depends on how wide the
     kernels happen to be. An observation is dropped when it weighs less
     than --min-rel-weight of the cell's best one; with the default 1e-15
     that is a hard locality cut at TAU = sqrt(-2 ln 1e-15) = 8.29 sigma.

    est(x)   = sum_i w_i * v_i / sum_i w_i                  <- the default `value`
    conf(x)  = max_i p_i(x)/p_i(x_i) * sum_i w_i            <- the evidence wave
             = sum of the *unnormalised* peak weights exp(-d_i^2 / 2 s_i^2)
    sd(x)    = sqrt( dispersion^2 + noise^2 / conf(x) )
    ess(x)   = (sum_i w_i)^2 / sum_i w_i^2                  <- diversity of evidence

Fringe semantics, on purpose: at a cell where a single fix is active, `value`
is exactly that fix's reading -- dBm does NOT decay towards -infinity just
because you are at the edge of the data. What decays is `conf`: the Gaussian
wave itself, 1.0 sitting on a fix, -> 0 at the TAU-sigma horizon. `sd` and the
CI widen as noise/sqrt(conf) so a fringe estimate honestly reports "same value,
vanishing confidence" instead of pretending to be as good as a covered cell.

`sd` is the weighted dispersion of the samples reaching x, floored at --noise
(per-sample measurement jitter) so a cell fed by one fix does not claim zero
uncertainty. `ess` is 1.0 for a single fix and grows as independent fixes
overlap; it is what to threshold on (--require-ess) for a map you trust.

Cells that no kernel reaches get null, not 0 -- 0 dBm is a real reading. Use
--fill-gaps to extrapolate them from the nearest fixes instead; those cells are
flagged "extrapolated": true and keep ess = 0.

est(x) is the conditional mean of signal strength at x -- the same thing you
get from a Gaussian likelihood with heteroscedastic noise, and (up to the
1/(2 pi s^2) constants, which cancel in the normalisation) a Nadaraya-Watson
kernel regression with an adaptive, per-observation bandwidth. Where there is
no data the weights decay to zero smoothly, so the field really does behave
like a decaying wave rather than a hard-edged Voronoi/IDW surface.

If `accuracy` is missing/0/negative (or a network/Wi-Fi fix reports something
silly), sigma falls back to `--default-accuracy`, or to the median accuracy of
the trace when that flag is left at `auto`.

Units note: 68% -> sigma is exact for a circular Gaussian; for a Rayleigh-distributed
horizontal error it is the same thing. If your provider publishes a 95% radius
instead, pass `--k68 0.51` (1/1.96).

------------------------------------------------------------------------
Pure standard library. numpy is used automatically when importable.
------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

__version__ = "1.0.0"

M_PER_DEG_LAT = 111_319.49079327358  # WGS84 mean meridian degree
# Weights are used *relative to the strongest kernel at each cell*, because the
# absolute peak of a Gaussian density (1/2*pi*s^2) depends on sigma and says
# nothing about evidence.
REL_FLOOR = 1e-15
# An observation only counts at a cell if it is within TAU sigmas of it. The
# test is done on d^2 <= (TAU*s)^2 -- no division, no exp() -- so numpy and the
# pure-Python kernel reach bit-identical membership decisions. Anything beyond
# TAU=8.29 sigmas weighs under 1e-15 of that cell's best kernel and cannot move
# a normalised statistic; leaving it in is what makes the two kernels disagree.
TAU = 8.29
EARTH_RADIUS_M = 6_371_008.8

# ramp used by the terminal heatmap: low (bad signal) -> high (good signal)
_RAMP = " .:-=+*#%@"
_BAD = "0123456789"  # 10 discrete levels, most-negative first


# --------------------------------------------------------------------------- #
# dependency-free PNG writer (RGB, 8-bit) so the field can be viewed on-device
# --------------------------------------------------------------------------- #
def _png_chunk(tag: bytes, data: bytes) -> bytes:
    import struct
    import zlib
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def _colormap(t: float):
    """t in [0,1] -> (r,g,b). Blue (worst) -> cyan -> green -> yellow -> red (best)."""
    stops = ((0.0, (30, 60, 180)), (0.25, (0, 170, 220)), (0.5, (40, 200, 110)),
             (0.75, (240, 210, 60)), (1.0, (215, 60, 45)))
    t = 0.0 if t != t else max(0.0, min(1.0, t))
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return tuple(int(round(a + (b - a) * f)) for a, b in zip(c0, c1))
    return stops[-1][1]


def write_png(path, grid, values, mask_inside, scale=4, lo=None, hi=None):
    """values: flat list (row-major j,i) of float or None. None -> transparent."""
    import struct
    import zlib
    scale = max(1, int(scale))
    w = grid.nx * scale
    h = grid.ny * scale
    got = [v for c, v in enumerate(values)
           if v is not None and v == v and (mask_inside is None or mask_inside[c])]
    if not got:
        raise ValueError("nothing to draw: every cell is null")
    lo = min(got) if lo is None else lo
    hi = max(got) if hi is None else hi
    span = (hi - lo) or 1.0

    rows = []
    for jj in range(grid.ny):
        j = grid.ny - 1 - jj              # row 0 of the image is the north edge
        px = bytearray()
        for i in range(grid.nx):
            cell = j * grid.nx + i
            v = values[cell]
            inside = mask_inside is None or mask_inside[cell]
            if v is None or v != v or not inside:
                rgb = (0, 0, 0)
                a = 0
            else:
                rgb = _colormap((v - lo) / span)
                a = 255
            for _ in range(scale):
                px += bytes(rgb) + bytes((a,))
        line = bytes(px)
        for _ in range(scale):
            rows.append(b"\x00" + line)          # filter type 0 per scanline
    raw = b"".join(rows)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)   # 8-bit RGBA
    png = (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
           + _png_chunk(b"IDAT", zlib.compress(raw, 6)) + _png_chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)
    return {"path": path, "width": w, "height": h, "min": lo, "max": hi}


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def die(msg: str, code: int = 2) -> "NoReturn":  # noqa: F821
    sys.stderr.write("error: %s\n" % msg)
    raise SystemExit(code)


def parse_time(text: str) -> datetime:
    """Parse an ISO-8601 timestamp (or 'YYYY-MM-DD HH:MM'), assume UTC if naive."""
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                    "%Y-%m-%d", "%H:%M:%S", "%H:%M"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError("cannot parse timestamp %r" % text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_duration(text: str) -> timedelta:
    """'90', '90s', '30m', '6h', '2d', '1h30m' -> timedelta."""
    s = str(text).strip().lower().replace(" ", "")
    if not s:
        raise ValueError("empty duration")
    total = 0.0
    num = ""
    seen = False
    for ch in s:
        if ch.isdigit() or ch == ".":
            num += ch
        elif ch in "smhdw":
            if not num:
                raise ValueError("bad duration %r" % text)
            mult = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[ch]
            total += float(num) * mult
            num = ""
            seen = True
        else:
            raise ValueError("bad duration %r" % text)
    if num:
        total += float(num)
        seen = True
    if not seen:
        raise ValueError("bad duration %r" % text)
    return timedelta(seconds=total)


def to_float(x):
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, str):
        try:
            f = float(x.strip())
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


def resolve_path(obj, path: str):
    """'signal.dbm' -> obj['signal']['dbm'], tolerant of missing links."""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def jget(obj, *keys, default=None):
    for k in keys:
        if isinstance(obj, dict) and obj.get(k) is not None:
            return obj[k]
    return default


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
class Projector:
    """Local equirectangular (plate carrée) projection, metres, origin (lat0, lon0)."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0 = lat0
        self.lon0 = lon0
        self.kx = M_PER_DEG_LAT * math.cos(math.radians(lat0))
        self.ky = M_PER_DEG_LAT

    def fwd(self, lat: float, lon: float):
        dlon = (lon - self.lon0 + 540.0) % 360.0 - 180.0
        return dlon * self.kx, (lat - self.lat0) * self.ky

    def inv(self, x: float, y: float):
        lat = self.lat0 + y / self.ky
        lon = self.lon0 + x / self.kx
        lon = (lon + 540.0) % 360.0 - 180.0
        return lat, lon

    def fwd_many(self, lats, lons):
        xs, ys = [], []
        for lat, lon in zip(lats, lons):
            x, y = self.fwd(lat, lon)
            xs.append(x)
            ys.append(y)
        return xs, ys


def polygon_area_m2(ring, proj: Projector) -> float:
    pts = [proj.fwd(lat, lon) for lon, lat in ring]
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def point_in_polygon(x: float, y: float, poly_xy) -> bool:
    inside = False
    n = len(poly_xy)
    j = n - 1
    for i in range(n):
        xi, yi = poly_xy[i]
        xj, yj = poly_xy[j]
        if (yi > y) != (yj > y):
            xint = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < xint:
                inside = not inside
        j = i
    return inside


# --------------------------------------------------------------------------- #
# region parsing
# --------------------------------------------------------------------------- #
def _pair(text: str):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError("%r is not 'lat,lon'" % text)
    return float(parts[1]), float(parts[0])  # -> lon, lat


def _bbox_candidates(text: str):
    """Every valid axis reading of a 4-number bbox, best guess first.

    '0.66,51.55,0.67,51.54' is genuinely ambiguous when all four numbers are
    <= 90: it may be two lon,lat corners, W,S,E,N, two lat,lon corners or
    S,W,N,E. Each axis is sorted (so any corner order works) and sanity
    checked; the caller picks the reading the data actually lives in.
    """
    parts = [p.strip() for p in text.split(",")]
    if len(parts) == 4:
        nums = [float(p) for p in parts]
        prio = ("lonlat", "latlon")          # commas: lon-first is the common form
    elif len(parts) == 1:
        toks = text.split()
        if len(toks) != 4:
            raise ValueError("--bbox wants four numbers")
        nums = [float(p) for p in toks]
        prio = ("latlon", "lonlat")          # 'lat lon lat lon'
    elif len(parts) == 2:
        p1, p2 = parts[0].split(), parts[1].split()
        if len(p1) != 2 or len(p2) != 2:
            raise ValueError("--bbox pairs want two numbers each")
        nums = [float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])]
        prio = ("latlon", "lonlat")          # spaces: 'lat lon lat lon' is documented
    else:
        raise ValueError("--bbox wants 'W,S,E,N', two corners 'lon,lat;...' style "
                         "('lon,lat,lon,lat' / 'lat,lon,lat,lon') or 'lat lon lat lon'")
    a, b, c, d = nums
    out = []

    def add(key, label, lon1, lat1, lon2, lat2):
        w, e = sorted((lon1, lon2))
        s, n = sorted((lat1, lat2))
        if (-90 <= s <= 90 and -90 <= n <= 90 and -180 <= w <= 180 and -180 <= e <= 180
                and n > s and e > w):
            box = (w, s, e, n)
            if all(box != o[2] for o in out):
                out.append((prio.index(key), label, box))

    add("lonlat", "lon,lat pairs / W,S,E,N", a, b, c, d)
    add("latlon", "lat,lon pairs / S,W,N,E", b, a, d, c)
    if not out:
        raise ValueError("bbox %r is not a valid extent in any reading" % text)
    out.sort(key=lambda t: t[0])
    return [(label, box) for _, label, box in out]


def parse_region(args) -> dict:
    """-> {'kind': 'bbox'|'polygon', 'bbox': (w,s,e,n), 'ring': [(lon,lat),...]}"""
    if args.region:
        spec = args.region.strip()
        if spec.lower().startswith("bbox:"):
            cands = _bbox_candidates(spec[5:])
            return {"kind": "bbox", "bbox": cands[0][1], "ring": None, "candidates": cands}
        ring = [_pair(tok) for tok in spec.split(";") if tok.strip()]
        if len(ring) < 3:
            raise ValueError("--region polygon needs >= 3 'lat,lon' points separated by ';'")
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        return {"kind": "polygon", "ring": ring, "bbox": ring_bbox(ring), "candidates": []}

    if args.region_geojson:
        with open(args.region_geojson) as fh:
            gj = json.load(fh)
        ring = geojson_to_ring(gj)
        return {"kind": "polygon", "ring": ring, "bbox": ring_bbox(ring), "candidates": []}

    if args.bbox:
        cands = _bbox_candidates(args.bbox)
        return {"kind": "bbox", "bbox": cands[0][1], "ring": None, "candidates": cands}

    if args.center and args.radius is not None:
        clon, clat = _pair(args.center)          # _pair returns (lon, lat)
        dlat = args.radius / M_PER_DEG_LAT
        dlon = args.radius / max(1e-9, M_PER_DEG_LAT * math.cos(math.radians(clat)))
        return {"kind": "bbox",
                "bbox": (clon - dlon, clat - dlat, clon + dlon, clat + dlat),
                "ring": None, "candidates": []}

    raise ValueError("no region given (use --bbox / --region / --center+--radius / "
                     "--region-geojson, or --auto-region to derive it from the data)")


def ring_bbox(ring):
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return (min(lons), min(lats), max(lons), max(lats))


def geojson_to_ring(gj):
    """Extract the first polygon ring from a GeoJSON Feature/Geometry/FeatureCollection."""
    if isinstance(gj, list):
        gj = {"type": "FeatureCollection", "features": gj}
    t = gj.get("type")
    if t == "FeatureCollection":
        feats = gj.get("features") or []
        if not feats:
            raise ValueError("empty FeatureCollection")
        return geojson_to_ring(feats[0])
    if t == "Feature":
        return geojson_to_ring(gj.get("geometry") or {})
    if t == "Polygon":
        return [tuple(pt[:2]) for pt in gj["coordinates"][0]]
    if t == "MultiPolygon":
        return [tuple(pt[:2]) for pt in gj["coordinates"][0][0]]
    raise ValueError("unsupported geometry type %r (need Polygon/MultiPolygon)" % t)


# --------------------------------------------------------------------------- #
# record loading
# --------------------------------------------------------------------------- #
def load_records(path: str):
    with open(path) as fh:
        raw = fh.read().strip()
    if not raw:
        return []
    data = json.loads(raw)
    if isinstance(data, dict):
        for key in ("records", "data", "features", "points"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError("%s does not contain a JSON array of records" % path)
    # tolerate a GeoJSON FeatureCollection too
    out = []
    for rec in data:
        if isinstance(rec, dict) and rec.get("type") == "Feature":
            props = rec.get("properties") or {}
            geom = rec.get("geometry") or {}
            coords = geom.get("coordinates") or [None, None]
            merged = dict(props)
            if "location" not in merged:
                merged["location"] = {"latitude": coords[1] if len(coords) > 1 else None,
                                      "longitude": coords[0] if coords else None}
            out.append(merged)
        elif isinstance(rec, dict):
            out.append(rec)
    return out


class Obs:
    __slots__ = ("t", "lat", "lon", "acc", "vals", "provider", "sigtype")

    def __init__(self, t, lat, lon, acc, vals, provider=None, sigtype=None):
        self.t = t
        self.lat = lat
        self.lon = lon
        self.acc = acc
        self.vals = vals
        self.provider = provider
        self.sigtype = sigtype


def extract_observations(records, value_paths, stats):
    """-> (obs_list, dropped_counts dict)"""
    obs = []
    dropped = {"no_location": 0, "no_signal": 0, "bad_time": 0}
    for rec in records:
        loc = rec.get("location")
        if not isinstance(loc, dict):
            # maybe flattened
            loc = rec
        lat = to_float(jget(loc, "latitude", "lat"))
        lon = to_float(jget(loc, "longitude", "lon", "long"))
        if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            dropped["no_location"] += 1
            continue

        sig = rec.get("signal") if isinstance(rec.get("signal"), dict) else rec
        vals = {}
        for name, path in value_paths.items():
            v = to_float(resolve_path(rec, path)) if "." in path else to_float(jget(sig, path))
            if v is None:
                v = to_float(resolve_path(rec, path))
            vals[name] = v
        if not any(v is not None for v in vals.values()):
            dropped["no_signal"] += 1
            continue

        acc = to_float(jget(loc, "accuracy", "horizontal_accuracy", "hAcc"))
        ts_raw = jget(rec, "timestamp", "time", "ts", "datetime")
        t = None
        if ts_raw is not None:
            if isinstance(ts_raw, (int, float)):
                v = float(ts_raw)
                t = datetime.fromtimestamp(v / 1000.0 if v > 1e11 else v, tz=timezone.utc)
            else:
                try:
                    t = parse_time(str(ts_raw))
                except ValueError:
                    dropped["bad_time"] += 1
                    t = None
        obs.append(Obs(t, lat, lon, acc, vals,
                       provider=jget(loc, "provider"),
                       sigtype=jget(sig, "type")))
    return obs, dropped


def sigma_for_obs(obs_list, args, stats):
    """Per-observation positional sigma in metres."""
    scale = args.accuracy_scale
    accs = [o.acc for o in obs_list if o.acc and o.acc > 0]
    fallback = args.default_accuracy
    if fallback == "auto":
        fallback = statistics.median(accs) if accs else 25.0
    else:
        fallback = float(fallback)
    cap = args.max_accuracy
    sigmas = []
    n_fallback = 0
    for o in obs_list:
        a = o.acc
        if a is None or a <= 0 or (cap and a > cap):
            a = fallback
            n_fallback += 1
        s = a * scale * args.k68
        sigmas.append(max(s, 1e-6))
    stats["default_accuracy_used"] = round(fallback, 3)
    stats["sigma_fallback_count"] = n_fallback
    return sigmas, fallback


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #
class Grid:
    def __init__(self, x0, y0, dx, dy, nx, ny, proj):
        self.x0, self.y0, self.dx, self.dy = x0, y0, dx, dy
        self.nx, self.ny, self.proj = nx, ny, proj
        self.n_cells = nx * ny

    def centers_lonlat(self):
        out = []
        for j in range(self.ny):
            for i in range(self.nx):
                x = self.x0 + (i + 0.5) * self.dx
                y = self.y0 + (j + 0.5) * self.dy
                out.append(self.proj.inv(x, y))
        return out

    def centers_xy(self):
        xs = [self.x0 + (i + 0.5) * self.dx for i in range(self.nx)]
        ys = [self.y0 + (j + 0.5) * self.dy for j in range(self.ny)]
        return xs, ys


def build_grid(region, proj, cell_size, snap_lonlat=None):
    w, s, e, n = region["bbox"]
    x0, y0 = proj.fwd(s, w)
    x1, y1 = proj.fwd(n, e)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    nx = max(1, int(math.ceil((x1 - x0) / cell_size)))
    ny = max(1, int(math.ceil((y1 - y0) / cell_size)))
    gx0, gy0 = x0, y0
    if snap_lonlat:
        sx, sy = proj.fwd(*snap_lonlat)
        gx0 = sx + math.floor((x0 - sx) / cell_size) * cell_size
        gy0 = sy + math.floor((y0 - sy) / cell_size) * cell_size
    return Grid(gx0, gy0, cell_size, cell_size, nx, ny, proj)


def auto_cell_size(region, proj, obs, args):
    """Pick a cell size from the region extent / spread of the data."""
    w, s, e, n = region["bbox"]
    x0, y0 = proj.fwd(s, w)
    x1, y1 = proj.fwd(n, e)
    span = max(abs(x1 - x0), abs(y1 - y0))
    if args.cells:
        target = max(2, int(args.cells))
    else:
        target = 120
    if span <= 0:
        return 10.0
    cs = span / target
    # never go below ~3 m or above 2 km for a phone-scale trace
    return min(2000.0, max(3.0, cs))


# --------------------------------------------------------------------------- #
# statistics helpers
# --------------------------------------------------------------------------- #
def weighted_quantile_sorted(vals, wts, qs):
    """vals, wts already sorted by vals ascending. qs sorted ascending in [0,1]."""
    total = sum(wts)
    if total <= 0:
        return [None] * len(qs)
    cum = 0.0
    cdfs = []
    half = 0.0
    for v, w in zip(vals, wts):
        cdfs.append((cum + half * w) / total)
        cum += w
        half = w
    out = []
    for q in qs:
        # first index with cdf >= q
        lo, hi = 0, len(cdfs) - 1
        idx = hi
        while lo <= hi:
            mid = (lo + hi) // 2
            if cdfs[mid] >= q:
                idx = mid
                hi = mid - 1
            else:
                lo = mid + 1
        out.append(vals[idx])
    return out


# --------------------------------------------------------------------------- #
# the field evaluation
# --------------------------------------------------------------------------- #
def evaluate_field(obs, sigmas, grid, args, want, value_paths):
    """Returns dict of arrays (flat, row-major over j,i) for each requested stat.

    `_ess` / `_conf` are always computed (they are what identifies cells that
    have no evidence at all) and dropped afterwards if not requested.
    """
    want = set(want) | {"ess", "conf"}
    value_names = list(value_paths.keys())
    bw2 = float(args.bandwidth) ** 2
    cutoff = float(args.cutoff)
    max_obs = int(args.max_obs) if args.max_obs else 0
    cut2 = cutoff * cutoff
    eps = 1e-300

    xs, ys = grid.centers_xy()
    ox = [None] * len(obs)
    oy = [None] * len(obs)
    for k, o in enumerate(obs):
        ox[k], oy[k] = grid.proj.fwd(o.lat, o.lon)

    n = grid.n_cells
    res = {}
    for name in value_names:
        res[name] = {}

    def _new(key):
        return [None] * n

    # allocate
    for name in value_names:
        if "value" in want:
            res[name]["value"] = _new("value")
        if "sd" in want:
            res[name]["sd"] = _new("sd")
        if "ci" in want:
            res[name]["ci_low"] = _new("ci")
            res[name]["ci_high"] = _new("ci")
        if "min" in want:
            res[name]["min"] = _new("min")
        if "max" in want:
            res[name]["max"] = _new("max")
        if "median" in want:
            res[name]["median"] = _new("median")
    if "ess" in want:
        res["_ess"] = _new("ess")
    if "conf" in want:
        res["_conf"] = _new("conf")
    res["_cover"] = [False] * n          # internal: did any kernel reach the cell?
    if "sigma" in want:
        res["_sigma"] = _new("sigma")

    use_np = _np is not None and not args.no_numpy
    if use_np:
        _eval_numpy(obs, sigmas, grid, xs, ys, ox, oy, bw2, cut2, max_obs,
                    value_names, want, res, args, eps)
    else:
        _eval_pure(obs, sigmas, grid, xs, ys, ox, oy, bw2, cut2, max_obs,
                   value_names, want, res, args, eps)
    return res


def _nearest_order(ox, oy, x, y, k):
    """Indices of the k nearest observations to (x,y), nearest first."""
    order = sorted(range(len(ox)), key=lambda kk: (ox[kk] - x) ** 2 + (oy[kk] - y) ** 2)
    return order[:k]


def _eval_pure(obs, sigmas, grid, xs, ys, ox, oy, bw2, cut2, max_obs,
               value_names, want, res, args, eps):
    ci_z = float(args.ci_z)
    noise_floor = float(args.noise)
    floor = float(args.min_rel_weight)
    tau2 = TAU * TAU
    fill = bool(args.fill_gaps)
    gap_k = max(3, max_obs or int(args.gap_obs))
    nobs = len(obs)
    for j in range(grid.ny):
        y = ys[j]
        for i in range(grid.nx):
            x = xs[i]
            cell = j * grid.nx + i

            # raw kernel weights, kept as (obs index, weight)
            raw = []
            for k in range(nobs):
                d2 = (ox[k] - x) ** 2 + (oy[k] - y) ** 2
                if cut2 and d2 > cut2:
                    continue
                s2 = sigmas[k] * sigmas[k] + bw2
                if d2 > tau2 * s2:
                    continue                      # beyond TAU sigmas: not evidence
                raw.append((k, math.exp(-0.5 * d2 / s2)))

            covered = bool(raw)
            wmax = 0.0
            if covered:
                wmax = max(w for _, w in raw)
                weights = [(k, w / wmax) for k, w in raw if w / wmax > floor]
                covered = bool(weights)
            else:
                weights = []

            res["_cover"][cell] = covered
            if not covered:
                if not fill:
                    if "ess" in want:
                        res["_ess"][cell] = 0.0
                    if "conf" in want:
                        res["_conf"][cell] = 0.0
                    continue
                # No usable evidence: extrapolate from the nearest fixes, with
                # the exponent re-based on the closest one so the relative
                # weights survive underflow. Marked extrapolated, ess = 0.
                pool = [k for k in range(nobs)
                        if not (cut2 and (ox[k] - x) ** 2 + (oy[k] - y) ** 2 > cut2)]
                near = sorted(pool,
                              key=lambda k: (ox[k] - x) ** 2 + (oy[k] - y) ** 2)[:gap_k]
                d2min = ((ox[near[0]] - x) ** 2 + (oy[near[0]] - y) ** 2) if near else 0.0
                gw = []
                for k in near:
                    d2 = (ox[k] - x) ** 2 + (oy[k] - y) ** 2
                    s2 = sigmas[k] * sigmas[k] + bw2
                    w = math.exp(-0.5 * (d2 - d2min) / s2)
                    if w > floor:
                        gw.append((k, w))
                if not gw:
                    if "ess" in want:
                        res["_ess"][cell] = 0.0
                    if "conf" in want:
                        res["_conf"][cell] = 0.0
                    continue
                gmax = max(w for _, w in gw)
                weights = [(k, w / gmax) for k, w in gw]
                wmax = 0.0            # re-based gap weights carry no evidence

            if max_obs and len(weights) > max_obs:
                weights.sort(key=lambda p: (-p[1], p[0]))   # highest weight first
                weights = weights[:max_obs]

            wsum = 0.0
            w2sum = 0.0
            for _, w in weights:
                wsum += w
                w2sum += w * w
            # absolute evidence: un-normalise the relative weights by the cell's
            # own peak. 1.0 on a fix, decaying to 0 at the horizon -- the wave.
            conf = wmax * wsum if covered else 0.0
            if "ess" in want:
                res["_ess"][cell] = ((wsum * wsum / w2sum) if w2sum > 0 else 0.0) if covered else 0.0
            if "conf" in want:
                res["_conf"][cell] = conf
            if "sigma" in want:
                res["_sigma"][cell] = (sum(w * sigmas[k] for k, w in weights) / wsum
                                       if (wsum > 0 and covered) else None)
            if wsum <= 0:
                continue

            for name in value_names:
                # (value, observation index, weight): the index keeps tie-breaks
                # identical to the numpy path's stable argsort over observations
                pairs = [(obs[k].vals.get(name), k, w) for k, w in weights]
                pairs = [(v, k, w) for v, k, w in pairs if v is not None]
                if not pairs:
                    continue
                wtot = sum(w for _, _, w in pairs)
                if wtot <= 0:
                    continue
                est = sum(v * w for v, _, w in pairs) / wtot
                r = res[name]
                if "value" in want:
                    r["value"][cell] = est
                if "sd" in want or "ci" in want:
                    # scatter of the samples reaching this cell, plus the
                    # measurement-noise term scaled by 1/conf: as the evidence
                    # wave decays the uncertainty grows without bound
                    var = sum(w * (v - est) ** 2 for v, _, w in pairs) / wtot
                    sd = (math.sqrt(var + noise_floor * noise_floor / conf)
                          if conf > 0 else None)
                    if "sd" in want:
                        r["sd"][cell] = sd
                if "ci" in want and sd is not None:
                    r["ci_low"][cell] = est - ci_z * sd
                    r["ci_high"][cell] = est + ci_z * sd
                if "min" in want:
                    r["min"][cell] = min(v for v, _, _ in pairs)
                if "max" in want:
                    r["max"][cell] = max(v for v, _, _ in pairs)
                if "median" in want:
                    sp = sorted(pairs, key=lambda p: (p[0], p[1]))
                    r["median"][cell] = weighted_quantile_sorted(
                        [v for v, _, _ in sp], [w for _, _, w in sp], [0.5])[0]


def _eval_numpy(obs, sigmas, grid, xs, ys, ox, oy, bw2, cut2, max_obs,
                value_names, want, res, args, eps):
    """Vectorised kernel evaluation.

    Per-chunk layout is (nx, nrows, nobs), flattened row-major to (cells, nobs)
    so slices line up with the (j, i) cell index used everywhere else. Chunks
    shrink automatically if an allocation is refused.
    """
    np = _np
    nx, nobs = grid.nx, len(obs)
    XA = np.asarray(xs, dtype=np.float64).reshape(nx, 1, 1)
    OX = np.asarray(ox, dtype=np.float64).reshape(1, 1, nobs)
    OY = np.asarray(oy, dtype=np.float64).reshape(1, 1, nobs)
    S = np.asarray(sigmas, dtype=np.float64)
    S2 = (S * S + bw2).reshape(1, 1, nobs)
    V = {name: np.array([o.vals.get(name) for o in obs], dtype=np.float64)
         for name in value_names}
    ci_z = float(args.ci_z)
    noise_floor = float(args.noise)
    floor = float(args.min_rel_weight)
    fill = bool(args.fill_gaps)
    gap_k = max(3, max_obs or int(args.gap_obs))

    budget = float(args.chunk_budget)
    j0 = 0
    while j0 < grid.ny:
        nr = max(1, min(grid.ny - j0, int(budget / max(1.0, nx * nobs * 3.0))))
        try:
            _chunk_numpy(np, j0, nr, grid, XA, OX, OY, S, S2, cut2, V, ys, value_names,
                         want, res, ci_z, max_obs, fill, gap_k, nobs, nx,
                         noise_floor, floor)
        except MemoryError:
            if budget <= 1e5:
                raise
            budget /= 4.0
            continue
        j0 += nr


def _lst(a):
    """float array -> python list with NaN/inf mapped to None (JSON null)."""
    return [None if (v != v or v == float("inf") or v == float("-inf")) else v
            for v in a.tolist()]


def _chunk_numpy(np, j0, nr, grid, XA, OX, OY, S, S2, cut2, V, ys, value_names,
                 want, res, ci_z, max_obs, fill, gap_k, nobs, nx,
                 noise_floor, floor):
    YA = np.asarray(ys[j0:j0 + nr], dtype=np.float64).reshape(1, nr, 1)
    D2 = (OX - XA) ** 2 + (OY - YA) ** 2                        # (nx, nr, nobs)
    if cut2:
        D2 = np.where(D2 > cut2, np.inf, D2)
    base = j0 * nx
    nc = nr * nx

    def flat(a):
        if a.ndim == 3:
            return np.ascontiguousarray(np.transpose(a, (1, 0, 2))).reshape(nc, nobs)
        return np.ascontiguousarray(np.transpose(a, (1, 0))).reshape(nc)

    # Membership test on d^2 vs (TAU*s)^2 -- same comparison, same operand order
    # as the pure kernel, so the two agree bit for bit on which cells have data.
    member = D2 <= (TAU * TAU) * S2
    with np.errstate(over="ignore", invalid="ignore"):
        W = np.where(member, np.exp(-0.5 * np.where(member, D2, 0.0) / S2), 0.0)
    # normalise each cell by its own strongest kernel: absolute peak height
    # depends on sigma and carries no information about evidence
    wmax = W.max(axis=2, keepdims=True)
    W = np.where(wmax > 0.0, W / np.where(wmax > 0.0, wmax, 1.0), 0.0)
    W = np.where(W > floor, W, 0.0)
    has = (W > 0.0).any(axis=2)                                   # (nx, nr) cell has evidence
    has_flat = flat(has)
    wmax_flat = np.ascontiguousarray(np.transpose(wmax, (1, 0, 2))).reshape(nc)
    del wmax, member

    if fill and not has_flat.all():
        k = min(gap_k, nobs)
        # full stable argsort, not argpartition: breaks distance ties by
        # observation index exactly like sorted() does in the pure kernel
        keep = np.argsort(D2, axis=2, kind="stable")[:, :, :k]     # k nearest
        m = np.zeros(D2.shape, dtype=bool)
        np.put_along_axis(m, keep, True, axis=2)
        D2k = np.where(m, D2, np.inf)
        d2min = D2k.min(axis=2, keepdims=True)                     # (nx, nr, 1)
        with np.errstate(over="ignore", invalid="ignore"):
            Wf = np.exp(-0.5 * (D2k - d2min) / S2)                 # re-based: no underflow
        Wf = np.nan_to_num(Wf, nan=0.0, posinf=0.0, neginf=0.0)
        Wf = np.where(m, Wf, 0.0)
        fmax = Wf.max(axis=2, keepdims=True)
        Wf = np.where(fmax > 0.0, Wf / np.where(fmax > 0.0, fmax, 1.0), 0.0)
        Wf = np.where(Wf > floor, Wf, 0.0)
        # `has` is (nx, nr) -- the axis order of W. has_flat is in *cell* order
        # (j*nx+i), so it must not be reshaped back as (nx, nr, 1).
        W = np.where(has[:, :, None], W, Wf)
        del m, Wf, keep, D2k, d2min, fmax

    if max_obs and nobs > max_obs:
        k = min(max_obs, nobs)
        keep = np.argpartition(-W, k - 1, axis=2)[:, :, :k]
        m = np.zeros(W.shape, dtype=bool)
        np.put_along_axis(m, keep, True, axis=2)
        W = W * m
        del m, keep

    Wt = flat(W)                                                # (cells, nobs)
    del W, D2
    wsum = Wt.sum(axis=1)
    # not einsum: its BLAS/pairwise reduction can flush subnormal weights to 0,
    # which would make ess disagree with the pure-Python path
    w2sum = (Wt * Wt).sum(axis=1)
    ok = wsum > 0
    res["_cover"][base:base + nc] = has_flat.tolist()

    conf = np.where(has_flat, wmax_flat * wsum, 0.0)   # the evidence wave
    if "ess" in want:
        ess = np.where(w2sum > 0, wsum * wsum / np.maximum(w2sum, 1e-300), 0.0)
        # re-based gap weights are a plotting convenience, not evidence
        ess = np.where(has_flat, ess, 0.0)
        res["_ess"][base:base + nc] = ess.tolist()
    if "conf" in want:
        res["_conf"][base:base + nc] = conf.tolist()
    if "sigma" in want:
        a = np.full(nc, np.nan)
        sel = ok & has_flat
        a[sel] = (Wt[sel] @ S) / wsum[sel]
        res["_sigma"][base:base + nc] = _lst(a)

    for name in value_names:
        v = V[name]
        finite = np.isfinite(v)
        Wo = Wt[:, finite]
        vo = v[finite]
        wtot = Wo.sum(axis=1)
        vok = wtot > 0
        r = res[name]
        est = np.full(nc, np.nan)
        sd = np.full(nc, np.nan)
        if vok.any():
            est[vok] = (Wo[vok] @ vo) / wtot[vok]
        if "value" in want:
            r["value"][base:base + nc] = _lst(est)
        if ("sd" in want or "ci" in want) and vok.any():
            ex2 = np.full(nc, np.nan)
            ex2[vok] = (Wo[vok] @ (vo * vo)) / wtot[vok]
            var = np.maximum(ex2 - est * est, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                sd = np.sqrt(var + noise_floor * noise_floor / np.where(conf > 0, conf, np.nan))
            if "sd" in want:
                r["sd"][base:base + nc] = _lst(sd)
        if "ci" in want:
            r["ci_low"][base:base + nc] = _lst(est - ci_z * sd)
            r["ci_high"][base:base + nc] = _lst(est + ci_z * sd)
        if "min" in want or "max" in want or "median" in want:
            have = Wo > 0
            any_have = have.any(axis=1)
            if "min" in want:
                vm = np.where(have, vo[None, :], np.inf).min(axis=1)
                vm[~any_have] = np.nan
                r["min"][base:base + nc] = _lst(vm)
            if "max" in want:
                vm = np.where(have, vo[None, :], -np.inf).max(axis=1)
                vm[~any_have] = np.nan
                r["max"][base:base + nc] = _lst(vm)
            if "median" in want:
                med = np.full(nc, np.nan)
                order = np.argsort(vo, kind="stable")
                vs = vo[order]
                for c in np.nonzero(any_have)[0]:
                    w = Wo[c][order]
                    nz = w > 0
                    med[c] = weighted_quantile_sorted(vs[nz].tolist(), w[nz].tolist(), [0.5])[0]
                r["median"][base:base + nc] = _lst(med)
        del Wo
    del Wt


def _blank_uncovered(res, grid):
    """Cells no kernel reached keep null values -- 0.0 is a legal dB reading."""
    cover = res.get("_cover")
    if cover is None:
        return
    blanks = [c for c in range(grid.n_cells) if not cover[c]]
    if not blanks:
        return
    for name in [k for k in res if not k.startswith("_")]:
        for arr in res[name].values():
            for c in blanks:
                arr[c] = None


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def _progress(args, done, total, label="features"):
    """Periodic write-progress line. stdout, unless stdout carries the payload."""
    every = int(getattr(args, "progress_every", 0) or 0)
    if every <= 0 or done <= 0 or done % every:
        return
    stream = sys.stderr if args.output == "-" else sys.stdout
    pct = 100.0 * done / max(1, total)
    msg = "progress: %s/%s %s (%.1f%%)" % (format(done, ","), format(total, ","), label, pct)
    if stream.isatty():
        stream.write("\r" + msg.ljust(64))
        stream.flush()
    else:
        stream.write(msg + "\n")
        stream.flush()


def _progress_end(args):
    if int(getattr(args, "progress_every", 0) or 0) <= 0:
        return
    stream = sys.stderr if args.output == "-" else sys.stdout
    if stream.isatty():
        stream.write("\r" + " " * 64 + "\r")
        stream.flush()


def rnd(x, p):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    if p <= 0:
        return round(f)
    return round(f, p)


def build_geojson(grid, res, args, value_names, mask_inside, stats):
    feats = []
    lons = []
    lats = []
    for j in range(grid.ny):
        for i in range(grid.nx):
            x = grid.x0 + (i + 0.5) * grid.dx
            y = grid.y0 + (j + 0.5) * grid.dy
            lat, lon = grid.proj.inv(x, y)
            lons.append(lon)
            lats.append(lat)
    emitted = 0
    for cell in range(grid.n_cells):
        if mask_inside is not None and not mask_inside[cell]:
            continue
        props = {}
        for name in value_names:
            r = res[name]
            if "value" in r:
                props[name] = rnd(r["value"][cell], args.precision)
            if "sd" in r:
                props["%s_sd" % name] = rnd(r["sd"][cell], args.precision)
            if "ci_low" in r:
                props["%s_ci_low" % name] = rnd(r["ci_low"][cell], args.precision)
                props["%s_ci_high" % name] = rnd(r["ci_high"][cell], args.precision)
            if "min" in r:
                props["%s_min" % name] = rnd(r["min"][cell], args.precision)
            if "max" in r:
                props["%s_max" % name] = rnd(r["max"][cell], args.precision)
            if "median" in r:
                props["%s_median" % name] = rnd(r["median"][cell], args.precision)
        if "_cover" in res and args.fill_gaps:
            props["extrapolated"] = not bool(res["_cover"][cell])
        if "_ess" in res:
            props["ess"] = rnd(res["_ess"][cell], 3)
        if "_conf" in res:
            props["conf"] = rnd(res["_conf"][cell], 4)
        if "_sigma" in res:
            props["sigma_m"] = rnd(res["_sigma"][cell], 2)
        if not any(v is not None for k, v in props.items() if k in value_names):
            if args.skip_empty:
                continue
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [round(lons[cell], 7), round(lats[cell], 7)]},
            "properties": props,
        })
        emitted += 1
        _progress(args, emitted, grid.n_cells)
    _progress_end(args)
    gj = {
        "type": "FeatureCollection",
        "name": args.out_name or "signal_field",
        "properties": stats,
        "features": feats,
    }
    return gj


def write_csv(grid, res, args, value_names, mask_inside, path):
    lons, lats = [], []
    for j in range(grid.ny):
        for i in range(grid.nx):
            x = grid.x0 + (i + 0.5) * grid.dx
            y = grid.y0 + (j + 0.5) * grid.dy
            lat, lon = grid.proj.inv(x, y)
            lons.append(lon)
            lats.append(lat)
    cols = ["longitude", "latitude"]
    for name in value_names:
        r = res[name]
        for k in ("value", "sd", "ci_low", "ci_high", "min", "max", "median"):
            if k in r:
                cols.append(name if k == "value" else "%s_%s" % (name, k))
    if "_cover" in res and args.fill_gaps:
        cols.append("extrapolated")
    if "_ess" in res:
        cols.append("ess")
    if "_conf" in res:
        cols.append("conf")
    if "_sigma" in res:
        cols.append("sigma_m")
    written = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for cell in range(grid.n_cells):
            if mask_inside is not None and not mask_inside[cell]:
                continue
            row = [round(lons[cell], 7), round(lats[cell], 7)]
            for name in value_names:
                r = res[name]
                for k in ("value", "sd", "ci_low", "ci_high", "min", "max", "median"):
                    if k in r:
                        row.append(rnd(r[k][cell], args.precision))
            if "_cover" in res and args.fill_gaps:
                row.append(bool(res["_cover"][cell]) is False)
            if "_ess" in res:
                row.append(rnd(res["_ess"][cell], 3))
            if "_conf" in res:
                row.append(rnd(res["_conf"][cell], 4))
            if "_sigma" in res:
                row.append(rnd(res["_sigma"][cell], 2))
            if args.skip_empty and all(v is None for v in row[2:2 + len(value_names)]):
                continue
            w.writerow(row)
            written += 1
            _progress(args, written, grid.n_cells, "rows")
    _progress_end(args)


def render_heatmap(grid, res, value_name, mask_inside, width=78, height=24, ramp=_RAMP):
    arr = res[value_name].get("value")
    if arr is None:
        return ""
    vals = [v for c, v in enumerate(arr)
            if v is not None and math.isfinite(v) and (mask_inside is None or mask_inside[c])]
    if not vals:
        return "(no data)"
    lo, hi = min(vals), max(vals)
    lines = []
    nlev = len(ramp)
    for jj in range(height - 1, -1, -1):
        j = int(jj * (grid.ny - 1) / max(1, height - 1)) if grid.ny > 1 else 0
        row = []
        for ii in range(width):
            i = int(ii * (grid.nx - 1) / max(1, width - 1)) if grid.nx > 1 else 0
            cell = j * grid.nx + i
            v = arr[cell]
            if v is None or not math.isfinite(v) or (mask_inside is not None and not mask_inside[cell]):
                row.append(" ")
                continue
            if hi - lo < 1e-9:
                row.append(ramp[-1])
            else:
                k = int((v - lo) / (hi - lo) * (nlev - 1) + 0.5)
                row.append(ramp[max(0, min(nlev - 1, k))])
        lines.append("".join(row))
    header = "%s  [%.1f .. %.1f]   '%s' = low -> high" % (value_name, lo, hi, ramp)
    return header + "\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_value_spec(specs):
    """'dbm' -> {'dbm':'signal.dbm'}; 'sinr=signal.sinr' -> {'sinr':'signal.sinr'}"""
    out = {}
    for s in specs:
        s = s.strip()
        if not s:
            continue
        if "=" in s:
            name, path = [p.strip() for p in s.split("=", 1)]
        else:
            name, path = s, "signal." + s
        out[name] = path
    if not out:
        raise ValueError("no --value channels given")
    return out


def build_parser():
    p = argparse.ArgumentParser(
        prog="signal_field.py",
        description="Project a signal.json trace onto a coordinate grid using the GPS "
                    "accuracy values as a Gaussian probability density.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
accuracy model:
  Android reports Location.accuracy as the radius of 68% horizontal confidence,
  i.e. one sigma of a circular Gaussian. Each fix therefore contributes
      p_i(x) = exp(-d^2 / 2 s_i^2) / (2 pi s_i^2),   s_i^2 = (k68*acc_i)^2 + bandwidth^2
  and the value at a coordinate is the p-weighted mean of the signal samples.

examples:
  # field over a bounding box, 10 m cells, dbm only
  ./signal_field.py -i signal.json --bbox 12.9716,77.5946,12.9816,77.6046 --cell-size 10

  # everything in the log, auto region, auto cell size, print an ASCII heatmap
  ./signal_field.py -i signal.json --auto-region --heatmap

  # only a time window, rsrp + rssi, include uncertainty and coverage
  ./signal_field.py -i signal.json --auto-region --value rsrp --value rssi \\
      --since 2026-10-09T08:00Z --until 2026-10-09T12:00Z --stats value,sd,ess

  # polygon region, csv out
  ./signal_field.py -i signal.json --region '12.971,77.594;12.980,77.590;12.985,77.605' \\
      --cell-size 5 -o field.geojson --csv field.csv
""")
    p.add_argument("-i", "--input", default="signal.json", help="input JSON log (default: signal.json)")
    p.add_argument("-o", "--output", default="-", help="output path, '-' for stdout (default: -)")
    p.add_argument("--out-format", choices=["geojson", "json", "csv"], default="geojson",
                   help="geojson = FeatureCollection of points (default), json = nested grid, csv = flat table")
    p.add_argument("--csv", metavar="PATH", help="also write a CSV alongside the main output")
    p.add_argument("--png", metavar="PATH",
                   help="also write the primary channel as a transparent-background PNG raster")
    p.add_argument("--png-scale", type=int, default=6, metavar="N",
                   help="pixels per grid cell in the PNG (default 6)")
    p.add_argument("--png-range", metavar="LO,HI",
                   help="clamp the PNG colour scale to these values (default: the data range)")
    p.add_argument("--out-name", help="name property for the output collection")

    g = p.add_argument_group("region")
    g.add_argument("--bbox", metavar="W,S,E,N",
                   help="bounding box in any of: 'w,s,e,n', 's,w,n,e', two corners "
                        "'lon,lat,lon,lat' or 'lat,lon,lat,lon' (any corner order), "
                        "'lat lon lat lon'. Ambiguous readings are resolved towards "
                        "the one containing your data (a note is printed)")
    g.add_argument("--region", metavar="SPEC",
                   help="'bbox:W,S,E,N' or a polygon 'lat,lon;lat,lon;lat,lon[;...]'")
    g.add_argument("--region-geojson", metavar="FILE", help="GeoJSON file containing a Polygon/MultiPolygon")
    g.add_argument("--center", metavar="LAT,LON", help="centre for --radius")
    g.add_argument("--radius", type=float, metavar="M", help="half-width in metres around --center")
    g.add_argument("--auto-region", action="store_true",
                   help="use the bounding box of the (time-filtered) observations, padded")
    g.add_argument("--pad", type=float, default=0.0, metavar="M",
                   help="pad the region by this many metres (default 0)")
    g.add_argument("--auto-quantile", type=float, default=2.0, metavar="PCT",
                   help="with --auto-region, trim this percentile of outlier fixes off each "
                        "side so one 1 km network fix does not define the region (default 2; 0 = full extent)")

    g = p.add_argument_group("grid")
    g.add_argument("--cell-size", type=float, metavar="M", help="grid cell edge in metres")
    g.add_argument("--cells", type=int, metavar="N", help="target cells along the longest axis (used with --auto-region)")
    g.add_argument("--snap-to", metavar="LAT,LON", help="align the grid origin to this coordinate")

    g = p.add_argument_group("time range (inclusive; UTC)")
    g.add_argument("--since", "--from", dest="since", metavar="ISO8601", help="start of range")
    g.add_argument("--until", "--to", dest="until", metavar="ISO8601", help="end of range")
    g.add_argument("--last", metavar="DUR", help="only the most recent DUR of the log (e.g. 30m, 6h, 2d)")
    g.add_argument("--max-age", metavar="DUR", help="drop records older than DUR before now")
    g.add_argument("--require-timestamp", action="store_true",
                   help="drop records whose timestamp is missing/unparseable when a range is given")

    g = p.add_argument_group("model")
    g.add_argument("--value", action="append", metavar="CHANNEL",
                   help="signal channel to evaluate; 'dbm' means signal.dbm, or use name=path "
                        "(e.g. sinr=signal.sinr). repeatable. default: dbm,rsrp,rssi (whichever exist)")
    g.add_argument("--bandwidth", type=float, default=20.0, metavar="M",
                   help="spatial correlation floor added in quadrature to accuracy (default 20; 0 = raw density)")
    g.add_argument("--accuracy-scale", type=float, default=1.0,
                   help="multiplier applied to reported accuracy (default 1.0)")
    g.add_argument("--k68", type=float, default=1.0,
                   help="accuracy->sigma factor: 1.0 if accuracy is a 68%% radius (Android default), "
                        "0.51 if it is a 95%% radius")
    g.add_argument("--default-accuracy", default="auto", metavar="M|auto",
                   help="sigma to use when accuracy is missing/<=0 (default auto = median of the trace)")
    g.add_argument("--max-accuracy", type=float, default=0, metavar="M",
                   help="treat accuracies above this as missing (0 = no limit)")
    g.add_argument("--provider", action="append", metavar="NAME",
                   help="only use these location providers (gps/network/fused/passive)")
    g.add_argument("--cutoff", type=float, default=0.0, metavar="M",
                   help="hard distance cutoff in metres (0 = use the 6-sigma kernel tail)")
    g.add_argument("--max-obs", type=int, default=0, metavar="N",
                   help="use at most the N highest-weight observations per cell "
                        "(0 = all; approximation that speeds up big logs)")
    g.add_argument("--fill-gaps", action="store_true",
                   help="also give cells that fall outside every kernel's 6-sigma tail a "
                        "value extrapolated from the nearest fixes (default: leave them null)")
    g.add_argument("--gap-obs", type=int, default=8, metavar="N",
                   help="how many nearest fixes --fill-gaps extrapolates from (default 8)")
    g.add_argument("--min-rel-weight", type=float, default=REL_FLOOR, metavar="R",
                   help="per-cell evidence threshold: ignore observations whose kernel weighs "
                        "less than R times the strongest one at that cell, and treat a cell "
                        "where nothing survives as having no data (default %g). Raise it (e.g. "
                        "1e-3 ~ 3.7 sigma, 1e-6 ~ 5.3 sigma) to make the field strictly local." % REL_FLOOR)
    p.add_argument("--min-weight", type=float, default=0.0, metavar="W",
                   help="null out cells whose evidence (ess, or n_eff if that is all you "
                        "asked for) is below W; ess is 1.0 for a single fix")
    p.add_argument("--min-conf", type=float, default=0.0, metavar="W",
                   help="null out cells whose confidence wave conf(x) is below W. "
                        "conf is the total evidence in equivalent full-strength fixes "
                        "(1.0 on a lone fix, more where fixes overlap), so W=1 keeps "
                        "cells with at least one fix's worth of evidence; on a single "
                        "fix W=0.5 ~ inside 1.2 sigma, W=1e-3 ~ 3.7 sigma")
    p.add_argument("--require-ess", type=float, default=0.0, metavar="E",
                   help="null out cells supported by fewer than E independent fixes "
                        "(E=2 is a good default for a map you intend to trust; "
                        "E=0 keeps everything the kernels reach)")
    g.add_argument("--stats", default="value",
                   help="comma list from: value,sd,ci,min,max,median,ess,conf,sigma "
                        "(default: value; 'neff' is accepted as an alias of conf)")
    p.add_argument("--ci-z", type=float, default=1.96, help="z for --stats ci (default 1.96)")
    g.add_argument("--noise", type=float, default=3.0, metavar="V",
                   help="per-sample measurement noise floor in the value's units; --stats sd "
                        "never reports less than this (default 3.0 dB, typical RSRP/RSSI jitter)")
    p.add_argument("--precision", type=int, default=2, help="decimal places in output values (default 2)")
    p.add_argument("--skip-empty", action="store_true", help="omit cells with no value")
    p.add_argument("--no-mask", action="store_true", help="with a polygon region, also emit cells outside it")
    p.add_argument("--heatmap", action="store_true", help="print an ASCII heatmap to stderr")
    p.add_argument("--no-numpy", action="store_true", help="force the pure-Python kernel")
    p.add_argument("--chunk-budget", type=float, default=8e6, metavar="N",
                   help="max array elements per numpy chunk; shrunk automatically on MemoryError")
    p.add_argument("--progress-every", type=int, default=2500, metavar="N",
                   help="print a progress line every N features while writing "
                        "(default 2500; 0 disables). Goes to stdout, or to stderr "
                        "when stdout carries the payload (-o -)")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress the summary on stderr")
    p.add_argument("-V", "--version", action="version", version="signal_field.py " + __version__)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    want = set(s.strip() for s in args.stats.split(",") if s.strip())
    if "neff" in want:            # old name for the evidence wave
        want.discard("neff")
        want.add("conf")
    bad = want - {"value", "sd", "ci", "min", "max", "median", "ess", "conf", "sigma"}
    if bad:
        die("unknown --stats entries: %s" % ", ".join(sorted(bad)))
    if not want:
        want = {"value"}

    # ---- load ------------------------------------------------------------
    try:
        records = load_records(args.input)
    except FileNotFoundError:
        die("input file not found: %s" % args.input)
    except json.JSONDecodeError as e:
        die("%s is not valid JSON: %s" % (args.input, e))
    if not records:
        die("no records found in %s" % args.input)

    # default channels: whatever the log actually carries
    if args.value:
        value_paths = parse_value_spec(args.value)
    else:
        probe = records[0].get("signal") if isinstance(records[0].get("signal"), dict) else {}
        chans = [c for c in ("dbm", "rsrp", "rssi", "rsrq", "sinr", "level") if c in probe]
        value_paths = parse_value_spec(chans or ["dbm"])

    obs_all, dropped = extract_observations(records, value_paths, None)
    if not obs_all:
        die("no usable records (need location.latitude/longitude and a signal value); "
            "dropped=%s" % dropped)

    # ---- time filter -----------------------------------------------------
    try:
        since = parse_time(args.since) if args.since else None
        until = parse_time(args.until) if args.until else None
    except ValueError as e:
        die("bad --since/--until: %s" % e)
    if since and until and until < since:
        die("--until %s is before --since %s" % (until.isoformat(), since.isoformat()))
    try:
        if args.last:
            times = [o.t for o in obs_all if o.t]
            if not times:
                die("--last given but no record has a usable timestamp")
            since = max(times) - parse_duration(args.last)
        if args.max_age:
            cut = datetime.now(timezone.utc) - parse_duration(args.max_age)
            since = cut if since is None else max(since, cut)
    except ValueError as e:
        die("bad --last/--max-age: %s" % e)
    range_active = since is not None or until is not None

    obs = []
    n_no_ts = 0
    for o in obs_all:
        if range_active:
            if o.t is None:
                n_no_ts += 1
                if args.require_timestamp:
                    continue
                # without a timestamp we cannot place it in the window: skip it
                continue
            if since and o.t < since:
                continue
            if until and o.t > until:
                continue
        obs.append(o)
    if not obs:
        die("no records left after the time filter (%s .. %s)" %
            (since.isoformat() if since else "-inf", until.isoformat() if until else "+inf"))

    # ---- provider filter -------------------------------------------------
    if args.provider:
        keep = set(p.lower() for p in args.provider)
        obs = [o for o in obs if (o.provider or "").lower() in keep] or obs
        if not obs:
            die("no records left after --provider filter")

    # ---- region ----------------------------------------------------------
    if args.auto_region:
        lats = [o.lat for o in obs]
        lons = [o.lon for o in obs]
        midlat = statistics.median(lats)
        midlon = statistics.median(lons)
        aproj = Projector(midlat, midlon)
        px, py = aproj.fwd_many(lats, lons)
        lo_q, hi_q = args.auto_quantile / 100.0, 1.0 - args.auto_quantile / 100.0

        def _q(vals, q):
            v = sorted(vals)
            if len(v) == 1:
                return v[0]
            pos = q * (len(v) - 1)
            k = int(math.floor(pos))
            frac = pos - k
            return v[k] * (1 - frac) + v[min(k + 1, len(v) - 1)] * frac

        # robust extent: a single 1.6 km network fix must not define the region
        x_lo, x_hi = _q(px, lo_q), _q(px, hi_q)
        y_lo, y_hi = _q(py, lo_q), _q(py, hi_q)
        if x_hi - x_lo < 1e-6 and y_hi - y_lo < 1e-6:
            x_lo, x_hi = min(px), max(px)
            y_lo, y_hi = min(py), max(py)
        x_lo -= args.pad
        x_hi += args.pad
        y_lo -= args.pad
        y_hi += args.pad
        s_lat, w_lon = aproj.inv(x_lo, y_lo)
        n_lat, e_lon = aproj.inv(x_hi, y_hi)
        region = {"kind": "bbox", "bbox": (w_lon, s_lat, e_lon, n_lat), "ring": None}
    else:
        try:
            region = parse_region(args)
        except ValueError as e:
            die(str(e))

    # A 4-number bbox with every value <= 90 is ambiguous; prefer the reading
    # that actually contains the (time-filtered) trace. This never changes a
    # command that already worked: it only fires when the first reading is empty.
    cands = region.get("candidates") or []
    cand_counts = []
    if len(cands) > 1:
        cand_counts = [sum(1 for o in obs
                           if bb[0] <= o.lon <= bb[2] and bb[1] <= o.lat <= bb[3])
                       for _, bb in cands]
        pick = next((i for i, c in enumerate(cand_counts) if c > 0), 0)
        if pick:
            sys.stderr.write("note: region read as '%s' (%d obs inside); the '%s' "
                             "reading contains none\n"
                             % (cands[pick][0], cand_counts[pick], cands[0][0]))
            region["bbox"] = cands[pick][1]

    if not args.auto_region and args.pad:
        w, s, e_, n = region["bbox"]
        pdlat = args.pad / M_PER_DEG_LAT
        midlat = (s + n) / 2
        pdlon = args.pad / max(1e-9, M_PER_DEG_LAT * math.cos(math.radians(midlat)))
        region["bbox"] = (w - pdlon, s - pdlat, e_ + pdlon, n + pdlat)

    w, s, e_, n = region["bbox"]
    proj = Projector((s + n) / 2.0, (w + e_) / 2.0)

    cell_size = args.cell_size
    if not cell_size:
        cell_size = auto_cell_size(region, proj, obs, args)
    if cell_size <= 0:
        die("--cell-size must be > 0")
    snap = _pair(args.snap_to) if args.snap_to else None
    grid = build_grid(region, proj, cell_size, snap_lonlat=snap)
    if grid.n_cells > 4_000_000:
        die("grid is %d cells (%.1f m) -- too large; raise --cell-size or shrink the region"
            % (grid.n_cells, cell_size))

    ring_xy = None
    if region["ring"]:
        ring_xy = [proj.fwd(lat, lon) for lon, lat in region["ring"]]

    # ---- observations in metres + sigmas ---------------------------------
    stats = {}
    sigmas, fallback_acc = sigma_for_obs(obs, args, stats)

    # region membership pre-filter (keep a margin so kernels bleed in)
    # keep every fix whose kernel can reach the region: the kernel's sigma is
    # sqrt(accuracy^2 + bandwidth^2) and its tail runs to TAU sigmas
    _bw = float(args.bandwidth)
    _se = sorted(math.sqrt(sg * sg + _bw * _bw) for sg in sigmas)
    margin = TAU * _se[int(0.9 * (len(_se) - 1))]
    sel = []
    bx0, by0 = proj.fwd(s, w)
    bx1p, by1p = proj.fwd(n, e_)
    for k, o in enumerate(obs):
        x, y = proj.fwd(o.lat, o.lon)
        _bx0 = bx0
        if not (_bx0 - margin <= x <= bx1p + margin and by0 - margin <= y <= by1p + margin):
            continue
        sel.append(k)
    if not sel:
        tried = ""
        if len(cands) > 1:
            tried = "\n  readings tried: " + "; ".join(
                "%s -> %d obs inside" % (label, c)
                for (label, _), c in zip(cands, cand_counts))
        die("no observations inside/near the region %s%s"
            % (tuple(round(v, 6) for v in region["bbox"]), tried))
    obs_sel = [obs[k] for k in sel]
    sig_sel = [sigmas[k] for k in sel]

    # ---- evaluate --------------------------------------------------------
    res = evaluate_field(obs_sel, sig_sel, grid, args, want, value_paths)
    _blank_uncovered(res, grid)
    cover_arr = res.get("_cover")

    # ---- masking ---------------------------------------------------------
    mask_inside = None
    if ring_xy is not None and not args.no_mask:
        mask_inside = []
        xs, ys = grid.centers_xy()
        for j in range(grid.ny):
            for i in range(grid.nx):
                mask_inside.append(point_in_polygon(xs[i], ys[j], ring_xy))
        mask_inside_arr = mask_inside
    else:
        mask_inside_arr = None

    # evidence thresholds (`_ess`/`_conf` are always computed internally)
    threshold = max(float(args.min_weight), float(args.require_ess))
    if threshold > 0:
        for name in value_names_of(res):
            for cell in range(grid.n_cells):
                if res["_ess"][cell] is not None and res["_ess"][cell] < threshold:
                    for k in res[name]:
                        res[name][k][cell] = None
    if args.min_conf > 0:
        # the trust-radius cut: drop cells whose evidence wave is below W
        # (W=0.5 keeps roughly the inside of each fix's 1.2-sigma disk)
        for name in value_names_of(res):
            for cell in range(grid.n_cells):
                if res["_conf"][cell] is not None and res["_conf"][cell] < args.min_conf:
                    for k in res[name]:
                        res[name][k][cell] = None

    if "ess" not in want:
        res.pop("_ess", None)
    if "conf" not in want:
        res.pop("_conf", None)

    value_names = list(value_paths.keys())

    # ---- summary ---------------------------------------------------------
    times = [o.t for o in obs_sel if o.t]
    filled = 0
    for cell in range(grid.n_cells):
        if mask_inside_arr is not None and not mask_inside_arr[cell]:
            continue
        v = res[value_names[0]].get("value", [None] * grid.n_cells)[cell]
        if v is not None and math.isfinite(v):
            filled += 1
    cells_out = grid.n_cells if mask_inside_arr is None else sum(mask_inside_arr)
    stats.update({
        "input": os.path.basename(args.input),
        "records_total": len(records),
        "observations_used": len(obs_sel),
        "observations_in_range": len(obs),
        "dropped": {k: v for k, v in dropped.items() if v},
        "records_without_timestamp": n_no_ts if range_active else 0,
        "time_range": {
            "from": since.isoformat().replace("+00:00", "Z") if since else None,
            "to": until.isoformat().replace("+00:00", "Z") if until else None,
            "data_from": min(times).isoformat().replace("+00:00", "Z") if times else None,
            "data_to": max(times).isoformat().replace("+00:00", "Z") if times else None,
        },
        "region": {"kind": region["kind"], "bbox_wsen": [round(v, 7) for v in region["bbox"]]},
        "grid": {"cell_size_m": round(cell_size, 3), "nx": grid.nx, "ny": grid.ny,
                 "cells": grid.n_cells, "cells_in_region": cells_out, "cells_with_value": filled},
        "model": {
            "channels": value_paths,
            "accuracy_semantics": "68%% confidence radius -> sigma (k68=%g)" % args.k68,
            "bandwidth_m": float(args.bandwidth),
            "accuracy_scale": float(args.accuracy_scale),
            "kernel": "gaussian-2d, weights normalised per cell",
            "estimator": "weighted mean" if "value" in want else None,
            "numpy": _np is not None and not args.no_numpy,
        },
    })

    # ---- serialise -------------------------------------------------------
    if args.out_format == "geojson":
        payload = build_geojson(grid, res, args, value_names, mask_inside_arr, stats)
        text = json.dumps(payload, indent=None if args.output != "-" else 2,
                          separators=(",", ":") if args.output != "-" else None)
    elif args.out_format == "json":
        xs, ys = grid.centers_xy()
        rows = []
        for j in range(grid.ny):
            row = []
            for i in range(grid.nx):
                cell = j * grid.nx + i
                if mask_inside_arr is not None and not mask_inside_arr[cell]:
                    row.append(None)
                    continue
                lat, lon = grid.proj.inv(xs[i], ys[j])
                entry = {"lat": round(lat, 7), "lon": round(lon, 7)}
                for name in value_names:
                    r = res[name]
                    entry[name] = rnd(r.get("value", [None] * grid.n_cells)[cell], args.precision)
                    for k, out in (("sd", "sd"), ("ci_low", "ci_low"), ("ci_high", "ci_high"),
                                   ("min", "min"), ("max", "max"), ("median", "median")):
                        if k in r:
                            entry["%s_%s" % (name, out)] = rnd(r[k][cell], args.precision)
                if "_cover" in res and args.fill_gaps:
                    entry["extrapolated"] = not bool(res["_cover"][cell])
                if "_ess" in res:
                    entry["ess"] = rnd(res["_ess"][cell], 3)
                if "_conf" in res:
                    entry["conf"] = rnd(res["_conf"][cell], 4)
                if "_sigma" in res:
                    entry["sigma_m"] = rnd(res["_sigma"][cell], 2)
                row.append(entry)
            rows.append(row)
        payload = {"meta": stats,
                   "grid": {"nx": grid.nx, "ny": grid.ny, "cell_size_m": cell_size,
                            "origin": {"lon": round(proj.inv(grid.x0, grid.y0)[1], 7),
                                       "lat": round(proj.inv(grid.x0, grid.y0)[0], 7)}},
                   "cells": rows}
        text = json.dumps(payload, indent=2)
    else:
        text = ""

    if args.out_format == "csv":
        path = args.output if args.output != "-" else "signal_field.csv"
        write_csv(grid, res, args, value_names, mask_inside_arr, path)
        if not args.quiet:
            sys.stderr.write("wrote %s (%d cells)\n" % (path, cells_out))
    else:
        if args.output == "-":
            sys.stdout.write(text + "\n")
        else:
            with open(args.output, "w") as fh:
                fh.write(text + "\n")
            if not args.quiet:
                sys.stderr.write("wrote %s (%d features)\n" % (args.output, cells_out))
        if args.csv:
            write_csv(grid, res, args, value_names, mask_inside_arr, args.csv)
            if not args.quiet:
                sys.stderr.write("wrote %s\n" % args.csv)

    if args.png:
        primary = value_names[0]
        vals = res[primary].get("value")
        if vals is None:
            die("--png needs the value statistic (it is on by default)")
        plo = phi = None
        if args.png_range:
            try:
                plo, phi = (float(t) for t in args.png_range.split(","))
            except ValueError:
                die("--png-range wants 'LO,HI'")
        info = write_png(args.png, grid, vals, mask_inside_arr,
                        scale=args.png_scale, lo=plo, hi=phi)
        if not args.quiet:
            sys.stderr.write("wrote %s (%dx%d px, %s scaled %.1f..%.1f)\n" % (
                info["path"], info["width"], info["height"], primary,
                info["min"], info["max"]))

    if not args.quiet:
        accs = [o.acc for o in obs_sel if o.acc]
        vals = [o.vals.get(value_names[0]) for o in obs_sel]
        vals = [v for v in vals if v is not None]
        sys.stderr.write(
            "obs used %d/%d | cell %.2f m | grid %dx%d | bandwidth %.1f m | "
            "accuracy med %s m (min %s / max %s) | %s med %s (min %s / max %s)\n" % (
                len(obs_sel), len(records), cell_size, grid.nx, grid.ny, args.bandwidth,
                rnd(statistics.median(accs), 1) if accs else "n/a",
                rnd(min(accs), 1) if accs else "n/a",
                rnd(max(accs), 1) if accs else "n/a",
                value_names[0],
                rnd(statistics.median(vals), 1) if vals else "n/a",
                rnd(min(vals), 1) if vals else "n/a",
                rnd(max(vals), 1) if vals else "n/a",
            ))
        if args.heatmap:
            try:
                width = min(100, max(40, os.get_terminal_size(2).columns - 2))
            except OSError:
                width = 78
            sys.stderr.write("\n" + render_heatmap(grid, res, value_names[0], mask_inside_arr,
                                                  width=width, height=max(10, width // 3)) + "\n")
    return 0


def value_names_of(res):
    return [k for k in res.keys() if not k.startswith("_")]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        sys.exit(130)
