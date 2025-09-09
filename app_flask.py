"""
DOZE — Drone Operation Zone Editor (Flask + Bootstrap 5.3)
Single-file app.py so you can `python app.py` and go.

Requires (pip install):
  flask shapely pyproj simplekml

This version uses Leaflet + Leaflet.Draw on the frontend (no Streamlit/Folium),
mirrors the original calculations, and supports:
- Draw/save a polygon (FG / CV / GRB)
- Compute derived zones (FG, CV, OV, GRB)
- Show metrics & fit-to-bounds
- Download KMZ with proper 3D volumes (FG/CV/OV extruded to heights)

Notes:
- The UI is Bootstrap 5.3. We keep state on the client and POST to /compute and /download_kmz.
- EPSG is chosen per-geometry via UTM from centroid, same idea as the original.
- Colors and logic follow the original script closely.
"""
from __future__ import annotations

import io
import json
import math
import zipfile
from datetime import datetime
from typing import Tuple

from flask import Flask, render_template, request, jsonify, send_file

from shapely.geometry import shape, mapping, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid
from pyproj import Transformer
import simplekml

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-doze-secret"

# ---------------- Constants & helpers ----------------
G = 9.81
MAX_HEIGHT_AGL_M = 120.0
FG_TOP_CAP_M = 120.0  # FG cannot exceed this
CV_TOP_CAP_M = 150.0  # CV apex cannot exceed this (FG + buffer)
GRB_PRISM_HEIGHT_M = 10.0  # height for GRB volume in KMZ, above 3D buildings

WGS84 = "EPSG:4326"

drivers_note = (
    "Create a flight geometry - Based on https://www.lba.de/SharedDocs/Downloads/DE/B/B5_UAS/"
    "Leitfaden_FG_CV_GRB_eng.pdf which are in turn based on Regulation (EU) 2019/947.\n"
    "This application DOES NOT take into account locations where you are and are not allowed to fly."
)

DRONE_PROFILES = {
    "DJI Matrice 30": {
        "v0_ms": 23.0,       # max groundspeed
        "t_react_s": 1.0,    # reaction time
        "theta_deg": 35.0,   # max pitch
        "s_gps_m": 3.0,      # GPS inaccuracy
        "s_pos_m": 0.3,      # position hold error
        "s_map_m": 1.0,      # map error
        "cd_m": 0.60,        # characteristic dimension (tip-to-tip)
        "h_baro_mode": "Barometric (1 m)",
    },
    "DJI Mavic 3": {
        "v0_ms": 21.0,
        "t_react_s": 1.0,
        "theta_deg": 35.0,
        "s_gps_m": 3.0,
        "s_pos_m": 0.3,
        "s_map_m": 1.0,
        "cd_m": 0.38,
        "h_baro_mode": "Barometric (1 m)",
    },
}

# ---------- Core maths ----------

def cv_lateral_multirotor(v0_ms, t_react_s, theta_deg, s_gps_m, s_pos_m, s_map_m) -> float:
    theta_rad = math.radians(max(1e-6, theta_deg))
    s_rz = v0_ms * max(0.0, t_react_s)
    s_cm = 0.5 * ((v0_ms ** 2) / (G * math.tan(theta_rad)))
    return s_gps_m + s_pos_m + s_map_m + s_rz + s_cm


def cv_vertical_multirotor(h_fg_m, v0_ms, t_react_s, h_baro_m) -> float:
    h_rz = v0_ms * 0.7 * max(0.0, t_react_s)
    h_cm = 0.5 * ((v0_ms ** 2) / (2.0 * G))
    return h_fg_m + h_baro_m + h_rz + h_cm


# GRB models

def grb_simplified(h_cv_m: float, cd_m: float) -> float:
    """1:1 rule: S_GRB = H_CV + 0.5 * CD"""
    return max(0.0, h_cv_m) + 0.5 * max(0.0, cd_m)


def grb_ballistic(v0_ms: float, h_cv_m: float, cd_m: float) -> float:
    """Ballistic: S_GRB = V0 * sqrt(2*H_CV/g) + 0.5 * CD"""
    term = v0_ms * math.sqrt(max(0.0, 2.0 * h_cv_m / G))
    return term + 0.5 * max(0.0, cd_m)


# ---------- Projection helpers ----------

def _utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"


def _get_local_transformers(geom_wgs) -> tuple[str, Transformer, Transformer]:
    c = geom_wgs.centroid
    lon, lat = float(c.x), float(c.y)
    epsg = _utm_epsg_from_lonlat(lon, lat)
    to_local = Transformer.from_crs(WGS84, epsg, always_xy=True)
    to_wgs = Transformer.from_crs(epsg, WGS84, always_xy=True)
    return epsg, to_local, to_wgs


def geojson_to_shapely_wgs(feature_geojson) -> MultiPolygon | Polygon:
    return make_valid(shape(feature_geojson["geometry"]))


def shapely_wgs_to_geojson(geom) -> dict:
    return {"type": "Feature", "properties": {}, "geometry": mapping(geom)}


def _project_poly(poly: Polygon, to_local: Transformer) -> Polygon:
    ext = [to_local.transform(x, y) for x, y in poly.exterior.coords]
    holes = [[to_local.transform(x, y) for x, y in r.coords] for r in poly.interiors]
    return Polygon(ext, holes)


def _unproject_poly(poly: Polygon, to_wgs: Transformer) -> Polygon:
    ext = [to_wgs.transform(x, y) for x, y in poly.exterior.coords]
    holes = [[to_wgs.transform(x, y) for x, y in r.coords] for r in poly.interiors]
    return Polygon(ext, holes)


def buffer_m(geom_wgs, radius_m: float, cap_style=1, join_style=1):
    """Buffer (outward if >0, inward if <0) in metres using a local UTM, then return WGS84 geometry."""
    # OLD: if radius_m <= 0: return geom_wgs
    if abs(radius_m) < 1e-9:
        return geom_wgs

    epsg, to_local, to_wgs = _get_local_transformers(geom_wgs)

    def project_poly(poly: Polygon) -> Polygon:
        ext = [to_local.transform(x, y) for x, y in poly.exterior.coords]
        holes = [[to_local.transform(x, y) for x, y in r.coords] for r in poly.interiors]
        return Polygon(ext, holes)

    def unproject_poly(poly: Polygon) -> Polygon:
        ext = [to_wgs.transform(x, y) for x, y in poly.exterior.coords]
        holes = [[to_wgs.transform(x, y) for x, y in r.coords] for r in poly.interiors]
        return Polygon(ext, holes)

    if isinstance(geom_wgs, Polygon):
        local = project_poly(geom_wgs)
    elif isinstance(geom_wgs, MultiPolygon):
        local = MultiPolygon([project_poly(p) for p in geom_wgs.geoms])
    else:
        raise ValueError("Geometry must be Polygon or MultiPolygon")

    buf = local.buffer(radius_m, cap_style=cap_style, join_style=join_style)
    if buf.is_empty:
        return None

    if isinstance(buf, GeometryCollection):
        parts = [g for g in buf.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not parts:
            return None
        buf = unary_union(parts)

    if isinstance(buf, Polygon):
        return unproject_poly(buf)
    else:
        return MultiPolygon([unproject_poly(p) for p in buf.geoms])



def offset_m(geom_wgs, offset_meters: float, cap_style=1, join_style=1):
    return buffer_m(geom_wgs, offset_meters, cap_style=cap_style, join_style=join_style)


def area_perimeter_m(geom_wgs) -> Tuple[float, float, str]:
    epsg, to_local, _ = _get_local_transformers(geom_wgs)

    if isinstance(geom_wgs, Polygon):
        local = _project_poly(geom_wgs, to_local)
    elif isinstance(geom_wgs, MultiPolygon):
        local = MultiPolygon([_project_poly(p, to_local) for p in geom_wgs.geoms])
    else:
        raise ValueError("Geometry must be Polygon or MultiPolygon")

    return float(local.area), float(local.length), epsg


# ---------- KMZ helpers ----------

def _kml_color(hex_rgb: str, opacity: float = 0.6) -> str:
    """Convert '#RRGGBB' + opacity [0..1] to KML aabbggrr (little-endian ARGB)."""
    hex_rgb = hex_rgb.lstrip("#")
    r = int(hex_rgb[0:2], 16)
    g = int(hex_rgb[2:4], 16)
    b = int(hex_rgb[4:6], 16)
    a = int(max(0, min(255, round(opacity * 255))))
    return f"{a:02x}{b:02x}{g:02x}{r:02x}"


def _add_geojson_polygon_to_kml_folder(
    fol: simplekml.Folder,
    feature: dict,
    name: str,
    line_hex: str,
    fill_opacity: float,
    height_m: float,
    extrude: bool,
):
    geom = feature["geometry"]
    gtype = geom["type"].lower()
    ringsets = []
    if gtype == "polygon":
        ringsets = [geom["coordinates"]]
    elif gtype == "multipolygon":
        ringsets = geom["coordinates"]
    else:
        return

    for idx, rings in enumerate(ringsets, start=1):
        pol = fol.newpolygon(name=f"{name} {idx}" if len(ringsets) > 1 else name)

        outer = rings[0]
        holes = rings[1:] if len(rings) > 1 else []

        if extrude:
            pol.outerboundaryis = [(lon, lat, height_m) for lon, lat in outer]
            if holes:
                pol.innerboundaryis = [[(lon, lat, height_m) for lon, lat in hole] for hole in holes]
            pol.altitudemode = simplekml.AltitudeMode.relativetoground
            pol.extrude = 1
        else:
            pol.outerboundaryis = outer
            if holes:
                pol.innerboundaryis = holes
            pol.altitudemode = simplekml.AltitudeMode.clamptoground
            pol.extrude = 0

        pol.style.linestyle.color = _kml_color(line_hex, 1.0)
        pol.style.linestyle.width = 2
        pol.style.polystyle.color = _kml_color(line_hex, fill_opacity)


def write_kmz(zones: dict, params: dict) -> bytes:
    kml = simplekml.Kml()
    kml.document.name = f"DOZE Export {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    kml.document.description = (
        "DOZE — Drone Operation Zone Editor\n"
        "Based on LBA FG/CV/GRB guidance and EU 2019/947.\n\n"
        f"Parameters:\n{json.dumps(params, indent=2)}"
    )

    palette = {
        "FG": ("#15c048", 0.8),
        "CV": ("#ff9800", 0.5),
        "OV": ("#42a5f5", 0.05),
        "GRB": ("#e53935", 0.9),
    }

    h_fg = params["h_fg_m"]
    h_cv = params["h_cv_m"]

    zone_properties = {
        "FG": {"height": h_fg, "extrude": True},
        "CV": {"height": h_cv, "extrude": True},
        "OV": {"height": h_cv, "extrude": True},  # OV up to CV
        "GRB": {"height": GRB_PRISM_HEIGHT_M, "extrude": True},
    }

    for layer in ["FG", "CV", "OV", "GRB"]:
        if layer not in zones or not zones[layer]:
            continue
        props = zone_properties[layer]
        fol = kml.newfolder(name=layer)
        _add_geojson_polygon_to_kml_folder(
            fol=fol,
            feature=zones[layer],
            name=layer,
            line_hex=palette[layer][0],
            fill_opacity=palette[layer][1],
            height_m=props["height"],
            extrude=props["extrude"],
        )

    kml_content_bytes = kml.kml().encode("utf-8")
    in_memory = io.BytesIO()
    with zipfile.ZipFile(in_memory, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_content_bytes)
    in_memory.seek(0)
    return in_memory.read()


# ---------- Flask routes ----------

@app.route("/")
def index():
    return render_template(
        "template.jinja.html",
        drone_profiles=DRONE_PROFILES,
        default_profile_name="DJI Matrice 30",
        drivers_note=drivers_note,
    )


@app.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.get_json(force=True)
        base_layer = data["input_layer"]  # FG | CV | GRB
        feature = data["input_geojson"]
        planned_fg_input_m = float(data.get("planned_fg_input_m", 50.0))

        v0_ms = float(data.get("v0_ms", 23.0))
        t_react_s = float(data.get("t_react_s", 1.0))
        theta_deg = float(data.get("theta_deg", 35.0))
        s_gps_m = float(data.get("s_gps_m", 3.0))
        s_pos_m = float(data.get("s_pos_m", 0.3))
        s_map_m = float(data.get("s_map_m", 1.0))
        cd_m = float(data.get("cd_m", 0.6))
        h_baro_mode = str(data.get("h_baro_mode", "Barometric (1 m)"))
        grb_method = str(data.get("grb_method", "Ballistic"))

        h_baro_m = 1.0 if "Barometric" in h_baro_mode else 4.0

        cv_m = cv_lateral_multirotor(v0_ms, t_react_s, theta_deg, s_gps_m, s_pos_m, s_map_m)

        # Vertical buffer components (display only)
        h_rz = v0_ms * 0.7 * max(0.0, t_react_s)
        h_cm = 0.5 * ((v0_ms ** 2) / (2.0 * G))
        vertical_buffer_m = h_baro_m + h_rz + h_cm

        allowed_fg_by_cv = max(0.0, CV_TOP_CAP_M - vertical_buffer_m)
        fg_user_capped = min(FG_TOP_CAP_M, planned_fg_input_m)
        calculated_h_fg_m = max(0.0, min(fg_user_capped, allowed_fg_by_cv))
        h_cv_apex_m = calculated_h_fg_m + vertical_buffer_m

        if grb_method.lower().startswith("simplified"):
            grb_margin = grb_simplified(h_cv_apex_m, cd_m)
        else:
            grb_margin = grb_ballistic(v0_ms, h_cv_apex_m, cd_m)

        base_geom = geojson_to_shapely_wgs(feature)

        fg_geom = cv_geom = ov_geom = grb_geom = None

        if base_layer == "FG":
            fg_geom = base_geom
            cv_geom = buffer_m(fg_geom, cv_m, cap_style=1, join_style=1)
            ov_geom = cv_geom  # equals CV in this model
            grb_geom = buffer_m(ov_geom, grb_margin, cap_style=1, join_style=1)
        
        elif base_layer == "CV":
            cv_geom = base_geom
            fg_geom = offset_m(cv_geom, -cv_m, cap_style=1, join_style=1)
            ov_geom = cv_geom
            grb_geom = buffer_m(ov_geom, grb_margin, cap_style=1, join_style=1)
        
        elif base_layer == "GRB":
            grb_geom = base_geom
            ov_geom = offset_m(grb_geom, -grb_margin, cap_style=1, join_style=1)
            if ov_geom is not None:
                cv_geom = ov_geom
                fg_geom = offset_m(cv_geom, -cv_m, cap_style=1, join_style=1)
        else:
            return jsonify({"ok": False, "error": "Invalid base layer."}), 400

        zones_data = {}
        bounds = None
        metrics = {}
        crs_used = None

        def add_zone(name, geom):
            nonlocal bounds, crs_used
            if not geom:
                return
            zones_data[name] = shapely_wgs_to_geojson(geom)
            minx, miny, maxx, maxy = geom.bounds
            if bounds is None:
                bounds = [minx, miny, maxx, maxy]
            else:
                bounds = [
                    min(bounds[0], minx),
                    min(bounds[1], miny),
                    max(bounds[2], maxx),
                    max(bounds[3], maxy),
                ]
            area, peri, epsg = area_perimeter_m(geom)
            crs_used = epsg
            metrics[name] = {
                "area_m2": round(area, 2),
                "perimeter_m": round(peri, 2),
            }

        for nm, gm in (("FG", fg_geom), ("CV", cv_geom), ("OV", ov_geom), ("GRB", grb_geom)):
            add_zone(nm, gm)

        if not zones_data:
            return jsonify({"ok": False, "error": "No valid zones could be computed. Draw a larger polygon or adjust margins."}), 400

        export_params = {
            "h_fg_m": round(calculated_h_fg_m, 2),
            "h_cv_m": round(h_cv_apex_m, 2),
            "v0_ms": v0_ms,
            "t_react_s": t_react_s,
            "theta_deg": theta_deg,
            "s_gps_m": s_gps_m,
            "s_pos_m": s_pos_m,
            "s_map_m": s_map_m,
            "cv_m": round(cv_m, 2),
            "cd_m": cd_m,
            "grb_method": grb_method,
            "grb_margin_m": round(grb_margin, 2),
            "crs_buffering": crs_used or "local UTM",
            "altitude_mode": "relativeToGround",
            "base_layer": base_layer,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        return jsonify({
            "ok": True,
            "zones": zones_data,
            "bounds": bounds,
            "metrics": metrics,
            "export_params": export_params,
            "values": {
                "cv_m": round(cv_m, 2),
                "planned_fg_input_m": planned_fg_input_m,
                "calculated_h_fg_m": round(calculated_h_fg_m, 2),
                "vertical_buffer_m": round(vertical_buffer_m, 2),
                "h_cv_apex_m": round(h_cv_apex_m, 2),
                "grb_margin": round(grb_margin, 2),
            },
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/download_kmz", methods=["POST"])
def download_kmz():
    try:
        payload = request.get_json(force=True)
        zones = payload["zones"]
        export_params = payload["export_params"]
        kmz_bytes = write_kmz(zones, export_params)
        fname = f"DOZE_zones_{datetime.now().strftime('%Y%m%d_%H%M')}.kmz"
        return send_file(io.BytesIO(kmz_bytes), mimetype="application/vnd.google-earth.kmz", as_attachment=True, download_name=fname)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------- HTML template (Bootstrap 5.3 + Leaflet + Leaflet.Draw) ----------
TEMPLATE = "template.jinja.html"
# r"""
# <!doctype html>
# <html lang="en">
#   <head>
#     <meta charset="utf-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1">
#     <title>DOZE — Drone Operation Zone Editor</title>

#     <!-- Bootstrap 5.3 -->
#     <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">

#     <!-- Leaflet & Leaflet.Draw -->
#     <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
#     <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

#     <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet-draw@1.0.4/dist/leaflet.draw.css"/>
#     <script src="https://cdn.jsdelivr.net/npm/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>

#     <style>
#       body { background: #f8f9fa; }
#       #map { height: 650px; }
#       .sidebar { max-width: 380px; }
#       .metric { font-variant-numeric: tabular-nums; }
#       code.small { font-size: .825rem; }
#     </style>
#   </head>
#   <body>
#     <nav class="navbar navbar-expand-lg bg-body border-bottom sticky-top">
#       <div class="container-fluid">
#         <span class="navbar-brand fw-semibold">DOZE — Drone Operation Zone Editor</span>
#         <span class="navbar-text">(Flask + Bootstrap 5.3)</span>
#       </div>
#     </nav>

#     <div class="container-fluid py-3">
#       <div class="row g-3">
#         <div class="col-12 col-lg-4 col-xxl-3">
#           <div class="card shadow-sm sidebar">
#             <div class="card-body">
#               <h5 class="card-title">Drawing mode</h5>
#               <div class="btn-group" role="group">
#                 <input type="radio" class="btn-check" name="inputLayer" id="layerFG" autocomplete="off" value="FG" checked>
#                 <label class="btn btn-outline-primary" for="layerFG">FG</label>
#                 <input type="radio" class="btn-check" name="inputLayer" id="layerCV" autocomplete="off" value="CV">
#                 <label class="btn btn-outline-primary" for="layerCV">CV</label>
#                 <input type="radio" class="btn-check" name="inputLayer" id="layerGRB" autocomplete="off" value="GRB">
#                 <label class="btn btn-outline-primary" for="layerGRB">GRB</label>
#               </div>

#               <hr>
#               <h5>Flight Height</h5>
#               <div class="mb-3">
#                 <label class="form-label">Planned maximum flight height (m)</label>
#                 <input type="number" step="1" min="0" max="120" class="form-control" id="plannedFG" value="50">
#                 <div class="form-text">Up to 120 m. Will be lowered if CV apex would exceed 150 m.</div>
#               </div>

#               <hr>
#               <h5>Contingency Volume Calculator <small class="text-muted">(Multirotor)</small></h5>
#               <div class="mb-2">
#                 <label class="form-label">Drone type</label>
#                 <select class="form-select" id="droneProfile">
#                   {% for name, vals in drone_profiles.items() %}
#                     <option value="{{name}}" {% if name==default_profile_name %}selected{% endif %}>{{name}}</option>
#                   {% endfor %}
#                 </select>
#               </div>

#               <div class="row g-2">
#                 <div class="col-6">
#                   <label class="form-label">V₀ (m/s)</label>
#                   <input type="number" step="0.5" min="0" max="40" class="form-control" id="v0" value="{{drone_profiles[default_profile_name]['v0_ms']}}">
#                 </div>
#                 <div class="col-6">
#                   <label class="form-label">Reaction t (s)</label>
#                   <input type="number" step="0.1" min="0" max="5" class="form-control" id="tReact" value="{{drone_profiles[default_profile_name]['t_react_s']}}">
#                 </div>
#                 <div class="col-6">
#                   <label class="form-label">Pitch θ (deg)</label>
#                   <input type="number" step="1" min="1" max="45" class="form-control" id="theta" value="{{drone_profiles[default_profile_name]['theta_deg']}}">
#                 </div>
#                 <div class="col-6">
#                   <label class="form-label">GPS inacc. (m)</label>
#                   <input type="number" step="0.5" min="0" max="10" class="form-control" id="sGps" value="{{drone_profiles[default_profile_name]['s_gps_m']}}">
#                 </div>
#                 <div class="col-6">
#                   <label class="form-label">Pos hold (m)</label>
#                   <input type="number" step="0.1" min="0" max="10" class="form-control" id="sPos" value="{{drone_profiles[default_profile_name]['s_pos_m']}}">
#                 </div>
#                 <div class="col-6">
#                   <label class="form-label">Map error (m)</label>
#                   <input type="number" step="0.5" min="0" max="10" class="form-control" id="sMap" value="{{drone_profiles[default_profile_name]['s_map_m']}}">
#                 </div>
#                 <div class="col-6">
#                   <label class="form-label">CD (m)</label>
#                   <input type="number" step="0.1" min="0" max="10" class="form-control" id="cd" value="{{drone_profiles[default_profile_name]['cd_m']}}">
#                 </div>
#                 <div class="col-6">
#                   <label class="form-label">Altitude measurement</label>
#                   <select class="form-select" id="hBaroMode">
#                     <option {% if 'Barometric' in drone_profiles[default_profile_name]['h_baro_mode'] %}selected{% endif %}>Barometric (1 m)</option>
#                     <option {% if 'GPS' in drone_profiles[default_profile_name]['h_baro_mode'] %}selected{% endif %}>GPS-based (4 m)</option>
#                   </select>
#                 </div>
#               </div>

#               <div class="mt-2">
#                 <label class="form-label">GRB Method</label>
#                 <select class="form-select" id="grbMethod">
#                   <option>Ballistic</option>
#                   <option>Simplified (1:1)</option>
#                 </select>
#               </div>

#               <hr>
#               <h5>Actions</h5>
#               <div class="d-grid gap-2">
#                 <button class="btn btn-success" id="btnSave">Save <span id="saveLabel">FG</span></button>
#                 <button class="btn btn-primary" id="btnCompute">Compute Zones</button>
#                 <button class="btn btn-outline-secondary" id="btnDownload" disabled>Download KMZ</button>
#                 <button class="btn btn-outline-danger" id="btnClear">Clear</button>
#               </div>

#               <hr>
#               <small class="text-muted">{{drivers_note}}</small>
#             </div>
#           </div>
#         </div>

#         <div class="col-12 col-lg-8 col-xxl-9">
#           <div class="card shadow-sm">
#             <div class="card-body">
#               <div id="map" class="rounded"></div>
#               <div class="mt-3" id="metrics"></div>
#               <div class="mt-2" id="inputPreview" style="display:none"></div>
#               <div class="mt-3 text-muted small">© DOZE — Drone Operation Zone Editor</div>
#             </div>
#           </div>
#         </div>
#       </div>
#     </div>

#     <script>
#       const DRONE_PROFILES = {{ drone_profiles | tojson }};

#       const colors = {
#         FG:  { color: '#1565c0', fillOpacity: 0.15 },
#         CV:  { color: '#ff9800', fillOpacity: 0.15 },
#         OV:  { color: '#42a5f5', fillOpacity: 0.05 },
#         GRB: { color: '#e53935', fillOpacity: 0.12 },
#       };

#       const appState = {
#         inputLayer: 'FG',
#         inputGeoJSON: null,
#         zones: null,
#         exportParams: null,
#         layers: {},
#       };

#       // --- Map setup ---
#       const defaultCenter = [52.1, 5.3];
#       const map = L.map('map', { zoomControl: true }).setView(defaultCenter, 8);

#       const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
#         maxZoom: 19,
#         attribution: '&copy; OpenStreetMap contributors'
#       }).addTo(map);

#       const positron = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
#         attribution: '&copy; CARTO, OpenStreetMap'
#       });

#       const darkMatter = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
#         attribution: '&copy; CARTO, OpenStreetMap'
#       });

#       const baseMaps = { 'OpenStreetMap': osm, 'CartoDB positron': positron, 'CartoDB dark_matter': darkMatter };
#       L.control.layers(baseMaps, null, { collapsed: true }).addTo(map);

#       const drawnItems = new L.FeatureGroup();
#       map.addLayer(drawnItems);

#       const drawControl = new L.Control.Draw({
#         position: 'topleft',
#         draw: {
#           polyline: false, rectangle: false, circle: false, marker: false, circlemarker: false,
#           polygon: { allowIntersection: false, showArea: true, shapeOptions: { color: '#2e7d32', weight: 3, fillOpacity: 0.2 } }
#         },
#         edit: { featureGroup: drawnItems }
#       });
#       map.addControl(drawControl);

#       map.on(L.Draw.Event.CREATED, function (e) {
#         drawnItems.addLayer(e.layer);
#       });
#       map.on(L.Draw.Event.EDITED, function (e) {});

#       function latestPolygonFeature() {
#         let last = null;
#         drawnItems.eachLayer(function(layer){
#           if (layer.toGeoJSON && layer.toGeoJSON().geometry && /Polygon$/i.test(layer.toGeoJSON().geometry.type)) {
#             last = layer;
#           }
#         });
#         return last ? last.toGeoJSON() : null;
#       }

#       function setSaveLabel() {
#         const v = document.querySelector('input[name="inputLayer"]:checked').value;
#         document.getElementById('saveLabel').textContent = v;
#       }

#       document.querySelectorAll('input[name="inputLayer"]').forEach(r => {
#         r.addEventListener('change', () => {
#           appState.inputLayer = document.querySelector('input[name="inputLayer"]:checked').value;
#           setSaveLabel();
#         });
#       });
#       setSaveLabel();

#       // Drone profile autofill
#       document.getElementById('droneProfile').addEventListener('change', (e) => {
#         const p = DRONE_PROFILES[e.target.value];
#         if (!p) return;
#         document.getElementById('v0').value = p.v0_ms;
#         document.getElementById('tReact').value = p.t_react_s;
#         document.getElementById('theta').value = p.theta_deg;
#         document.getElementById('sGps').value = p.s_gps_m;
#         document.getElementById('sPos').value = p.s_pos_m;
#         document.getElementById('sMap').value = p.s_map_m;
#         document.getElementById('cd').value = p.cd_m;
#         document.getElementById('hBaroMode').value = p.h_baro_mode;
#       });

#       // Buttons
#       document.getElementById('btnSave').addEventListener('click', () => {
#         const feat = latestPolygonFeature();
#         if (!feat) { alert('Please draw a polygon first.'); return; }
#         appState.inputGeoJSON = { type: 'Feature', properties: { name: appState.inputLayer, source: 'leaflet-draw' }, geometry: feat.geometry };
#         document.getElementById('inputPreview').style.display = 'block';
#         document.getElementById('inputPreview').innerHTML = `<h6>Current input polygon</h6><div>Layer: <strong>${appState.inputLayer}</strong></div><pre class="small bg-light p-2 border rounded"><code>${JSON.stringify({ type: appState.inputGeoJSON.geometry.type, coordinates: appState.inputGeoJSON.geometry.coordinates.slice(0,1) }, null, 2)}</code></pre>`;
#         alert(appState.inputLayer + ' saved to session.');
#       });

#       function getFormValues() {
#         return {
#           input_layer: appState.inputLayer,
#           input_geojson: appState.inputGeoJSON,
#           planned_fg_input_m: parseFloat(document.getElementById('plannedFG').value || '50'),
#           v0_ms: parseFloat(document.getElementById('v0').value || '23'),
#           t_react_s: parseFloat(document.getElementById('tReact').value || '1'),
#           theta_deg: parseFloat(document.getElementById('theta').value || '35'),
#           s_gps_m: parseFloat(document.getElementById('sGps').value || '3'),
#           s_pos_m: parseFloat(document.getElementById('sPos').value || '0.3'),
#           s_map_m: parseFloat(document.getElementById('sMap').value || '1'),
#           cd_m: parseFloat(document.getElementById('cd').value || '0.6'),
#           h_baro_mode: document.getElementById('hBaroMode').value,
#           grb_method: document.getElementById('grbMethod').value,
#         };
#       }

#       function clearDerivedLayers() {
#         ['FG','CV','OV','GRB'].forEach(n => {
#           if (appState.layers[n]) { map.removeLayer(appState.layers[n]); appState.layers[n] = null; }
#         });
#       }

#       function addGeoLayer(name, feat) {
#         if (!feat) return;
#         if (appState.layers[name]) map.removeLayer(appState.layers[name]);
#         appState.layers[name] = L.geoJSON(feat, {
#           style: () => ({ color: colors[name].color, weight: 3, fillOpacity: colors[name].fillOpacity })
#         }).addTo(map);
#       }

#       document.getElementById('btnCompute').addEventListener('click', async () => {
#         if (!appState.inputGeoJSON) { alert('Please draw and Save your polygon first.'); return; }
#         clearDerivedLayers();
#         const payload = getFormValues();
#         try {
#           const res = await fetch('/compute', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
#           const data = await res.json();
#           if (!data.ok) { alert('Failed: ' + (data.error || 'unknown')); return; }

#           appState.zones = data.zones;
#           appState.exportParams = data.export_params;

#           // Add layers, but skip duplicating base layer if desired — here we add all returned
#           addGeoLayer('FG', data.zones.FG);
#           addGeoLayer('CV', data.zones.CV);
#           addGeoLayer('OV', data.zones.OV);
#           addGeoLayer('GRB', data.zones.GRB);

#           if (data.bounds) {
#             const b = data.bounds; // [minx,miny,maxx,maxy]
#             const sw = [b[1], b[0]], ne = [b[3], b[2]];
#             map.fitBounds([sw, ne], { padding: [20,20] });
#           }

#           document.getElementById('btnDownload').disabled = false;
#           renderMetrics(data.values, data.metrics);
#         } catch (e) {
#           alert('Error: ' + e);
#         }
#       });

#       document.getElementById('btnDownload').addEventListener('click', async () => {
#         if (!appState.zones || !appState.exportParams) { alert('Nothing to export yet.'); return; }
#         try {
#           const res = await fetch('/download_kmz', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ zones: appState.zones, export_params: appState.exportParams }) });
#           if (!res.ok) { const t = await res.text(); alert('Export failed: ' + t); return; }
#           const blob = await res.blob();
#           const url = window.URL.createObjectURL(blob);
#           const a = document.createElement('a');
#           a.href = url;
#           const fg = appState.exportParams.h_fg_m.toFixed ? appState.exportParams.h_fg_m.toFixed(0) : appState.exportParams.h_fg_m;
#           const cv = appState.exportParams.h_cv_m.toFixed ? appState.exportParams.h_cv_m.toFixed(0) : appState.exportParams.h_cv_m;
#           a.download = `DOZE_zones_${new Date().toISOString().slice(0,16).replace(/[-:T]/g,'')}.kmz`;
#           document.body.appendChild(a); a.click(); a.remove(); window.URL.revokeObjectURL(url);
#         } catch (e) { alert('Error: ' + e); }
#       });

#       document.getElementById('btnClear').addEventListener('click', () => {
#         drawnItems.clearLayers();
#         clearDerivedLayers();
#         appState.inputGeoJSON = null;
#         appState.zones = null;
#         appState.exportParams = null;
#         document.getElementById('metrics').innerHTML = '';
#         document.getElementById('inputPreview').style.display = 'none';
#         document.getElementById('btnDownload').disabled = true;
#       });

#       function renderMetrics(values, metrics) {
#         const lines = [];
#         if (metrics.FG) lines.push(`<li><strong>FG</strong>: area ${metrics.FG.area_m2.toLocaleString()} m² • perimeter ${metrics.FG.perimeter_m.toLocaleString()} m</li>`);
#         if (metrics.CV) lines.push(`<li><strong>CV</strong>: area ${metrics.CV.area_m2.toLocaleString()} m² • perimeter ${metrics.CV.perimeter_m.toLocaleString()} m</li>`);
#         if (metrics.OV) lines.push(`<li><strong>OV</strong>: area ${metrics.OV.area_m2.toLocaleString()} m² • perimeter ${metrics.OV.perimeter_m.toLocaleString()} m</li>`);
#         if (metrics.GRB) lines.push(`<li><strong>GRB</strong>: area ${metrics.GRB.area_m2.toLocaleString()} m² • perimeter ${metrics.GRB.perimeter_m.toLocaleString()} m</li>`);

#         const warn = (parseFloat(document.getElementById('plannedFG').value) > values.calculated_h_fg_m + 1e-6)
#           ? `<div class="alert alert-warning mt-2">Planned FG reduced from ${parseFloat(document.getElementById('plannedFG').value).toFixed(1)} m to ${values.calculated_h_fg_m.toFixed(1)} m to keep CV apex ≤ 150 m.</div>`
#           : '';

#         const imposs = (values.calculated_h_fg_m <= 0)
#           ? `<div class="alert alert-danger mt-2">Flight not possible: buffer alone exceeds the CV ceiling of 150 m. Reduce speed or reaction time.</div>`
#           : '';

#         document.getElementById('metrics').innerHTML = `
#           <div class="row g-2">
#             <div class="col-12 col-xl-6">
#               <div class="card">
#                 <div class="card-body">
#                   <h6 class="card-title">Calculated values</h6>
#                   <div class="row row-cols-2 g-2">
#                     <div class="col"><div class="text-muted">Lateral CV margin</div><div class="metric h5">${values.cv_m.toFixed(1)} m</div></div>
#                     <div class="col"><div class="text-muted">Planned FG (input)</div><div class="metric h5">${values.planned_fg_input_m.toFixed(1)} m</div></div>
#                     <div class="col"><div class="text-muted">Resulting FG apex H_FG</div><div class="metric h5">${values.calculated_h_fg_m.toFixed(1)} m</div></div>
#                     <div class="col"><div class="text-muted">Vertical Buffer</div><div class="metric h5">${values.vertical_buffer_m.toFixed(1)} m</div></div>
#                     <div class="col"><div class="text-muted">Resulting CV apex H_CV</div><div class="metric h5">${values.h_cv_apex_m.toFixed(1)} m</div></div>
#                     <div class="col"><div class="text-muted">GRB margin</div><div class="metric h5">${values.grb_margin.toFixed(1)} m</div></div>
#                   </div>
#                   ${warn}
#                   ${imposs}
#                 </div>
#               </div>
#             </div>
#             <div class="col-12 col-xl-6">
#               <div class="card h-100">
#                 <div class="card-body">
#                   <h6 class="card-title">Areas & perimeters</h6>
#                   <ul class="mb-0">
#                     ${lines.join('')}
#                   </ul>
#                 </div>
#               </div>
#             </div>
#           </div>`;
#       }
#     </script>

#     <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
#   </body>
# </html>
# """


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
