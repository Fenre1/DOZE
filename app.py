# DOZE — Draw FG + CV/OV/GRB overlays
# Adds buffering with Shapely in EPSG:28992 and renders layers on the map.

import json
import math
from typing import Optional, Tuple

import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from shapely.geometry import shape, mapping, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid
from pyproj import Transformer

import io
import zipfile 

from datetime import datetime
import simplekml

st.set_page_config(page_title="DOZE — Zones Preview", layout="wide")

st.title("DOZE — Drone Operation Zone Editor")
st.caption("Create a flight geometry - Based on https://www.lba.de/SharedDocs/Downloads/DE/B/B5_UAS/Leitfaden_FG_CV_GRB_eng.pdf which are in turn based on Regulation (EU) 2019/947.")
st.caption("This application DOES NOT take into account locations where you are and are not allowed to fly.")


# Constants & helpers
G = 9.81
MAX_HEIGHT_AGL_M = 120.0
FG_TOP_CAP_M = 120.0  # FG cannot exceed this
CV_TOP_CAP_M = 150.0  # CV apex cannot exceed this (FG + buffer)
GRB_PRISM_HEIGHT_M = 10.0 # height for GBR, so that it goes above Google Earth's 3d buildings.

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
    # Example 2nd drone (adjust to your second platform’s real values)
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

# Initialize session keys once
def _ensure_param(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

def _apply_profile(profile_name: str):
    prof = DRONE_PROFILES[profile_name]
    for k, v in prof.items():
        st.session_state[k] = v



def cv_lateral_multirotor(v0_ms, t_react_s, theta_deg, s_gps_m, s_pos_m, s_map_m) -> float:
    theta_rad = math.radians(max(1e-6, theta_deg))
    s_rz = v0_ms * max(0.0, t_react_s)
    s_cm = 0.5*((v0_ms**2) / (G * math.tan(theta_rad)))
    return s_gps_m + s_pos_m + s_map_m + s_rz + s_cm

def cv_vertical_multirotor(h_fg_m, v0_ms, t_react_s, h_baro_m) -> float:
    h_rz = v0_ms * 0.7 * max(0.0, t_react_s)
    h_cm = 0.5*((v0_ms**2) / (2.0 * G))
    return h_fg_m + h_baro_m + h_rz + h_cm

# ---- GRB models (multirotor) per your table ----
def grb_simplified(h_cv_m: float, cd_m: float) -> float:
    """1:1 rule: S_GRB = H_CV + 0.5 * CD"""
    return max(0.0, h_cv_m) + 0.5 * max(0.0, cd_m)

def grb_ballistic(v0_ms: float, h_cv_m: float, cd_m: float) -> float:
    """Ballistic: S_GRB = V0 * sqrt(2*H_CV/g) + 0.5 * CD"""
    term = v0_ms * math.sqrt(max(0.0, 2.0 * h_cv_m / G))
    return term + 0.5 * max(0.0, cd_m)

# ---- Projection helpers (WGS84 <-> RD New) ----
WGS84 = "EPSG:4326"
RDNEW = "EPSG:28992"
to_rd = Transformer.from_crs(WGS84, RDNEW, always_xy=True)
to_wgs = Transformer.from_crs(RDNEW, WGS84, always_xy=True)

def _utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    """Return an EPSG code for the UTM zone covering (lon, lat)."""
    zone = int((lon + 180) // 6) + 1  # 1..60
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"

def _get_local_transformers(geom_wgs) -> tuple[str, Transformer, Transformer]:
    """Pick a suitable UTM CRS from the geometry centroid and return (epsg, to_local, to_wgs)."""
    c = geom_wgs.centroid
    lon, lat = float(c.x), float(c.y)
    epsg = _utm_epsg_from_lonlat(lon, lat)
    to_local = Transformer.from_crs(WGS84, epsg, always_xy=True)
    to_wgs   = Transformer.from_crs(epsg, WGS84, always_xy=True)
    return epsg, to_local, to_wgs

def wgs_to_rd_coords(coords):
    return [to_rd.transform(x, y) for x, y in coords]

def rd_to_wgs_coords(coords):
    return [to_wgs.transform(x, y) for x, y in coords]

def geojson_to_shapely_wgs(feature_geojson) -> MultiPolygon | Polygon:
    """Feature -> shapely geometry in WGS84."""
    geom = make_valid(shape(feature_geojson["geometry"]))
    # Ensure polygon orientation/validity
    return geom

def shapely_wgs_to_geojson(geom) -> dict:
    return {"type": "Feature", "properties": {}, "geometry": mapping(geom)}

def project_geom(geom_wgs):
    """Project a Polygon/MultiPolygon from WGS84 to RD New."""
    def project_poly(poly: Polygon) -> Polygon:
        exterior = wgs_to_rd_coords(list(poly.exterior.coords))
        interiors = [wgs_to_rd_coords(list(r.coords)) for r in poly.interiors]
        return Polygon(exterior, interiors)
    if isinstance(geom_wgs, Polygon):
        return project_poly(geom_wgs)
    elif isinstance(geom_wgs, MultiPolygon):
        return MultiPolygon([project_poly(p) for p in geom_wgs.geoms])
    else:
        raise ValueError("FG must be Polygon or MultiPolygon")

def unproject_geom(geom_rd):
    """Back to WGS84."""
    def unproject_poly(poly: Polygon) -> Polygon:
        exterior = rd_to_wgs_coords(list(poly.exterior.coords))
        interiors = [rd_to_wgs_coords(list(r.coords)) for r in poly.interiors]
        return Polygon(exterior, interiors)
    if isinstance(geom_rd, Polygon):
        return unproject_poly(geom_rd)
    elif isinstance(geom_rd, MultiPolygon):
        return MultiPolygon([unproject_poly(p) for p in geom_rd.geoms])
    else:
        raise ValueError("Unexpected geometry type")

def buffer_m(geom_wgs, radius_m: float, cap_style=1, join_style=1):
    """Buffer in metres using a local UTM, then return WGS84 geometry."""
    if radius_m <= 0:
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

    # remember CRS used
    st.session_state["last_metric_crs"] = epsg

    if isinstance(buf, Polygon):
        return unproject_poly(buf)
    else:
        return MultiPolygon([unproject_poly(p) for p in buf.geoms])

def offset_m(geom_wgs, offset_meters: float, cap_style=1, join_style=1):
    """Positive = outward, negative = inward (same local UTM approach)."""
    # Reuse buffer_m semantics; shapely negative buffer = inward offset
    return buffer_m(geom_wgs, offset_meters, cap_style=cap_style, join_style=join_style)

def area_perimeter_m(geom_wgs) -> Tuple[float, float]:
    """Return (area_m2, perimeter_m) using a local UTM for metric accuracy."""
    epsg, to_local, _ = _get_local_transformers(geom_wgs)

    def project_poly(poly: Polygon) -> Polygon:
        ext = [to_local.transform(x, y) for x, y in poly.exterior.coords]
        holes = [[to_local.transform(x, y) for x, y in r.coords] for r in poly.interiors]
        return Polygon(ext, holes)

    if isinstance(geom_wgs, Polygon):
        local = project_poly(geom_wgs)
    elif isinstance(geom_wgs, MultiPolygon):
        local = MultiPolygon([project_poly(p) for p in geom_wgs.geoms])
    else:
        raise ValueError("Geometry must be Polygon or MultiPolygon")

    # remember CRS used
    st.session_state["last_metric_crs"] = epsg

    return float(local.area), float(local.length)

# ----------------- Sidebar controls -----------------
# What does the drawn polygon represent?


st.sidebar.header("Drawing mode")
st.session_state["input_layer"] = st.sidebar.radio(
    "The polygon I draw represents:",
    ["FG", "CV", "GRB"],
    horizontal=True,
    index=["FG", "CV", "GRB"].index(st.session_state.get("input_layer", "FG")),
)


st.sidebar.header("Flight Height")
planned_fg_input_m = st.sidebar.number_input("Planned maximum flight height (m)", 0.0, 120.0, 50.0, 1.0,
    help="Your planned maximum flight height (up to 120 m). It will be lowered automatically if FG + buffer would exceed the 150 m CV apex."
)
st.sidebar.divider()

st.sidebar.header("Contingency Volume Calculator")
st.sidebar.caption("(Multirotor only, defaults = DJI Matrice 30)")

# --- Drone profile selector ---
# First-time defaults
default_profile_name = "DJI Matrice 30"
_ensure_param("drone_profile", default_profile_name)
_ensure_param("last_drone_profile", st.session_state["drone_profile"])

# Show selector
profile_choice = st.sidebar.selectbox(
    "Drone type",
    options=list(DRONE_PROFILES.keys()),
    index=list(DRONE_PROFILES.keys()).index(st.session_state["drone_profile"]),
    help="Choose a drone to prefill speed, reaction time, pitch, sensor errors, and CD."
)

# Apply profile if changed OR if first run with no params set
if (profile_choice != st.session_state["last_drone_profile"]) or any(
    k not in st.session_state for k in
    ["v0_ms","t_react_s","theta_deg","s_gps_m","s_pos_m","s_map_m","cd_m","h_baro_mode"]
):
    _apply_profile(profile_choice)
    st.session_state["drone_profile"] = profile_choice
    st.session_state["last_drone_profile"] = profile_choice

# Convenience handles
opt_alt_modes = ["Barometric (1 m)", "GPS-based (4 m)"]

# --- Inputs (prefilled from profile, still editable) ---
v0_ms = st.sidebar.number_input(
    "Max groundspeed V₀ (m/s)", 0.0, 40.0,
    value=float(st.session_state["v0_ms"]), step=0.5, key="v0_ms"
)
t_react_s = st.sidebar.number_input(
    "Reaction time t (s)", 0.0, 5.0,
    value=float(st.session_state["t_react_s"]), step=0.1, key="t_react_s"
)
theta_deg = st.sidebar.number_input(
    "Max pitch θ (deg)", 1.0, 45.0,
    value=float(st.session_state["theta_deg"]), step=1.0, key="theta_deg"
)
s_gps_m = st.sidebar.number_input(
    "GPS inaccuracy S_GPS (m)", 0.0, 10.0,
    value=float(st.session_state["s_gps_m"]), step=0.5, key="s_gps_m"
)
s_pos_m = st.sidebar.number_input(
    "Position hold error S_pos (m)", 0.0, 10.0,
    value=float(st.session_state["s_pos_m"]), step=0.1, key="s_pos_m"
)
s_map_m = st.sidebar.number_input(
    "Map error S_K (m)", 0.0, 10.0,
    value=float(st.session_state["s_map_m"]), step=0.5, key="s_map_m"
)

cv_m = cv_lateral_multirotor(v0_ms, t_react_s, theta_deg, s_gps_m, s_pos_m, s_map_m)

# Altitude measurement mode from profile
_ensure_param("h_baro_mode", DRONE_PROFILES[st.session_state["drone_profile"]]["h_baro_mode"])
h_baro_mode = st.sidebar.selectbox(
    "Altitude measurement",
    opt_alt_modes,
    index=opt_alt_modes.index(st.session_state["h_baro_mode"]),
    key="h_baro_mode"
)
h_baro_m = 1.0 if "Barometric" in h_baro_mode else 4.0

cd_m = st.sidebar.number_input(
    "Characteristic dimension CD (m)", 0.0, 10.0,
    value=float(st.session_state["cd_m"]), step=0.1, key="cd_m"
)
st.sidebar.caption("CD is the drone’s largest physical span, e.g., max diagonal tip-to-tip of the propellers.")

grb_method = st.sidebar.selectbox("Method", ["Simplified (1:1)", "Ballistic"], index=1)


# st.sidebar.header("Contingency Volume Calculator")
# st.sidebar.caption("(Multirotor only, defaults = DJI Matrice M30)")
# v0_ms = st.sidebar.number_input("Max groundspeed V₀ (m/s)", 0.0, 40.0, 23.0, 0.5)
# t_react_s = st.sidebar.number_input("Reaction time t (s)", 0.0, 5.0, 1.0, 0.1)
# theta_deg = st.sidebar.number_input("Max pitch θ (deg)", 1.0, 45.0, 35.0, 1.0)
# s_gps_m = st.sidebar.number_input("GPS inaccuracy S_GPS (m)", 0.0, 10.0, 3.0, 0.5)
# s_pos_m = st.sidebar.number_input("Position hold error S_pos (m)", 0.0, 10.0, 0.3, 0.5)
# s_map_m = st.sidebar.number_input("Map error S_K (m)", 0.0, 10.0, 1.0, 0.5)
# cv_m = cv_lateral_multirotor(v0_ms, t_react_s, theta_deg, s_gps_m, s_pos_m, s_map_m)

# h_baro_mode = st.sidebar.selectbox("Altitude measurement", ["Barometric (1 m)", "GPS-based (4 m)"], index=0)
# h_baro_m = 1.0 if "Barometric" in h_baro_mode else 4.0


# cd_m = st.sidebar.number_input("Characteristic dimension CD (m) [needs to be checked with m30]", 0.0, 10.0, 0.6, 0.1)
# st.sidebar.caption("CD is  the drone’s largest physical span, usually the maximum diagonal tip-to-tip of the propellers.")
# grb_method = st.sidebar.selectbox("Method", ["Simplified (1:1)", "Ballistic"], index=1)


h_rz = v0_ms * 0.7 * max(0.0, t_react_s)
h_cm = 0.5 * ((v0_ms**2) / (2.0 * G))  # keep consistent with the rest of your file
vertical_buffer_m = h_baro_m + h_rz + h_cm

# --- Apply constraints ---
allowed_fg_by_cv = max(0.0, CV_TOP_CAP_M - vertical_buffer_m)   # FG must also satisfy FG + buffer ≤ 150
fg_user_capped   = min(FG_TOP_CAP_M, planned_fg_input_m)         # never above 120 (UI already enforces)
calculated_h_fg_m = max(0.0, min(fg_user_capped, allowed_fg_by_cv))
h_cv_apex_m = calculated_h_fg_m + vertical_buffer_m              # ≤ 150 by construction


if grb_method.startswith("Simplified"):
    grb_margin = grb_simplified(h_cv_apex_m, cd_m)
else:
    grb_margin = grb_ballistic(v0_ms, h_cv_apex_m, cd_m)



st.sidebar.divider()


st.sidebar.header("Calculated values")



# --- Vertical buffer components ---

# Readout
st.sidebar.metric("Lateral CV margin", f"{cv_m:.1f} m")


st.sidebar.metric("Planned FG (user input)", f"{planned_fg_input_m:.1f} m")
st.sidebar.metric("Resulting FG apex H_FG", f"{calculated_h_fg_m:.1f} m")
st.sidebar.metric("Required Vertical Buffer", f"{vertical_buffer_m:.1f} m")
st.sidebar.metric("Resulting CV apex H_CV", f"{h_cv_apex_m:.1f} m")



if planned_fg_input_m > allowed_fg_by_cv + 1e-6:
    st.sidebar.warning(
        f"Planned FG reduced from {planned_fg_input_m:.1f} m to {calculated_h_fg_m:.1f} m "
        f"to keep CV apex ≤ {CV_TOP_CAP_M:.0f} m."
    )
if calculated_h_fg_m <= 0:
    st.sidebar.error("Flight not possible: buffer alone exceeds the CV ceiling of 150 m. Reduce speed or reaction time.")

st.sidebar.metric("GRB margin (m)", f"{grb_margin:.1f}")


st.sidebar.divider()

st.sidebar.header("Map Settings")

metric_crs = st.session_state.get("last_metric_crs", "local UTM")
st.sidebar.text(f"Metrics (projected in {metric_crs})")

default_center = [52.1, 5.3]
start_zoom = st.sidebar.slider("Initial zoom", 5, 14, 8)
tile_choice = st.sidebar.selectbox(
    "Base map",
    ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"],
    index=0,
)



# ---- KML/KMZ helpers ----

def _kml_color(hex_rgb: str, opacity: float = 0.6) -> str:
    """
    Convert '#RRGGBB' + opacity [0..1] to KML aabbggrr.
    KML expects little-endian ARGB in hex.
    """
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
    """
    Adds a (Multi)Polygon GeoJSON feature to a KML folder.
    - If extrude=True, creates a 3D volume up to height_m.
    - If extrude=False, creates a 2D polygon clamped to the ground.
    """
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
            # For 3D volumes (FG, CV, OV)
            pol.outerboundaryis = [(lon, lat, height_m) for lon, lat in outer]
            if holes:
                pol.innerboundaryis = [
                    [(lon, lat, height_m) for lon, lat in hole] for hole in holes
                ]
            pol.altitudemode = simplekml.AltitudeMode.relativetoground
            pol.extrude = 1
        else:
            # For 2D ground areas (GRB)
            pol.outerboundaryis = outer
            if holes:
                pol.innerboundaryis = holes
            pol.altitudemode = simplekml.AltitudeMode.clamptoground
            pol.extrude = 0

        # Style applies to both
        pol.style.linestyle.color = _kml_color(line_hex, 1.0)
        pol.style.linestyle.width = 2
        pol.style.polystyle.color = _kml_color(line_hex, fill_opacity)

  

    
def write_kmz(zones: dict, params: dict) -> bytes:
    """
    Build a KMZ from zones (FG/CV/OV/GRB) and return its content as bytes.
    This version creates volumes with zone-specific heights.
    """
    kml = simplekml.Kml()
    kml.document.name = f"DOZE Export {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    kml.document.description = (
        "DOZE — Drone Operation Zone Editor\n"
        "Based on LBA FG/CV/GRB guidance and EU 2019/947.\n\n"
        f"Parameters:\n{json.dumps(params, indent=2)}"
    )

    palette = {
        "FG":  ("#15c048", 0.8),
        "CV":  ("#ff9800", 0.5),
        "OV":  ("#42a5f5", 0.05),
        "GRB": ("#e53935", 0.9),
    }

    # Define the vertical properties for each zone from the passed parameters
    h_fg = params['h_fg_m']
    h_cv = params['h_cv_m']

    zone_properties = {
        "FG":  {"height": h_fg,                 "extrude": True},
        "CV":  {"height": h_cv,                 "extrude": True},
        "OV":  {"height": h_cv,                 "extrude": True},   # OV up to CV
        "GRB": {"height": GRB_PRISM_HEIGHT_M,   "extrude": True},   # << was 0 / False
    }

    # One folder per layer, applying the correct properties
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
    # In-memory KMZ creation (remains the same)
    kml_content_bytes = kml.kml().encode('utf-8')
    in_memory_file = io.BytesIO()
    with zipfile.ZipFile(in_memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('doc.kml', kml_content_bytes)
    in_memory_file.seek(0)
    return in_memory_file.read()

  


# ----------------- Session state & map -----------------
st.session_state.setdefault("input_geojson", None)     # what the user drew
st.session_state.setdefault("input_layer", "FG")       # FG | CV | GRB
st.session_state.setdefault("zones", None)
st.session_state.setdefault("zones_bounds", None)
st.session_state.setdefault("export_params", None)

st.markdown("**Instructions**")
st.markdown(
    "1) Choose what your polygon represents: Flight Geometry (FG), Contingency Volume (CV), or Ground Risk Buffer (GRB).\n"
    "2) Draw the polygon with the tool.\n"
    "3) Click **Save**.\n"
    "4) Click **Compute Zones** to derive the other boundaries.\n"
    "5) Click **Download KMZ** and open it in Google Earth."
)


# Define colors for zones
colors = {
    "FG":  {"color": "#1565c0", "fillOpacity": 0.15},
    "CV":  {"color": "#ff9800", "fillOpacity": 0.15},
    "OV":  {"color": "#42a5f5", "fillOpacity": 0.05},
    "GRB": {"color": "#e53935", "fillOpacity": 0.12},
}

# Initialize map
m = folium.Map(location=default_center, zoom_start=start_zoom, tiles=tile_choice, control_scale=True)
draw = Draw(
    export=False,
    position="topleft",
    draw_options={
        "polyline": False,
        "rectangle": False,
        "circle": False,
        "marker": False,
        "circlemarker": False,
        "polygon": {
            "allowIntersection": False,
            "showArea": True,
            "shapeOptions": {"color": "#2e7d32", "weight": 3, "fillOpacity": 0.2},
        },
    },
    edit_options={"edit": True, "remove": True},
)
draw.add_to(m)

if st.session_state.input_geojson:
    saved_name = f"{st.session_state.input_layer} (saved)"
    this_color = {
        "FG":  "#1565c0",
        "CV":  "#ff9800",
        "OV":  "#42a5f5",
        "GRB": "#e53935",
    }.get(st.session_state.input_layer, "#1565c0")

    folium.GeoJson(
        st.session_state.input_geojson,
        name=saved_name,
        style_function=lambda _: {"color": this_color, "weight": 3, "fillOpacity": 0.15},
        tooltip=saved_name,
    ).add_to(m)

# Add computed zones (avoid duplicating the saved layer)
if st.session_state.zones:
    for name in ["FG", "CV", "OV", "GRB"]:
        if name in st.session_state.zones and st.session_state.zones[name]:
            if name == st.session_state.get("input_layer"):
                continue  # skip duplicating the saved layer
            folium.GeoJson(
                st.session_state.zones[name],
                name=name,
                style_function=(lambda _, n=name: {"color": colors[n]["color"], "weight": 3, "fillOpacity": colors[n]["fillOpacity"]}),
                tooltip=name,
            ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # Auto-zoom to the widest derived layer available
    if st.session_state.zones_bounds:
        minx, miny, maxx, maxy = st.session_state.zones_bounds
        m.fit_bounds([[miny, minx], [maxy, maxx]])

map_data = st_folium(
    m,
    width=None,
    height=650,
    returned_objects=["last_active_drawing", "all_drawings", "last_drawn"],
    key="doze_map",
)

def get_current_polygon_feature(md: dict) -> Optional[dict]:
    if not md:
        return None
    drawings = md.get("all_drawings") or []
    polys = [g for g in drawings if g and g.get("geometry", {}).get("type", "").lower().endswith("polygon")]
    return polys[-1] if polys else None

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    if st.button(f"Save {st.session_state.input_layer}", use_container_width=True):
        feature = get_current_polygon_feature(map_data)
        if feature is None:
            st.warning("Please draw a polygon first (use the polygon tool).")
        else:
            st.session_state.input_geojson = {
                "type": "Feature",
                "properties": {"name": st.session_state.input_layer, "source": "leaflet-draw"},
                "geometry": feature["geometry"],
            }
            st.success(f"{st.session_state.input_layer} saved to session.")
            st.rerun()

with col2:
    if st.button("Clear Input", use_container_width=True):
        st.session_state.input_geojson = None
        st.session_state.zones = None
        st.session_state.zones_bounds = None
        st.session_state.export_params = None
        st.rerun()

with col3:
    compute_button_clicked = st.button("Compute Zones", use_container_width=True)

with col4:
    # Always show a Download KMZ control.
    if st.session_state.get("zones") and st.session_state.get("export_params"):
        kmz_bytes = write_kmz(st.session_state.zones, st.session_state.export_params)
        st.download_button(
            label=f"Download KMZ (FG {calculated_h_fg_m:.0f} m, CV {h_cv_apex_m:.0f} m)",
            data=kmz_bytes,
            file_name=f"DOZE_zones_{datetime.now().strftime('%Y%m%d_%H%M')}.kmz",
            mime="application/vnd.google-earth.kmz",
            use_container_width=True,
            key="download_kmz_ready",
        )
    else:
        if st.button("Download KMZ", use_container_width=True, key="download_kmz_stub"):
            if not st.session_state.get("input_geojson"):
                st.warning("Please draw and **Save** your polygon first.")
            else:
                st.warning("Please click **Compute Zones** before downloading the KMZ.")

    

# --------------- Compute & render zones ---------------

if compute_button_clicked and st.session_state.input_geojson:
    try:
        base_layer = st.session_state.input_layer
        base_geom  = geojson_to_shapely_wgs(st.session_state.input_geojson)

        fg_geom = cv_geom = ov_geom = grb_geom = None

        if base_layer == "FG":
            fg_geom = base_geom
            cv_geom = buffer_m(fg_geom, cv_m, cap_style=1, join_style=1)
            ov_geom = unary_union([fg_geom, cv_geom])  # typically equals CV
            grb_geom = buffer_m(ov_geom, grb_margin, cap_style=1, join_style=1)

        elif base_layer == "CV":
            cv_geom = base_geom
            fg_geom = offset_m(cv_geom, -cv_m, cap_style=1, join_style=1)
            if fg_geom is None:
                st.warning("The drawn CV is too narrow to derive an FG with the selected CV margin.")
                ov_geom = cv_geom
            else:
                ov_geom = unary_union([fg_geom, cv_geom])
            grb_geom = buffer_m(ov_geom, grb_margin, cap_style=1, join_style=1)

        elif base_layer == "GRB":
            grb_geom = base_geom
            ov_geom = offset_m(grb_geom, -grb_margin, cap_style=1, join_style=1)
            if ov_geom is None:
                st.error("The drawn GRB is too narrow to derive an OV with the selected GRB margin.")
            else:
                cv_geom = ov_geom  # OV == CV in this model
                fg_geom = offset_m(cv_geom, -cv_m, cap_style=1, join_style=1)
                if fg_geom is None:
                    st.warning("The derived OV is too narrow to derive an FG with the selected CV margin.")

        # Collect any successfully derived layers
        zones_data = {}
        if fg_geom:  zones_data["FG"]  = shapely_wgs_to_geojson(fg_geom)
        if cv_geom:  zones_data["CV"]  = shapely_wgs_to_geojson(cv_geom)
        if ov_geom:  zones_data["OV"]  = shapely_wgs_to_geojson(ov_geom)
        if grb_geom: zones_data["GRB"] = shapely_wgs_to_geojson(grb_geom)

        if not zones_data:
            st.error("No valid zones could be computed. Adjust your margins or draw a larger polygon.")
        else:
            st.session_state.zones = zones_data

            # Fit bounds to the widest available layer
            for _g in [grb_geom, ov_geom, cv_geom, fg_geom]:
                if _g:
                    minx, miny, maxx, maxy = _g.bounds
                    st.session_state.zones_bounds = (minx, miny, maxx, maxy)
                    break

            # Metrics (only for layers that exist)
            lines = []
            if fg_geom:
                fg_area, fg_perim = area_perimeter_m(fg_geom);   lines.append(f"- **FG**: area {fg_area:,.0f} m² • perimeter {fg_perim:,.0f} m")
            if cv_geom:
                cv_area, cv_perim = area_perimeter_m(cv_geom);   lines.append(f"- **CV**: area {cv_area:,.0f} m² • perimeter {cv_perim:,.0f} m")
            if ov_geom:
                ov_area, ov_perim = area_perimeter_m(ov_geom);   lines.append(f"- **OV**: area {ov_area:,.0f} m² • perimeter {ov_perim:,.0f} m")
            if grb_geom:
                grb_area, grb_perim = area_perimeter_m(grb_geom); lines.append(f"- **GRB**: area {grb_area:,.0f} m² • perimeter {grb_perim:,.0f} m")

            if lines:
                st.subheader("Metrics (projected in EPSG:28992)")
                st.write("\n".join(lines))

            # Export params (heights & margins still apply regardless of which base was drawn)
            st.session_state.export_params = {
                "h_fg_m": calculated_h_fg_m,
                "h_cv_m": h_cv_apex_m,
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
                "crs_buffering": st.session_state.get("last_metric_crs", "local UTM"),
                "altitude_mode": "relativeToGround",
                "base_layer": base_layer,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            st.success("Zones computed. They’re drawn on the main map above.")
            st.rerun()

    except Exception as e:
        st.error(f"Failed to compute zones: {e}")
        st.session_state.zones = None
        st.session_state.export_params = None

elif compute_button_clicked and not st.session_state.input_geojson:
    st.warning("Please save your polygon before computing zones.")
# --------------- FG readout ---------------
st.subheader("Current input polygon")
if st.session_state.input_geojson:
    st.write(f"Layer: **{st.session_state.input_layer}**")
    st.code(
        json.dumps(
            {
                "type": st.session_state.input_geojson["geometry"]["type"],
                "coordinates": st.session_state.input_geojson["geometry"]["coordinates"][:1],
            },
            indent=2,
        ),
        language="json",
    )
else:
    st.write("No input polygon saved yet.")
    
    
st.caption("© DOZE — Drone Operation Zone Editor")