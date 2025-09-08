# DOZE — Draw FG + CV/OV/GRB overlays
# Adds buffering with Shapely in EPSG:28992 and renders layers on the map.

import json
import math
from typing import Optional, Tuple

import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from pyproj import Transformer

st.set_page_config(page_title="DOZE — Zones Preview", layout="wide")

st.title("DOZE — Draw Flight Geography (FG)")
st.caption("Based on https://www.lba.de/SharedDocs/Downloads/DE/B/B5_UAS/Leitfaden_FG_CV_GRB_eng.pdf which are in turn based on Regulation (EU) 2019/947.")
st.caption("This application DOES NOT take into account locations where you are and are not allowed to fly.")


# ----------------- Constants & helpers -----------------
G = 9.81
MAX_HEIGHT_AGL_M = 120.0

def cv_lateral_multirotor(v0_ms, t_react_s, theta_deg, s_gps_m, s_pos_m, s_map_m) -> float:
    theta_rad = math.radians(max(1e-6, theta_deg))
    s_rz = v0_ms * max(0.0, t_react_s)
    s_cm = (v0_ms**2) / (G * math.tan(theta_rad))
    return s_gps_m + s_pos_m + s_map_m + s_rz + s_cm

def cv_vertical_multirotor(h_fg_m, v0_ms, t_react_s, h_baro_m) -> float:
    h_rz = v0_ms * 0.7 * max(0.0, t_react_s)
    h_cm = (v0_ms**2) / (2.0 * G)
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
    """Buffer in metres using RD New, then return WGS84."""
    if radius_m <= 0:
        return geom_wgs
    rd = project_geom(geom_wgs)
    buf = rd.buffer(radius_m, cap_style=cap_style, join_style=join_style)
    return unproject_geom(buf)

def area_perimeter_m(geom_wgs) -> Tuple[float, float]:
    """Return (area_m2, perimeter_m) using RD New for metric accuracy."""
    rd = project_geom(geom_wgs)
    return float(rd.area), float(rd.length)

# ----------------- Sidebar controls -----------------
st.sidebar.header("Map Settings")
default_center = [52.1, 5.3]
start_zoom = st.sidebar.slider("Initial zoom", 5, 14, 8)
tile_choice = st.sidebar.selectbox(
    "Base map",
    ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter", "Stamen Terrain", "Stamen Toner"],
    index=0,
)

st.sidebar.header("CV Calculator (Multirotor, defaults = DJI Matrice M30)")
v0_ms = st.sidebar.number_input("Max groundspeed V₀ (m/s)", 0.0, 40.0, 23.0, 0.5)
t_react_s = st.sidebar.number_input("Reaction time t (s)", 0.0, 5.0, 1.0, 0.1)
theta_deg = st.sidebar.number_input("Max pitch θ (deg)", 1.0, 45.0, 35.0, 1.0)
s_gps_m = st.sidebar.number_input("GPS inaccuracy S_GPS (m)", 0.0, 10.0, 3.0, 0.5)
s_pos_m = st.sidebar.number_input("Position hold error S_pos (m)", 0.0, 10.0, 0.3, 0.5)
s_map_m = st.sidebar.number_input("Map error S_K (m)", 0.0, 10.0, 1.0, 0.5)

st.sidebar.divider()
h_fg_m = st.sidebar.number_input("Max flight height (m AGL)", 0.0, 150.0, MAX_HEIGHT_AGL_M, 1.0)
h_baro_mode = st.sidebar.selectbox("Altitude measurement", ["Barometric (1 m)", "GPS-based (4 m)"], index=0)
h_baro_m = 1.0 if "Barometric" in h_baro_mode else 4.0

cv_m = cv_lateral_multirotor(v0_ms, t_react_s, theta_deg, s_gps_m, s_pos_m, s_map_m)
h_cv_m = cv_vertical_multirotor(h_fg_m, v0_ms, t_react_s, h_baro_m)

st.sidebar.metric("Lateral CV margin (m)", f"{cv_m:.1f}")
st.sidebar.metric("Vertical CV apex H_CV (m AGL)", f"{h_cv_m:.1f}")

st.sidebar.header("GRB (Multirotor)")
cd_m = st.sidebar.number_input("Characteristic dimension CD (m) [needs to be checked with m30]", 0.0, 10.0, 0.6, 0.1)
grb_method = st.sidebar.selectbox("Method", ["Simplified (1:1)", "Ballistic"], index=1)

if grb_method.startswith("Simplified"):
    grb_margin = grb_simplified(h_cv_m, cd_m)
else:
    grb_margin = grb_ballistic(v0_ms, h_cv_m, cd_m)

st.sidebar.metric("GRB margin (m)", f"{grb_margin:.1f}")

# ... (previous imports and helper functions)

# ----------------- Session state & map -----------------
st.session_state.setdefault("fg_geojson", None)
st.session_state.setdefault("zones", None)
st.session_state.setdefault("zones_bounds", None)  # (minx, miny, maxx, maxy) in WGS84

st.markdown("**Instructions**")
st.markdown(
    "1) Draw your FG polygon with the tool.\n"
    "2) Click **Save FG**.\n"
    "3) Click **Compute Zones** to preview CV, OV, GRB."
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

# Add saved FG if it exists
if st.session_state.fg_geojson:
    folium.GeoJson(
        st.session_state.fg_geojson,
        name="FG (saved)",
        style_function=lambda _: {"color": "#1565c0", "weight": 3, "fillOpacity": 0.15},
        tooltip="FG (saved)",
    ).add_to(m)

# Add computed zones if they exist
if st.session_state.zones:
    for name in ["CV", "OV", "GRB"]:
        folium.GeoJson(
            st.session_state.zones[name],
            name=name,
            style_function=(lambda _, n=name: {"color": colors[n]["color"], "weight": 3, "fillOpacity": colors[n]["fillOpacity"]}),
            tooltip=name,
        ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # Auto-zoom to the full zones extent
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

col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
with col1:
    if st.button("Save FG"):
        feature = get_current_polygon_feature(map_data)
        if feature is None:
            st.warning("Please draw a polygon first (use the polygon tool).")
        else:
            st.session_state.fg_geojson = {
                "type": "Feature",
                "properties": {"name": "FG", "source": "leaflet-draw"},
                "geometry": feature["geometry"],
            }
            st.success("FG saved to session.")
            # Important: Rerun to update the map with the saved FG
            st.rerun()

with col2:
    if st.button("Clear FG"):
        st.session_state.fg_geojson = None
        st.session_state.zones = None # Also clear zones when FG is cleared
        st.session_state.zones_bounds = None
        st.rerun()

# --------------- Compute & render zones ---------------
# Use a form to group the button and prevent immediate reruns on input changes
with st.form("zone_computation_form"):
    compute_button_clicked = st.form_submit_button("Compute Zones")

if compute_button_clicked and st.session_state.fg_geojson:
    try:
        fg_geom = geojson_to_shapely_wgs(st.session_state.fg_geojson)

        cv_geom  = buffer_m(fg_geom, cv_m, cap_style=1, join_style=1)
        ov_geom  = unary_union([fg_geom, cv_geom])
        grb_geom = buffer_m(ov_geom, grb_margin, cap_style=1, join_style=1)

        # Areas/perimeters
        fg_area, fg_perim   = area_perimeter_m(fg_geom)
        cv_area, cv_perim   = area_perimeter_m(cv_geom)
        ov_area, ov_perim   = area_perimeter_m(ov_geom)
        grb_area, grb_perim = area_perimeter_m(grb_geom)

        # Save zones as GeoJSON in session
        zones_data = {
            "FG":  shapely_wgs_to_geojson(fg_geom),
            "CV":  shapely_wgs_to_geojson(cv_geom),
            "OV":  shapely_wgs_to_geojson(ov_geom),
            "GRB": shapely_wgs_to_geojson(grb_geom),
        }
        st.session_state.zones = zones_data

        # Save bounds for fit_bounds
        minx, miny, maxx, maxy = grb_geom.bounds  # use widest layer
        st.session_state.zones_bounds = (minx, miny, maxx, maxy)

        st.subheader("Metrics (projected in EPSG:28992)")
        st.write(
            f"- **FG**: area {fg_area:,.0f} m² • perimeter {fg_perim:,.0f} m\n"
            f"- **CV**: area {cv_area:,.0f} m² • perimeter {cv_perim:,.0f} m\n"
            f"- **OV**: area {ov_area:,.0f} m² • perimeter {ov_perim:,.0f} m\n"
            f"- **GRB**: area {grb_area:,.0f} m² • perimeter {grb_perim:,.0f} m"
        )
        st.success("Zones computed. They’re drawn on the main map above.")
        
        # Rerun to ensure the map object `m` gets updated with the new layers
        # and fit_bounds is applied in the next st_folium render.
        st.rerun()

    except Exception as e:
        st.error(f"Failed to compute zones: {e}")
elif compute_button_clicked and not st.session_state.fg_geojson:
    st.warning("Please save your FG polygon first before computing zones.")


# --------------- FG readout ---------------
st.subheader("Current FG (captured)")
if st.session_state.fg_geojson:
    st.code(
        json.dumps(
            {
                "type": st.session_state.fg_geojson["geometry"]["type"],
                "coordinates": st.session_state.fg_geojson["geometry"]["coordinates"][:1],
            },
            indent=2,
        ),
        language="json",
    )
else:
    st.write("No FG saved yet.")

st.caption("© DOZE — Drone Operation Zone Editor (buffering preview)")