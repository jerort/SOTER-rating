"""
Generate a standalone HTML viewer that overlays the road-quality heatmap
on satellite imagery with hover tooltips showing calibrated class
probabilities.

The HTML uses Leaflet with Esri World Imagery tiles (free, no API key)
and embeds the road segments as GeoJSON coloured by the estimated rating.
Hovering on a segment shows a tooltip with the per-class probability
distribution from the best-performing calibrator (Gaussian vs Ordinal,
selected automatically by highest mean confidence).
"""

import json
from pathlib import Path

import geopandas as gpd

# ======================================================================
# User-configurable parameters  (edit these, then run the script)
# ======================================================================
# Which roads to include (None = all roads in the shapefile)
ROAD     = None           # e.g. "A-490" to show only that road, None for all

# Where to center the initial view (None = fit all segments)
CENTROID = "A-492__4"     # e.g. "A-492__4" to center+zoom on that segment

# Output path — None = auto-generated in the same directory
OUTPUT_HTML = None

# ======================================================================
# Paths (same layout as sibling scripts)
# ======================================================================
HEATMAP_SHP = Path(__file__).resolve().parent / "predictions_heatmap.shp"
OUTPUT_DIR  = Path(__file__).resolve().parent

# ======================================================================
# HTML template  — placeholders are replaced with json.dumps() values
# ======================================================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SOTER — Heatmap: __TITLE__</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: sans-serif; background: #efefef;
      display: flex; flex-direction: column; align-items: center;
      padding: 20px; min-height: 100vh;
    }
    .container {
      max-width: 900px; width: 100%; background: white;
      border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,.1);
      overflow: hidden;
    }
    .header { padding: 16px 20px; }
    .title  { font-size: 1.25rem; margin-bottom: 4px; }
    .subtitle { color: #555; font-size: .95rem; margin: 0; }

    /* ── Map ────────────────────────────────────────────────── */
    #map {
      width: 100%; height: 520px; position: relative;
      background: #1a1a2e;
    }

    /* ── Legend ──────────────────────────────────────────────── */
    .legend {
      position: absolute; bottom: 24px; right: 10px; z-index: 999;
      background: rgba(255,255,255,.92); padding: 10px 14px;
      border-radius: 6px; font-size: 12px;
      box-shadow: 0 2px 6px rgba(0,0,0,.2); pointer-events: none;
    }
    .legend-title { font-weight: 700; margin-bottom: 6px; }
    .legend-item  { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
    .legend-swatch { width: 28px; height: 5px; border-radius: 2px; }

    /* ── Hover tooltip ──────────────────────────────────────── */
    .tip-title { font-weight: 700; margin-bottom: 4px; font-size: 13px; }
    .tip-sub   { color: #555; font-size: 11px; margin-bottom: 6px; }
    .prob-row  {
      display: flex; align-items: center; gap: 4px;
      margin: 2px 0; font-size: 11px; font-variant-numeric: tabular-nums;
    }
    .prob-label {
      width: 12px; text-align: center; font-weight: 700; flex-shrink: 0;
    }
    .prob-track {
      width: 90px; height: 8px; background: #e8e8e8;
      border-radius: 4px; overflow: hidden; flex-shrink: 0;
    }
    .prob-fill { height: 100%; border-radius: 4px; }
    .prob-pct  { width: 38px; text-align: right; flex-shrink: 0; }
  </style>
</head>
<body>

  <div class="container">
    <div class="header">
      <h2 class="title">SOTER — Mapa de calidad del pavimento</h2>
      <p class="subtitle">
        Pasa el cursor sobre un tramo para ver su calificación.
      </p>
    </div>

    <div id="map">
      <div class="legend">
        <div class="legend-title">Calidad del pavimento</div>
        <div class="legend-item"><span class="legend-swatch" style="background:#1a9850"></span> 1 — Excelente</div>
        <div class="legend-item"><span class="legend-swatch" style="background:#91cf60"></span> 2 — Bueno</div>
        <div class="legend-item"><span class="legend-swatch" style="background:#fee08b"></span> 3 — Regular</div>
        <div class="legend-item"><span class="legend-swatch" style="background:#fc8d59"></span> 4 — Malo</div>
        <div class="legend-item"><span class="legend-swatch" style="background:#d73027"></span> 5 — Muy malo</div>
      </div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
  (function () {
    // ── Embedded data (injected by Python) ───────────────────────────
    var geojson     = __GEOJSON__;
    var sw          = __SW__;
    var ne          = __NE__;
    var initCenter  = __CENTER__;
    var highlightId = __HIGHLIGHT__;

    var COLORS = {
      1: "#1a9850", 2: "#91cf60", 3: "#fee08b",
      4: "#fc8d59", 5: "#d73027"
    };
    var CLASS_NAMES = {
      1: "Excelente", 2: "Bueno", 3: "Regular",
      4: "Malo", 5: "Muy malo"
    };

    // ── Map setup ────────────────────────────────────────────────────
    var map = L.map("map", { zoomControl: true });

    L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/" +
      "World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { attribution: "Tiles &copy; Esri", maxZoom: 19 }
    ).addTo(map);

    if (initCenter) {
      map.setView(initCenter, 16);
    } else {
      map.fitBounds([sw, ne], { padding: [30, 30] });
    }

    // ── Build tooltip HTML ───────────────────────────────────────────
    function buildTooltip(p) {
      var est = p.estimate || "—";
      var html = '<div class="tip-title">' + p.identifier + '</div>';
      html += '<div class="tip-sub">' + (CLASS_NAMES[est] || '') + '</div>';

      // Probability bars (p1..p5 injected by Python as best calibrator)
      var hasProbs = (p.p1 != null);
      if (hasProbs) {
        for (var c = 1; c <= 5; c++) {
          var v = p["p" + c];
          var pct = (v != null) ? (v * 100) : 0;
          html += '<div class="prob-row">'
            + '<span class="prob-label" style="color:' + COLORS[c] + '">' + c + '</span>'
            + '<div class="prob-track"><div class="prob-fill" style="width:'
            + pct.toFixed(0) + '%;background:' + COLORS[c] + '"></div></div>'
            + '<span class="prob-pct">' + pct.toFixed(1) + '%</span>'
            + '</div>';
        }
      }
      return html;
    }

    // ── Heatmap layer ────────────────────────────────────────────────
    var defaultWeight = 6;

    L.geoJSON(geojson, {
      style: function (feature) {
        var est = feature.properties.estimate || 3;
        var hl  = highlightId &&
                  feature.properties.identifier === highlightId;
        return {
          color:    COLORS[est] || "#888",
          weight:   hl ? 9 : defaultWeight,
          opacity:  0.88,
          lineCap:  "round",
          lineJoin: "round"
        };
      },
      onEachFeature: function (feature, layer) {
        // Hover tooltip with probability bars
        layer.bindTooltip(buildTooltip(feature.properties), {
          sticky: true,
          direction: "top",
          offset: [0, -8],
          className: "leaflet-tooltip"
        });
        // Highlight on hover
        layer.on("mouseover", function () {
          layer.setStyle({ weight: 10, opacity: 1 });
          layer.bringToFront();
        });
        layer.on("mouseout", function () {
          var est = feature.properties.estimate || 3;
          var hl  = highlightId &&
                    feature.properties.identifier === highlightId;
          layer.setStyle({
            weight: hl ? 9 : defaultWeight,
            opacity: 0.88
          });
        });
      }
    }).addTo(map);
  })();
  </script>
</body>
</html>"""


def pick_best_calibrator(gdf: gpd.GeoDataFrame) -> str:
    """Return 'gau' or 'ord' based on which has higher mean confidence."""
    gau_ok = "gau_conf" in gdf.columns and gdf["gau_conf"].notna().any()
    ord_ok = "ord_conf" in gdf.columns and gdf["ord_conf"].notna().any()
    if gau_ok and ord_ok:
        return "gau" if gdf["gau_conf"].mean() >= gdf["ord_conf"].mean() else "ord"
    if gau_ok:
        return "gau"
    if ord_ok:
        return "ord"
    return ""


def main() -> None:
    # ── Load shapefile ────────────────────────────────────────────────
    print(f"Loading {HEATMAP_SHP} ...")
    gdf = gpd.read_file(HEATMAP_SHP)
    print(f"  {len(gdf)} total segments")

    # ── Filter ────────────────────────────────────────────────────────
    if ROAD:
        gdf = gdf[gdf["road"] == ROAD].copy()
        title_label = "_" + ROAD
    else:
        title_label = ""

    highlight_id = CENTROID if CENTROID else None

    if gdf.empty:
        print("No segments found for the given filter.")
        return
    print(f"  {len(gdf)} segments after filtering")

    # ── Pick best calibrator and rename p1..p5 ────────────────────────
    cal = pick_best_calibrator(gdf)
    if cal:
        cal_label = {"gau": "Gaussiana", "ord": "Ordinal"}[cal]
        for k in range(1, 6):
            gdf[f"p{k}"] = gdf[f"{cal}_p{k}"]
        print(f"  Calibrator: {cal_label} (mean conf "
              f"{gdf[f'{cal}_conf'].mean():.3f})")
    else:
        cal_label = "sin calibración"
        print("  WARNING: no calibration probabilities found")

    # ── Reproject to WGS84 for Leaflet ────────────────────────────────
    gdf_wgs = gdf.to_crs("EPSG:4326")

    # ── Bounds and optional center ────────────────────────────────────
    minx, miny, maxx, maxy = gdf_wgs.total_bounds
    sw = [miny, minx]  # [lat, lon]
    ne = [maxy, maxx]

    center = None
    if highlight_id and highlight_id in gdf_wgs["identifier"].values:
        seg = gdf_wgs.loc[gdf_wgs["identifier"] == highlight_id].geometry.iloc[0]
        pt = seg.centroid
        center = [pt.y, pt.x]

    # ── Build GeoJSON (keep only useful columns) ──────────────────────
    keep = ["identifier", "estimate", "label", "subset",
            "road", "p1", "p2", "p3", "p4", "p5", "geometry"]
    keep = [c for c in keep if c in gdf_wgs.columns]
    geojson_str = gdf_wgs[keep].to_json()

    # ── Assemble HTML ─────────────────────────────────────────────────
    html = (
        HTML_TEMPLATE
        .replace("__TITLE__", title_label)
        .replace("__CALIBRATOR_LABEL__", f"cal. {cal_label}")
        .replace("__GEOJSON__", geojson_str)
        .replace("__SW__", json.dumps(sw))
        .replace("__NE__", json.dumps(ne))
        .replace("__CENTER__", json.dumps(center))
        .replace("__HIGHLIGHT__", json.dumps(highlight_id))
    )

    # ── Write output ──────────────────────────────────────────────────
    if OUTPUT_HTML:
        out_path = Path(OUTPUT_HTML)
    else:
        safe = title_label.replace("/", "-").replace("\\", "-").replace(" ", "_")
        out_path = OUTPUT_DIR / f"heatmap_viewer{safe}.html"

    out_path.write_text(html, encoding="utf-8")
    print(f"Viewer written to {out_path}")


if __name__ == "__main__":
    main()
