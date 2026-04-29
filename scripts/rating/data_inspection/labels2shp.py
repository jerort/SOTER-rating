import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ==========
# CONFIG
# ==========
input_xlsx = r"labels.xlsx"
sheet_name = "FINAL"

col_lat = "LAT"
col_lon = "LON"
col_label = "Etiqueta"

output_shp = r"labels_shp/etiquetas.shp"
os.makedirs(os.path.dirname(output_shp), exist_ok=True)

# ==========
# LEER EXCEL
# ==========
df = pd.read_excel(input_xlsx, sheet_name=sheet_name)

# Validaciones básicas
required = {col_lat, col_lon, col_label}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Faltan columnas requeridas en el Excel: {missing}")

# Asegurar numéricos y limpiar filas sin coordenadas o etiqueta
df[col_lat] = pd.to_numeric(df[col_lat], errors="coerce")
df[col_lon] = pd.to_numeric(df[col_lon], errors="coerce")
df[col_label] = pd.to_numeric(df[col_label], errors="coerce")

df = df.dropna(subset=[col_lat, col_lon, col_label]).copy()

# Convertir etiqueta a int (1..5)
df[col_label] = df[col_label].astype(int)

# ==========
# CREAR GEODATAFRAME (EPSG:4326) — sólo columna de clase
# ==========
geometry = [Point(xy) for xy in zip(df[col_lon], df[col_lat])]
gdf = gpd.GeoDataFrame(
    df[[col_label]].rename(columns={col_label: "ETIQUETA"}),
    geometry=geometry,
    crs="EPSG:4326",
)

# ==========
# EXPORTAR SHAPEFILE ÚNICO
# ==========
gdf.to_file(output_shp, driver="ESRI Shapefile", encoding="utf-8")
print(f"[OK] Generado: {output_shp}  ({len(gdf)} puntos)")
print("Listo.")
