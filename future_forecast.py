import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

PRED_CSV = "leadcont.csv"
COORDS_CSV = "Geospatial_Coordinates.csv"
PIPES_PATH = "distribution-watermain-4326.geojson"

if not os.path.exists(PRED_CSV): exit(1)

preds = pd.read_csv(PRED_CSV)
preds.columns = preds.columns.str.strip()
preds["PartialPostalCode"] = preds["PartialPostalCode"].astype(str).str.strip().str.upper()
preds["year"] = pd.to_numeric(preds["year"], errors="coerce").astype("Int64")
preds["y_pred_ppm"] = pd.to_numeric(preds["y_pred_ppm"], errors="coerce")

coords = pd.read_csv(COORDS_CSV)
coords.columns = coords.columns.str.strip()
coords = coords.rename(columns={"Postal Code": "PartialPostalCode"})
coords["PartialPostalCode"] = coords["PartialPostalCode"].astype(str).str.strip().str.upper()

pipes = None
if os.path.exists(PIPES_PATH):
    try:
        pipes = gpd.read_file(PIPES_PATH)
    except:
        pass

needed_seed_years = [2023, 2024, 2025]
have_years = set(preds["year"].dropna().unique().tolist())
missing = [y for y in needed_seed_years if y not in have_years]
if missing: exit(1)

piv = preds.pivot_table(index="PartialPostalCode", columns="year", values="y_pred_ppm", aggfunc="mean")
vals_2026 = piv[[2023, 2024, 2025]].mean(axis=1, skipna=True)

forecast_2026 = vals_2026.reset_index().rename(columns={0: "y_pred_ppm"}).assign(year=2026, y_true_ppm="")[["year", "PartialPostalCode", "y_pred_ppm", "y_true_ppm"]]

preds_no_2026 = preds[preds["year"] != 2026].copy()
preds_all = pd.concat([preds_no_2026, forecast_2026], ignore_index=True)
preds_all.to_csv(PRED_CSV, index=False)

g = preds_all.merge(coords, on="PartialPostalCode", how="left")
g = g.dropna(subset=["Longitude", "Latitude"]).copy()

gdf = gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g["Longitude"], g["Latitude"]), crs="EPSG:4326")

vmin, vmax = 1e-6, 1e-5
sub = gdf[gdf["year"] == 2026]
if not sub.empty:
    fig, ax = plt.subplots(figsize=(10, 10))
    if pipes is not None:
        try:
            pipes.plot(ax=ax, color="lightgray", linewidth=0.5, alpha=0.4)
        except:
            pass
    sub.plot(ax=ax, column="y_pred_ppm", cmap="viridis", markersize=60, legend=True, vmin=vmin, vmax=vmax, missing_kwds={"color": "lightgray", "label": "No data"})
    ax.set_title("Forecast Lead (ppm) — 2026\nMean of predicted 2023–2025")
    ax.axis("off")
    plt.savefig("out/maps/forecasts/Forecast_Lead_2026.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
