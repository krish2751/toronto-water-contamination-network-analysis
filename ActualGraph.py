import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

#importing all the req. csvs
lead_data = pd.read_csv("leadcont.csv")
postal_coords = pd.read_csv("Geospatial_Coordinates.csv")
lead_data.columns = lead_data.columns.str.strip()
postal_coords.columns = postal_coords.columns.str.strip()
postal_coords.rename(columns={'Postal Code': 'PartialPostalCode'}, inplace=True)

lead_data['Sample Date'] = pd.to_datetime(lead_data['Sample Date'], errors='coerce')
lead_data['year'] = lead_data['Sample Date'].dt.year.astype(str)
lead_data['Lead Amount (ppm)'] = pd.to_numeric(lead_data['Lead Amount (ppm)'], errors='coerce')
lead_data['Lead Amount (ppm)'] = lead_data.groupby('PartialPostalCode')['Lead Amount (ppm)'].transform(lambda grp: grp.interpolate(method='linear'))

years = sorted(lead_data['year'].unique())


full_index = pd.MultiIndex.from_product([postal_coords['PartialPostalCode'].unique(),years], names=['PartialPostalCode','year'])

full_df = pd.DataFrame( index=full_index).reset_index()

lead_full = full_df.merge(lead_data, on=['PartialPostalCode', 'year'], how='left')
lead_full = lead_full.merge(postal_coords,on='PartialPostalCode',how='left' )

lead_full['Lead Amount (ppm)'] =pd.to_numeric(lead_full['Lead Amount (ppm)'], errors='coerce')
lead_full = gpd.GeoDataFrame(lead_full, geometry=gpd.points_from_xy(lead_full['Longitude'], lead_full['Latitude']), crs="EPSG:4326")

pipes = gpd.read_file("distribution-watermain-4326.geojson")

#setting the value limits for the sclae on the map
vmin = 1e-5
vmax = 10e-5

for i in years[-9:]:

    per_year_gdf = lead_full[lead_full['year'] == i].copy()
    if per_year_gdf.empty:continue

    plt.figure(figsize=(12,12))
    pipes.plot(ax=plt.gca(),color='black',linewidth=0.5,alpha=0.3)
    per_year_gdf.plot(ax=plt.gca(), column='Lead Amount (ppm)', markersize=60, cmap='viridis', vmin=vmin, vmax=vmax, legend=True, missing_kwds={"color": "lightgray", "label": "No data"})
    plt.title(f"Toronto Water Network w Lead Contamination ({i})\n(All postal nodes shown)")
    plt.axis('off')
    plt.savefig(f"out/maps/actual/Actual_{i}.png", dpi=300, bbox_inches='tight')
    plt.close()
