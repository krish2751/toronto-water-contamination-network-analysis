import pandas as pd

import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point

from scipy.spatial import cKDTree

# load data
lead_data = pd.read_csv("leadcont.csv")
postal_coords = pd.read_csv("Geospatial_Coordinates.csv")

# clean column names
lead_data.columns = lead_data.columns.str.strip()
postal_coords.columns = postal_coords.columns.str.strip()
# rename to match
postal_coords.rename(columns={'Postal Code': 'PartialPostalCode'}, inplace=True)

# merge coords with lead data
lead_data = lead_data.merge(postal_coords, on='PartialPostalCode', how='left')
lead_data['Sample Date'] = pd.to_datetime(lead_data['Sample Date'], errors='coerce')
lead_data['year'] = lead_data['Sample Date'].dt.year.astype(str)
lead_data['Lead Amount (ppm)'] = pd.to_numeric(lead_data['Lead Amount (ppm)'], errors='coerce')

# fill missing values
lead_data['Lead Amount (ppm)'] = lead_data.groupby('PartialPostalCode')['Lead Amount (ppm)'].transform(lambda grp: grp.interpolate(method='linear'))

# convert to geodataframe
lead_gdf = gpd.GeoDataFrame(lead_data, geometry=gpd.points_from_xy(lead_data['Longitude'], lead_data['Latitude']), crs="EPSG:4326")
pipes = gpd.read_file("distribution-watermain-4326.geojson")

# remove invalid coords - had some NaN that broke things
lead_gdf_filtered = lead_gdf.dropna(subset=['Longitude', 'Latitude'])
lead_gdf_filtered = lead_gdf_filtered[np.isfinite(lead_gdf_filtered['Longitude']) & np.isfinite(lead_gdf_filtered['Latitude'])]

# get unique postal code coords for kd-tree
postal_pnts = np.array([(geom.x, geom.y) for geom in lead_gdf_filtered.drop_duplicates('PartialPostalCode')['geometry']])
postal_codes = lead_gdf_filtered.drop_duplicates('PartialPostalCode')['PartialPostalCode'].tolist()

# build kd-tree for nearest neighbor - way faster
tree = cKDTree(postal_pnts)

def nearest_postal_code(point):
    distnce, index = tree.query([point.x, point.y])
    nearest_code = postal_codes[index]
    return nearest_code

# setup graph
G_postal = nx.Graph()

# add postal codes as nodes
for code, geom in zip(postal_codes, postal_pnts):
    pt = Point(geom)
    G_postal.add_node(code, geometry=pt)

#print( "G_postal created")

for _, row in pipes.iterrows():
    geom = row.geometry
    if geom.geom_type == 'LineString':
        lines = [geom]
    elif geom.geom_type == 'MultiLineString':
        lines = list(geom.geoms)
    else:
        continue
    
    for line in lines:
        coords = list(line.coords)
        # find postal codes at start/end
        start_code = nearest_postal_code(Point(coords[0]))
        end_code = nearest_postal_code(Point(coords[-1]))
        # only add edge if different codes
        if start_code != end_code:
            if not G_postal.has_edge(start_code, end_code):
                G_postal.add_edge(start_code, end_code, diameters=[], materials=[], ages=[])
            age = 2025 - (row.get('Watermain Construction Year', 2000) or 2000)
            # some edges have multiple segments, storing as list
            G_postal[start_code][end_code]['diameters'].append(row.get('Watermain Diameter', 0))
            G_postal[start_code][end_code]['materials'].append(row.get('Watermain Material', 'unknown'))
            G_postal[start_code][end_code]['ages'].append(age)

# using 2025 data for now
year_to_use = '2025'
year_gdf = lead_gdf[lead_gdf['year'] == year_to_use]
# calc avg lead per postal code
lead_avg = year_gdf.groupby('PartialPostalCode')['Lead Amount (ppm)'].mean().to_dict()

# add lead and degree to nodes
for n in G_postal.nodes:
    G_postal.nodes[n]['lead_ppm'] = lead_avg.get(n, 0)  # default 0 if no data
    G_postal.nodes[n]['degree'] = G_postal.degree[n]

# calc edge averages (some have multiple segments)
for u, v, data in G_postal.edges(data=True):
    if data['diameters']:
        data['diameter_avg'] = np.mean(data['diameters'])
    else:
        data['diameter_avg'] = 0
    
    if data['ages']:
        data['age_avg'] = np.mean(data['ages'])
    else:
        data['age_avg'] = 0

G =G_postal
# print(6)
pred_dict = lead_avg
