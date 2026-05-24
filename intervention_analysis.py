import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

lead_data = pd.read_csv("leadcont.csv")
postal_coords = pd.read_csv("Geospatial_Coordinates.csv")
lead_data.columns = lead_data.columns.str.strip()
postal_coords.columns = postal_coords.columns.str.strip()
postal_coords.rename(columns={'Postal Code': 'PartialPostalCode'}, inplace=True)

lead_data = lead_data.merge(postal_coords, on='PartialPostalCode', how='left')
lead_data['Sample Date'] = pd.to_datetime(lead_data['Sample Date'], errors='coerce')
lead_data['year'] = lead_data['Sample Date'].dt.year
lead_data['Lead Amount (ppm)'] = pd.to_numeric(lead_data['Lead Amount (ppm)'], errors='coerce')
lead_data['Lead Amount (ppm)'] = lead_data.groupby('PartialPostalCode')['Lead Amount (ppm)'].transform(lambda grp: grp.interpolate(method='linear'))

lead_gdf = gpd.GeoDataFrame(lead_data, geometry=gpd.points_from_xy(lead_data['Longitude'], lead_data['Latitude']), crs="EPSG:4326")
pipes = gpd.read_file("distribution-watermain-4326.geojson")

lead_gdf_filtered = lead_gdf.dropna(subset=['Longitude', 'Latitude'])
postal_points = np.array([(geom.x, geom.y) for geom in lead_gdf_filtered.drop_duplicates('PartialPostalCode')['geometry']])
postal_codes = lead_gdf_filtered.drop_duplicates('PartialPostalCode')['PartialPostalCode'].tolist()
tree = cKDTree(postal_points)

def nearest_postal_code(point):
    _, index = tree.query([point.x, point.y])
    return postal_codes[index]

# nodes = postal codes, edges = pipes
# creating graph of postal codes and pipes
# network structure for GAT
def build_postal_graph(year, pipes_data=pipes):
    G = nx.Graph()
    for code, geom in zip(postal_codes, postal_points):
        G.add_node(code, geometry=Point(geom))
    for _, row in pipes_data.iterrows():
        geom = row.geometry
        if geom is None: continue
        lines = [geom] if geom.geom_type=='LineString' else list(geom.geoms) if geom.geom_type=='MultiLineString' else []
        for line in lines:
            coords = list(line.coords)
            start_code = nearest_postal_code(Point(coords[0]))
            end_code = nearest_postal_code(Point(coords[-1]))
            if start_code != end_code:
                if not G.has_edge(start_code, end_code):
                    G.add_edge(start_code, end_code, diameters=[], materials=[], ages=[])
                age = year - (row.get('Watermain Construction Year',2000) or 2000)
                G[start_code][end_code]['diameters'].append(row.get('Watermain Diameter',0))
                G[start_code][end_code]['materials'].append(str(row.get('Watermain Material','UNKNOWN')).upper())
                G[start_code][end_code]['ages'].append(age)
    return G

material_factor = {'LEAD':1.0,'CI':0.6,'CICL':0.6,'DIP':0.3,'COPPER':0.2,'PVC':0.1,'UNKNOWN':0.4}

# assign edge weights from material and diameter
# material: lead=1.0, CI=0.6, CICL=0.6, DIP=0.3, COPPER=0.2, PVC=0.1, UNKNOWN=0.4
def assign_edge_weights(G):
    for u,v,d in G.edges(data=True):
        mats = d.get('materials',[])
        dias = d.get('diameters',[])
        mat_factor = np.mean([material_factor.get(m,0.4) for m in mats]) if mats else 0.4
        avg_dia = np.mean(dias) if dias else 200
        G[u][v]['weight'] = mat_factor*(avg_dia/100.0)
    return G

# GAT model for lead prediction
# 2 layer GAT
# layer 1: input features + edge index
# layer 2: layer 1 output + edge index
# output: predicted lead conc
class GAT(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats=8, out_feats=1, heads=4):
        super().__init__()


        self.conv1 = GATConv(in_feats, hidden_feats, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_feats * heads, out_feats, heads=1, concat=False, dropout=0.6)
    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index))
        return self.conv2(x, data.edge_index)



def pipe_replacement_after_intervention(pipes_df, replacement_type='high_risk_materials', age_threshold=50, target_material='PVC'):

    pipes_modified = pipes_df.copy()
    # high risk materials first
    if replacement_type == 'high_risk_materials':
        # replace CI, CICL, DIP, DICL, UNKNOWN
        mask = pipes_modified['Watermain Material'].str.upper().isin(['CI', 'CICL', 'DIP', 'DICL', 'UNKNOWN', 'UNK'])
        pipes_modified.loc[mask, 'Watermain Material'] = target_material
        pipes_modified.loc[mask, 'Watermain Construction Year'] = 2020

    # age targeted
    elif replacement_type == 'age_targeted':
       
        pipe_ages = 2025 - pd.to_numeric(pipes_modified['Watermain Construction Year'], errors='coerce').fillna(2000)
        mask = pipe_ages >= age_threshold
        pipes_modified.loc[mask, 'Watermain Material'] = target_material
        pipes_modified.loc[mask, 'Watermain Construction Year'] = 2020
    return pipes_modified


def apply_node_fix_intervention(lead_train_dict, nodes_to_fix, reduction_factor=0.1):
    lead_mod = lead_train_dict.copy()  # just copy it
    for node in nodes_to_fix:
        if node in lead_mod:
            lead_mod[node] *= reduction_factor
    return lead_mod

def predicting_with_gat(G, lead_train_avg, lead_target_avg, device='cpu'):
    node_list = list(G.nodes)
    edges = []   # edge index
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    for u, v in G.edges:
        edges.extend([[node_to_idx[u], node_to_idx[v]], [node_to_idx[v], node_to_idx[u]]])
    if len(edges) == 0: return None, node_list
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    lead_train_vals = np.array([lead_train_avg.get(n, 0.0) for n in node_list])
    degree_vals = np.array([G.degree[n] for n in node_list])
    lead_target_vals = np.array([lead_target_avg.get(n, np.nan) for n in node_list])
    
    lead_min, lead_max = np.nanmin(lead_train_vals), np.nanmax(lead_train_vals)
    if lead_max - lead_min < 1e-8: return None, node_list
    
    lead_scaled = (lead_train_vals - lead_min) / (lead_max - lead_min + 1e-8)
    degree_scaled = (degree_vals - degree_vals.min()) / (degree_vals.max() - degree_vals.min() + 1e-8)
    x = torch.tensor(np.stack([lead_scaled, degree_scaled], axis=1), dtype=torch.float)
    y_target = torch.tensor(np.nan_to_num((lead_target_vals - lead_min) / (lead_max - lead_min + 1e-8)), dtype=torch.float, device=device)
    
    data = Data(x=x.to(device), edge_index=edge_index.to(device))
    model = GAT(in_feats=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    
    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        out = model(data).squeeze()
        F.mse_loss(out, y_target).backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        predicted_scaled = model(data).squeeze().cpu().numpy()
    
    return predicted_scaled*(lead_max - lead_min) + lead_min, node_list


def identify_pipes_high_risk_materials(pipes_df, target_percentage=0.14, high_risk_postal_codes=None):
    total_pipes = len(pipes_df)
    target_count = int(total_pipes * target_percentage)
    # filter high risk materials
    candidates = pipes_df[pipes_df['Watermain Material'].str.upper().isin(['CI', 'CICL', 'DIP', 'DICL', 'UNKNOWN', 'UNK'])].copy()
    
    if len(candidates) == 0: return gpd.GeoDataFrame(geometry=[], crs=pipes_df.crs)
    
    # calc risk score
    candidates['risk_score'] = 0.0
    candidates['material_risk'] = candidates['Watermain Material'].str.upper().map(material_factor).fillna(0.4)
    candidates['risk_score'] += candidates['material_risk'] * 5.0
    
    candidates['age'] = 2025 - pd.to_numeric(candidates['Watermain Construction Year'], errors='coerce').fillna(2000)
    candidates['age_risk'] = (candidates['age'] / 100.0).clip(0, 1)
    candidates['risk_score'] += candidates['age_risk'] * 2.0
    
    candidates['diameter'] = pd.to_numeric(candidates['Watermain Diameter'], errors='coerce').fillna(200)
    candidates['diameter_risk'] = 1.0 - (candidates['diameter'] / 500.0).clip(0, 1)
    candidates['risk_score'] += candidates['diameter_risk'] * 1.0
    
    # bonus for pipes near high risk areas
    if high_risk_postal_codes and len(high_risk_postal_codes) > 0:
        candidates['near_high_risk'] = False
        for idx in candidates.index:
            geom = candidates.loc[idx, 'geometry']
            if geom is None:
                continue
            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms) if geom.geom_type == 'MultiLineString' else []
            for line in lines:
                coords = list(line.coords)
                start_code = nearest_postal_code(Point(coords[0]))
                end_code = nearest_postal_code(Point(coords[-1]))
                if start_code in high_risk_postal_codes or end_code in high_risk_postal_codes:
                    candidates.loc[idx, 'near_high_risk'] = True
                    break
        candidates.loc[candidates['near_high_risk'], 'risk_score'] += 0.5
    

    pipes_to_replace = candidates.sort_values('risk_score', ascending=False).head(min(target_count, len(candidates))).copy()
    cols_to_keep = ['geometry'] + [c for c in ['Watermain Material', 'Watermain Construction Year', 'Watermain Diameter', 'risk_score'] if c in pipes_to_replace.columns]
    pipes_to_replace = pipes_to_replace[cols_to_keep]
    return pipes_to_replace

def identify_pipes_age_targeted(pipes_df, age_threshold=50, target_percentage=0.12, high_risk_postal_codes=None):
    total_pipes = len(pipes_df)
    target_count = int(total_pipes * target_percentage)
    # old pipes only
    pipe_ages = 2025 - pd.to_numeric(pipes_df['Watermain Construction Year'], errors='coerce').fillna(2000)
    candidates = pipes_df[pipe_ages >= age_threshold].copy()
    
    if len(candidates) == 0: return gpd.GeoDataFrame(geometry=[], crs=pipes_df.crs)
    
    # risk score - age is main thing here
    candidates['risk_score'] = 0.0
    candidates['age'] = 2025 - pd.to_numeric(candidates['Watermain Construction Year'], errors='coerce').fillna(2000)
    candidates['age_risk'] = (candidates['age'] / 100.0).clip(0, 1)
    candidates['risk_score'] += candidates['age_risk'] * 5.0
    
    candidates['material_risk'] = candidates['Watermain Material'].str.upper().map(material_factor).fillna(0.4)
    candidates['risk_score'] += candidates['material_risk'] * 3.0
    
    candidates['diameter'] = pd.to_numeric(candidates['Watermain Diameter'], errors='coerce').fillna(200)
    candidates['diameter_risk'] = 1.0 - (candidates['diameter'] / 500.0).clip(0, 1)
    candidates['risk_score'] += candidates['diameter_risk'] * 1.0
    
    if high_risk_postal_codes and len(high_risk_postal_codes) > 0:
        candidates['near_high_risk'] = False
        for idx in candidates.index:
            geom = candidates.loc[idx, 'geometry']
            if geom is None:
                continue
            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms) if geom.geom_type == 'MultiLineString' else []
            for line in lines:
                coords = list(line.coords)
                start_code = nearest_postal_code(Point(coords[0]))
                end_code = nearest_postal_code(Point(coords[-1]))
                if start_code in high_risk_postal_codes or end_code in high_risk_postal_codes:
                    candidates.loc[idx, 'near_high_risk'] = True
                    break
        candidates.loc[candidates['near_high_risk'], 'risk_score'] += 0.5
    
    # get top pipes
    pipes_to_replace = candidates.sort_values('risk_score', ascending=False).head(min(target_count, len(candidates))).copy()
    cols_to_keep = ['geometry'] + [c for c in ['Watermain Material', 'Watermain Construction Year', 'Watermain Diameter', 'risk_score'] if c in pipes_to_replace.columns]
    pipes_to_replace = pipes_to_replace[cols_to_keep]
    return pipes_to_replace

def identify_pipes_material_weighted(pipes_df, target_percentage=0.10, high_risk_postal_codes=None):
    total_pipes = len(pipes_df)
    target_count = int(total_pipes * target_percentage)
    candidates = pipes_df.copy()
    
    if len(candidates) == 0: return gpd.GeoDataFrame(geometry=[], crs=pipes_df.crs)
    
    # material weighted risk
    candidates['risk_score'] = 0.0
    candidates['material_risk'] = candidates['Watermain Material'].str.upper().map(material_factor).fillna(0.4)
    # material is main thing
    candidates['risk_score'] += candidates['material_risk'] * 5.0
    
    # age second
    candidates['age'] = 2025 - pd.to_numeric(candidates['Watermain Construction Year'], errors='coerce').fillna(2000)
    candidates['age_risk'] = (candidates['age'] / 100.0).clip(0, 1)
    candidates['risk_score'] += candidates['age_risk'] * 2.0
    
    # diameter third (smaller = worse)
    candidates['diameter'] = pd.to_numeric(candidates['Watermain Diameter'], errors='coerce').fillna(200)
    candidates['diameter_risk'] = 1.0 - (candidates['diameter'] / 500.0).clip(0, 1)
    candidates['risk_score'] += candidates['diameter_risk'] * 1.0
    
    # bonus if near high risk codes
    if high_risk_postal_codes and len(high_risk_postal_codes) > 0:
        candidates['near_high_risk'] = False
        for idx in candidates.index:
            geom = candidates.loc[idx, 'geometry']
            if geom is None:
                continue
            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms) if geom.geom_type == 'MultiLineString' else []
            for i in lines:
                coords = list(i.coords)
                start_code = nearest_postal_code(Point(coords[0]))
                end_code = nearest_postal_code(Point(coords[-1]))
                if start_code in high_risk_postal_codes or end_code in high_risk_postal_codes:
                    candidates.loc[idx, 'near_high_risk'] = True
                    break
        candidates.loc[candidates['near_high_risk'], 'risk_score'] += 0.5
    
    # top pipes by score
    pipes_to_replace = candidates.sort_values('risk_score', ascending=False).head(target_count).copy()
    cols_to_keep = ['geometry'] + [c for c in ['Watermain Material', 'Watermain Construction Year', 'Watermain Diameter', 'risk_score'] if c in pipes_to_replace.columns]
    pipes_to_replace = pipes_to_replace[cols_to_keep]
    return pipes_to_replace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
target_year = 2025
train_years = [2022, 2023, 2024]

train_gdf = lead_gdf[lead_gdf['year'].isin(train_years)]
target_gdf = lead_gdf[lead_gdf['year'] == target_year]
lead_train_avg = train_gdf.groupby('PartialPostalCode')['Lead Amount (ppm)'].mean().to_dict()
lead_target_avg = target_gdf.groupby('PartialPostalCode')['Lead Amount (ppm)'].mean().to_dict()

G_baseline = assign_edge_weights(build_postal_graph(target_year, pipes))
pred_baseline, node_list_baseline = predicting_with_gat(G_baseline, lead_train_avg, lead_target_avg, device)
if pred_baseline is None: exit(1)

baseline_dict = dict(zip(node_list_baseline, pred_baseline))

# Int 1: high risk materials
high_risk_postal_codes = [code for code, _ in sorted(baseline_dict.items(), key=lambda x: x[1], reverse=True)[:10]]
pipes_high_risk_materials_selected = identify_pipes_high_risk_materials(pipes, target_percentage=0.14, high_risk_postal_codes=high_risk_postal_codes)
pipes_high_risk_materials = pipes.copy()
if len(pipes_high_risk_materials_selected) > 0:
    selected_geoms = set(geom.wkt for geom in pipes_high_risk_materials_selected.geometry)
    for idx in pipes_high_risk_materials.index:
        if pipes_high_risk_materials.loc[idx, 'geometry'].wkt in selected_geoms:
            pipes_high_risk_materials.loc[idx, 'Watermain Material'] = 'PVC'
            pipes_high_risk_materials.loc[idx, 'Watermain Construction Year'] = 2020
G_high_risk_materials = assign_edge_weights(build_postal_graph(target_year, pipes_high_risk_materials))
pred_high_risk_materials, node_list_high_risk = predicting_with_gat(G_high_risk_materials, lead_train_avg, lead_target_avg, device)
high_risk_materials_dict = dict(zip(node_list_high_risk, pred_high_risk_materials)) if pred_high_risk_materials is not None else {}

# Int 2: age targeted
pipes_age_targeted_selected = identify_pipes_age_targeted(pipes, age_threshold=50, target_percentage=0.12, high_risk_postal_codes=high_risk_postal_codes)
pipes_age_targeted = pipes.copy()
if len(pipes_age_targeted_selected) > 0:
    selected_geoms = set(geom.wkt for geom in pipes_age_targeted_selected.geometry)
    for idx in pipes_age_targeted.index:
        if pipes_age_targeted.loc[idx, 'geometry'].wkt in selected_geoms:
            pipes_age_targeted.loc[idx, 'Watermain Material'] = 'PVC'
            pipes_age_targeted.loc[idx, 'Watermain Construction Year'] = 2020
G_age_targeted = assign_edge_weights(build_postal_graph(target_year, pipes_age_targeted))
pred_age_targeted, node_list_age = predicting_with_gat(G_age_targeted, lead_train_avg, lead_target_avg, device)
age_targeted_dict = dict(zip(node_list_age, pred_age_targeted)) if pred_age_targeted is not None else {}

# Int 3: node fix at hotspots
# top 10 risk codes
top_risk_postal_codes = [n[0] for n in sorted(baseline_dict.items(), key=lambda x: x[1], reverse=True)[:10]]
lead_train_fixed = apply_node_fix_intervention(lead_train_avg, top_risk_postal_codes, reduction_factor=0.1)
lead_target_fixed = apply_node_fix_intervention(lead_target_avg, top_risk_postal_codes, reduction_factor=0.1)
pred_fixed, node_list_fixed = predicting_with_gat(G_baseline, lead_train_fixed, lead_target_fixed, device)
if pred_fixed is not None:
    node_fix_dict_temp = dict(zip(node_list_fixed, pred_fixed))
    non_hotspot_values = [node_fix_dict_temp.get(code, 0) for code in node_list_fixed if code not in top_risk_postal_codes]
    non_hotspot_values = [v for v in non_hotspot_values if v > 0]  # filter zeros
    if len(non_hotspot_values) > 0:
        min_non_hotspot = min(non_hotspot_values)
        bottom_value = min_non_hotspot * 0.1
        for code in top_risk_postal_codes:
            if code in node_fix_dict_temp:
                node_fix_dict_temp[code] = bottom_value
    node_fix_dict = node_fix_dict_temp
else:
    node_fix_dict = {}

pipes_material_weighted = identify_pipes_material_weighted(pipes, target_percentage=0.10, high_risk_postal_codes=high_risk_postal_codes)
pipes_with_material_replacement = pipes.copy()
if len(pipes_material_weighted)>0:
    selected_geoms = set()
    for geom in pipes_material_weighted.geometry:
        selected_geoms.add(geom.wkt)
    for idx in pipes_with_material_replacement.index:
        if pipes_with_material_replacement.loc[idx, 'geometry'].wkt in selected_geoms:
            pipes_with_material_replacement.loc[idx, 'Watermain Material'] = 'PVC'
            pipes_with_material_replacement.loc[idx, 'Watermain Construction Year'] = 2020
G_material_weighted = assign_edge_weights(build_postal_graph(target_year, pipes_with_material_replacement))
pred_material_weighted, node_list_material = predicting_with_gat(G_material_weighted, lead_train_avg, lead_target_avg, device)
material_weighted_dict = dict(zip(node_list_material, pred_material_weighted)) if pred_material_weighted is not None else {}

postal_coords_clean = postal_coords.copy()
postal_coords_clean['PartialPostalCode'] = postal_coords_clean['PartialPostalCode'].str.strip().str.upper()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# pipes to replace for viz
pipes_high_risk_materials_replace = pipes[pipes['Watermain Material'].str.upper().isin(['CI', 'CICL', 'DIP', 'DICL', 'UNKNOWN', 'UNK'])]
pipe_ages_viz = 2025 - pd.to_numeric(pipes['Watermain Construction Year'], errors='coerce').fillna(2000)
pipes_age_targeted_replace = pipes[pipe_ages_viz >= 55]

scenarios = [('Baseline (Current State)', baseline_dict, None, 0), ('1. Replace High-Risk Materials\n(CI, CICL, DIP, DICL, UNK→PVC)', high_risk_materials_dict if high_risk_materials_dict else baseline_dict, pipes_high_risk_materials_selected, 1), ('2. Age-Targeted Renewal\n(Replace Pipes >50 Years)', age_targeted_dict if age_targeted_dict else baseline_dict, pipes_age_targeted_selected, 2), ('3. Node-Fix at Hotspots\n(Top 10 High-Risk Areas)', node_fix_dict if node_fix_dict else baseline_dict, None, 3)]

# 4th image
scenarios_with_material = [('Baseline (Current State)', baseline_dict, None), ('4. Material-Weighted Priority', material_weighted_dict if material_weighted_dict else baseline_dict, pipes_material_weighted)]


for scenario_name, pred_dict, pipes_to_replace, index in scenarios:
    ax = axes[index]
    postal_gdf = postal_coords_clean[postal_coords_clean['PartialPostalCode'].isin(pred_dict.keys())].copy()
    postal_gdf['lead_pred'] = postal_gdf['PartialPostalCode'].map(pred_dict).fillna(0)
    postal_gdf['lead_scaled'] = postal_gdf['lead_pred'] * 1e5
    postal_gdf = gpd.GeoDataFrame(postal_gdf, geometry=gpd.points_from_xy(postal_gdf['Longitude'], postal_gdf['Latitude']), crs="EPSG:4326")
    postal_gdf['lead_clipped'] = postal_gdf['lead_scaled'].clip(upper=10)
    
    pipes.plot(ax=ax, color='gray', linewidth=0.2, alpha=0.15, zorder=1)
    if pipes_to_replace is not None and len(pipes_to_replace) > 0:
        pipes_to_replace.plot(ax=ax, color='red', linewidth=1.5, alpha=0.7, zorder=2)
    postal_gdf.plot(ax=ax, column='lead_clipped', legend=True, markersize=50, vmin=0, vmax=10, cmap='viridis', zorder=3)
    title = f"{scenario_name}\nMean: {postal_gdf['lead_pred'].mean():.8f} ppm"
    if pipes_to_replace is not None and len(pipes_to_replace) > 0:
        title += f"\n{len(pipes_to_replace)} pipes to replace"
    ax.set_title(title, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig(f"out/analysis/intervention/Intervention_Analysis_{target_year}.png", dpi=300, bbox_inches='tight')
plt.close()

# material weighted figure
fig_mat, axes_mat = plt.subplots(1, 2, figsize=(16, 8))
for idx, (scenario_name, pred_dict, pipes_to_replace) in enumerate(scenarios_with_material):
    ax = axes_mat[idx]
    postal_gdf = postal_coords_clean[postal_coords_clean['PartialPostalCode'].isin(pred_dict.keys())].copy()
    postal_gdf['lead_pred'] = postal_gdf['PartialPostalCode'].map(pred_dict).fillna(0)
    postal_gdf['lead_scaled'] = postal_gdf['lead_pred'] * 1e5
    postal_gdf = gpd.GeoDataFrame(postal_gdf, geometry=gpd.points_from_xy(postal_gdf['Longitude'], postal_gdf['Latitude']), crs="EPSG:4326")
    postal_gdf['lead_clipped'] = postal_gdf['lead_scaled'].clip(upper=10)
    
    pipes.plot(ax=ax, color='gray', linewidth=0.2, alpha=0.15, zorder=1)
    if pipes_to_replace is not None and len(pipes_to_replace) > 0:
        pipes_to_replace.plot(ax=ax, color='red', linewidth=1.5, alpha=0.7, zorder=2)
    postal_gdf.plot(ax=ax, column='lead_clipped', legend=True, markersize=50, vmin=0, vmax=10, cmap='viridis', zorder=3)
    title = f"{scenario_name}\nMean: {postal_gdf['lead_pred'].mean():.8f} ppm"
    if pipes_to_replace is not None and len(pipes_to_replace) > 0:
        title += f"\n{len(pipes_to_replace)} pipes to replace"
    ax.set_title(title, fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.savefig(f"out/analysis/intervention/Intervention_MaterialWeighted_{target_year}.png", dpi=300, bbox_inches='tight')
plt.close()

fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
scenario_names = ['Baseline', 'High-Risk\nMaterials', 'Age-Targeted\nRenewal', 'Node-Fix\nHotspots', 'Material-\nWeighted']
mean_values = [np.array(list(baseline_dict.values())).mean(), np.array(list(high_risk_materials_dict.values())).mean() if high_risk_materials_dict else 0, np.array(list(age_targeted_dict.values())).mean() if age_targeted_dict else 0, np.array(list(node_fix_dict.values())).mean() if node_fix_dict else 0, np.array(list(material_weighted_dict.values())).mean() if material_weighted_dict else 0]

bars = ax_bar.bar(scenario_names, mean_values, color=['red' if i == 0 else 'green' for i in range(len(scenario_names))], alpha=0.7)
ax_bar.set_ylabel('Mean Predicted Lead (ppm)', fontsize=12)
ax_bar.set_title(f'Intervention Comparison - {target_year}', fontsize=14, fontweight='bold')
ax_bar.grid(True, axis='y', alpha=0.3)
for bar, val in zip(bars, mean_values):
    if val > 0:
        ax_bar.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:.2e}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(f"out/analysis/intervention/Intervention_Comparison_Chart_{target_year}.png", dpi=300, bbox_inches='tight')
plt.close()

fig_pipes, axes_pipes = plt.subplots(2, 2, figsize=(18, 18))
axes_pipes = axes_pipes.flatten()

# count pipes
pipes_high_risk_count = len(pipes_high_risk_materials_selected)
pipes_age_count = len(pipes_age_targeted_selected)
pipes_material_count = len(pipes_material_weighted)

pipe_interventions = [('High-Risk Materials Replacement', pipes_high_risk_materials_selected, f'CI, CICL, DIP, DICL, UNK pipes', f'Selected: {pipes_high_risk_count} pipes'), ('Age-Targeted Renewal', pipes_age_targeted_selected, 'Pipes older than 50 years', f'Selected: {pipes_age_count} pipes'), ('Node-Fix Hotspots', None, 'Top 10 high-risk postal codes', '10 nodes targeted'), ('Material-Weighted Priority', pipes_material_weighted, 'Top 10% by material risk score', f'Selected: {pipes_material_count} pipes')]

# visualize pipe replacements
for idx, (title, pipes_replace, description, count_label) in enumerate(pipe_interventions):
    ax = axes_pipes[idx]
    pipes.plot(ax=ax, color='lightgray', linewidth=0.3, alpha=0.3, zorder=1)
    if pipes_replace is not None and len(pipes_replace) > 0:
        pipes_replace.plot(ax=ax, color='red', linewidth=2.0, alpha=0.8, zorder=2)
    postal_gdf_all = gpd.GeoDataFrame(postal_coords_clean, geometry=gpd.points_from_xy(postal_coords_clean['Longitude'], postal_coords_clean['Latitude']), crs="EPSG:4326")
    postal_gdf_all.plot(ax=ax, color='blue', markersize=15, alpha=0.4, zorder=3)
    
    # highlight hotspots
    if idx == 2:  # Node-Fix Hotspots
        hotspot_gdf = postal_gdf_all[postal_gdf_all['PartialPostalCode'].isin(top_risk_postal_codes)]
        if len(hotspot_gdf) > 0:
            hotspot_gdf.plot(ax=ax, color='red', markersize=100, alpha=0.6, zorder=4, edgecolor='black', linewidth=2)
    
    ax.set_title(f"{title}\n{description}\n{count_label}", fontsize=11, fontweight='bold')
    ax.axis('off')
    
    if pipes_replace is not None and len(pipes_replace) > 0:
        legend_elements = [Patch(facecolor='red', edgecolor='red', alpha=0.8, label=f'Pipes to Replace ({len(pipes_replace)})'), Patch(facecolor='lightgray', edgecolor='lightgray', alpha=0.3, label='Existing Pipes'), Patch(facecolor='blue', edgecolor='blue', alpha=0.4, label='Postal Codes')]
    else:
        legend_elements = [Patch(facecolor='lightgray', edgecolor='lightgray', alpha=0.3, label='Existing Pipes'), Patch(facecolor='blue', edgecolor='blue', alpha=0.4, label='Postal Codes')]
        if idx == 2:  # hotspot marker
            legend_elements.append(Patch(facecolor='red', edgecolor='black', alpha=0.6, label='Hotspot Nodes'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.suptitle(f'Pipe Replacement Interventions for {target_year}\n(Red lines indicate pipes requiring replacement)', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(f"out/analysis/intervention/Pipe_Replacement_Map_{target_year}.png", dpi=300, bbox_inches='tight')
plt.close(fig_pipes)

# summary stats
scenario_names_list = ['Baseline', 'Replace High-Risk Materials (CI, CICL, DIP, DICL, UNK)', 'Age-Targeted Renewal (>50 Years)', 'Node-Fix at Hotspots (Top 10 Areas)', 'Material-Weighted Priority']

baseline_mean_val = np.array(list(baseline_dict.values())).mean()
high_risk_materials_mean = np.array(list(high_risk_materials_dict.values())).mean() if high_risk_materials_dict else np.nan
age_targeted_mean = np.array(list(age_targeted_dict.values())).mean() if age_targeted_dict else np.nan
node_fix_mean = np.array(list(node_fix_dict.values())).mean() if node_fix_dict else np.nan
material_weighted_mean = np.array(list(material_weighted_dict.values())).mean() if material_weighted_dict else np.nan

mean_vals_list = [baseline_mean_val, high_risk_materials_mean, age_targeted_mean, node_fix_mean, material_weighted_mean]

baseline_max_val = np.array(list(baseline_dict.values())).max()
high_risk_materials_max = np.array(list(high_risk_materials_dict.values())).max() if high_risk_materials_dict else np.nan
age_targeted_max = np.array(list(age_targeted_dict.values())).max() if age_targeted_dict else np.nan
node_fix_max = np.array(list(node_fix_dict.values())).max() if node_fix_dict else np.nan
material_weighted_max = np.array(list(material_weighted_dict.values())).max() if material_weighted_dict else np.nan

max_vals_list = [baseline_max_val, high_risk_materials_max, age_targeted_max, node_fix_max, material_weighted_max]

summary_data = {
    'Scenario': scenario_names_list,
    'Mean_Lead_ppm': mean_vals_list,
    'Max_Lead_ppm': max_vals_list
}

summary_df = pd.DataFrame(summary_data)
baseline_mean = summary_df.loc[0, 'Mean_Lead_ppm']
summary_df['Reduction_%'] = ((baseline_mean - summary_df['Mean_Lead_ppm']) / baseline_mean * 100).round(2).fillna(0)
summary_df.to_csv(f"out/analysis/intervention/Intervention_Summary_{target_year}.csv", index=False)

all_nodes = set(baseline_dict.keys())
for d in [high_risk_materials_dict, age_targeted_dict, node_fix_dict, material_weighted_dict]:
    if d: all_nodes.update(d.keys())

node_comparison = []
for n in all_nodes:
    row = {'PostalCode': n, 'Baseline': baseline_dict.get(n, np.nan)}
    row['HighRisk_Materials'] = high_risk_materials_dict.get(n, np.nan) if high_risk_materials_dict else np.nan
    row['Age_Targeted'] = age_targeted_dict.get(n, np.nan) if age_targeted_dict else np.nan
    row['Node_Fix'] = node_fix_dict.get(n, np.nan) if node_fix_dict else np.nan
    row['Material_Weighted'] = material_weighted_dict.get(n, np.nan) if material_weighted_dict else np.nan
    if row['Baseline'] > 0:
        for col in ['HighRisk_Materials', 'Age_Targeted', 'Node_Fix', 'Material_Weighted']:
            if not np.isnan(row[col]):
                row[f'Improvement_{col}_%'] = ((row['Baseline'] - row[col]) / row['Baseline'] * 100)
    node_comparison.append(row)

pd.DataFrame(node_comparison).to_csv(f"out/analysis/intervention/Intervention_NodeComparison_{target_year}.csv", index=False)
