import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import os
import dem2

# Load predictions for all years
predictions_csv = "out/predictions/gat_predictions_by_year.csv"
if os.path.exists(predictions_csv):
    preds_df = pd.read_csv(predictions_csv)
    preds_df['PartialPostalCode'] = preds_df['PartialPostalCode'].str.strip().str.upper()
    preds_df['year'] = pd.to_numeric(preds_df['year'], errors='coerce')
else:
    preds_df = pd.DataFrame()

years_from_2018_to_2027 = range(2018, 2027)
for y in years_from_2018_to_2027:
    # Get predict   s for this specific year
    if not preds_df.empty:
        year_preds = preds_df[preds_df['year'] == y]
        pred_dict_year = dict(zip(year_preds['PartialPostalCode'].str.strip().str.upper(), year_preds['y_pred_ppm']))
    else:
        pred_dict_year = {}
    
    edges_data_by_year = [] # this is used to store the data for each year.
    for u, v, d in dem2.G.edges(data=True):
        u_clean = str(u).strip().upper()
        v_clean = str(v).strip().upper()
        lead_u = pred_dict_year.get(u_clean, np.nan)
        lead_v = pred_dict_year.get(v_clean, np.nan)
        avg_lead = np.nanmean([lead_u, lead_v]) # average lead concentration for the edge

        #recalculate pipe age for the current year
        #og age is the age of the pipe in 2025 - construction year
        # so construction year = 2025 - original age
        # current age = year - construction year = year - (2025 - original age) = year - 2025 + original age
        original_ages = d.get('ages', [])
        if original_ages and not all(np.isnan([a for a in original_ages if a is not None])):
            current_ages = [y - 2025 + age for age in original_ages if not np.isnan(age)]
            pipe_age = np.nanmean(current_ages) if current_ages else np.nan
        else:
            pipe_age = np.nan

        # get the average pipe diameter for the edge
        pipe_dia = np.nanmean(d.get('diameters', [np.nan]))

        # get the pipe material for the edge
        mats = sorted(set(d.get('materials', ['UNKNOWN'])))
        pipe_mat = mats[0] if mats else 'UNKNOWN'
        edge_name = '-'.join(sorted([u,v]))
        edges_data_by_year.append({'edge': edge_name, 'avg_lead_ppm': avg_lead, 'pipe_age': pipe_age, 'pipe_diameter': pipe_dia, 'pipe_material': pipe_mat})

    edge_df = pd.DataFrame(edges_data_by_year )
    edge_csv = f"out/analysis/pipes/Pipe_Analysis_{y}.csv"
    edge_df.to_csv(edge_csv, index=False)

    # pipe age vs predicted lead
    plt.figure(figsize=(8,6))
    plot_df = edge_df[['pipe_age', 'avg_lead_ppm']].dropna()
    plt.scatter(plot_df['pipe_age'], plot_df['avg_lead_ppm'], alpha=0.6)
    plt.xlabel("Pipe Age (years)")
    plt.ylabel("Average Predicted Lead (ppm)")
    plt.title(f"Pipe Age vs Predicted Lead ({y})")
    plt.grid(True)
    plt.savefig(f"out/analysis/pipes/PipeAge_vs_Lead_{y}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # pipe diameter vs predicted lead
    plt.figure(figsize=(8,6))
    plot_df2 = edge_df[['pipe_diameter', 'avg_lead_ppm']].dropna()
    plt.scatter(plot_df2['pipe_diameter'], plot_df2['avg_lead_ppm'], alpha=0.6, color='orange')
    plt.xlabel("Pipe Diameter (mm)")
    plt.ylabel("Average Predicted Lead (ppm)")
    plt.title(f"Pipe Diameter vs Predicted Lead ({y})")
    plt.grid(True)
    plt.savefig(f"out/analysis/pipes/PipeDia_vs_Lead_{y}.png", dpi=300, bbox_inches='tight')
    plt.close()

    #plot for pipe material vs predicted lead
    plt.figure(figsize=(10,6))
    pipe_materials = sorted(edge_df['pipe_material'].dropna().unique())
    box_df = [edge_df.loc[edge_df['pipe_material']==mat, 'avg_lead_ppm'].dropna() for mat in pipe_materials]
    plt.boxplot(box_df, tick_labels=pipe_materials)
    plt.xticks(rotation=45)
    plt.ylabel("Average Predicted Lead (ppm)")
    plt.title(f"Boxplot of Predicted Lead vs Pipe Material ({y})", fontsize=12, fontweight='bold')
    plt.grid(True, axis='y')
    plt.savefig(f"out/analysis/pipes/PipeMat_vs_Lead_{y}.png", dpi=300, bbox_inches='tight')
    plt.close()
