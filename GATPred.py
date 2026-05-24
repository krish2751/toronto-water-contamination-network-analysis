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

# loading the lead contamination data
lead_data = pd.read_csv("leadcont.csv")
# getting postal code coordinates
postal_coords = pd.read_csv("Geospatial_Coordinates.csv")
# cleaning up column names, removing whitespace
lead_data.columns = lead_data.columns.str.strip()
postal_coords.columns = postal_coords.columns.str.strip()
# renaming to match the other dataframe
postal_coords.rename(columns={'Postal Code': 'PartialPostalCode'}, inplace=True)

# merging the datasets together
lead_data = lead_data.merge(postal_coords, on='PartialPostalCode', how='left')
# converting dates and extracting year
lead_data['Sample Date'] = pd.to_datetime(lead_data['Sample Date'], errors='coerce')
lead_data['year'] = lead_data['Sample Date'].dt.year
# converting lead amounts to numeric and interpolating missing values
lead_data['Lead Amount (ppm)'] = pd.to_numeric(lead_data['Lead Amount (ppm)'], errors='coerce')
lead_data['Lead Amount (ppm)'] = lead_data.groupby('PartialPostalCode')['Lead Amount (ppm)'].transform(lambda grp: grp.interpolate(method='linear'))

# creating geodataframe with point geometries
lead_gdf = gpd.GeoDataFrame(lead_data, geometry=gpd.points_from_xy(lead_data['Longitude'], lead_data['Latitude']), crs="EPSG:4326")
# loading the water pipe network data
pipes = gpd.read_file("distribution-watermain-4326.geojson")

# filtering out rows without coordinates
lead_gdf_filtered = lead_gdf.dropna(subset=['Longitude', 'Latitude'])
# extracting unique postal code points
postal_points_list = []
for geom in lead_gdf_filtered.drop_duplicates('PartialPostalCode')['geometry']:
    postal_points_list.append((geom.x, geom.y))
postal_points = np.array(postal_points_list)
postal_codes = lead_gdf_filtered.drop_duplicates('PartialPostalCode')['PartialPostalCode'].tolist()

# building kd tree for fast nearest neighbor lookup
tree = cKDTree(postal_points)

# finding the closest postal code to a given point
def nearest_postal_code(point):
    _, idx = tree.query([point.x, point.y])
    return postal_codes[idx]


# building the graph connecting postal codes through pipes
def build_postal_graph(year):
    
    G = nx.Graph()
    # adding all postal codes as nodes
    for code, geom in zip(postal_codes, postal_points):
        G.add_node(code, geometry=Point(geom))

    # iterating through all pipes and connecting postal codes
    for _, row in pipes.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        # handling different geometry types
        if geom.geom_type == 'LineString':
            lines = [geom]
        elif geom.geom_type == 'MultiLineString':
            lines = list(geom.geoms)
        else:
            lines = []

        # processing each line segment
        for i in lines:
            coords = list(i.coords)
            # finding which postal codes this pipe connects
            start_code = nearest_postal_code(Point(coords[0]))
            end_code = nearest_postal_code(Point(coords[-1]))
            if start_code != end_code:
                # creating edge if it doesn't exist
                if not G.has_edge(start_code,end_code):
                    G.add_edge(start_code,end_code, diameters=[], materials=[], ages=[])
                # calculating pipe age and storing attributes
                age = year - (row.get('Watermain Construction Year',2000) or 2000)
                G[start_code][end_code]['diameters'].append(row.get('Watermain Diameter',0))
                G[start_code][end_code]['materials'].append(str(row.get('Watermain Material','UNKNOWN')).upper())
                G[start_code][end_code]['ages'].append(age)
    return G


# material risk factors for lead contamination
material_factor = {'LEAD':1.0,'CI':0.6,'CICL':0.6,'DIP':0.3,'COPPER':0.2,'PVC':0.1,'UNKNOWN':0.4}

# assigning weights to edges based on material and diameter
def assign_edge_weights(G):
    for u,v,d in G.edges(data=True):
        mats = d.get('materials',[])
        dias = d.get('diameters',[])
        # calculating average material factor
        if mats:
            mat_vals = []
            for m in mats:
                mat_vals.append(material_factor.get(m, 0.4))
            mat_factor = np.mean(mat_vals)
        else:
            mat_factor = 0.4
        # getting average diameter or using default
        if dias:
            avg_dia = np.mean(dias)
        else:
            avg_dia = 200
        # combining material and diameter into edge weight
        G[u][v]['weight'] = mat_factor*(avg_dia/100.0)
    return G

# graph attention network model for predictions
class GAT(torch.nn.Module):
    def __init__(self,in_feats,hidden_feats=8,out_feats=1,heads=4):
        super().__init__()
        # first gat layer with multiple attention heads
        self.conv1 = GATConv(in_feats,hidden_feats,heads=heads,dropout=0.6)
        # second layer that combines the heads
        self.conv2 = GATConv(hidden_feats*heads,out_feats,heads=1,concat=False,dropout=0.6)
    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x,edge_index)
        # applying elu activation
        x = F.elu(x)
        x = self.conv2(x,edge_index)
        return x

# checking if gpu is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# years to predict for
years = range(2018, 2027)
# using 3 years of data for training
window = 3

for y in years:
    # collecting training years
    train_years = []
    for i in range(y - window, y):
        train_years.append(i)
    # skipping if we don't have enough historical data
    if min(train_years) < lead_gdf['year'].min():
        continue

    # splitting into train and target datasets
    train_gdf = lead_gdf[lead_gdf['year'].isin(train_years)]
    target_gdf = lead_gdf[lead_gdf['year'] == y]
    # averaging lead values by postal code
    lead_train_avg = train_gdf.groupby('PartialPostalCode')['Lead Amount (ppm)'].mean().to_dict()
    lead_target_avg = target_gdf.groupby('PartialPostalCode')['Lead Amount (ppm)'].mean().to_dict()

    # building graph for this year
    G = build_postal_graph(y)
    G = assign_edge_weights(G)
    node_list = list(G.nodes)

    # creating edge list for pytorch geometric
    edges = []
    node_to_index = {}
    for i, n in enumerate(node_list):
        node_to_index[n] = i

    # adding bidirectional edges
    for u, v in G.edges:
        idx_u, idx_v = node_to_index[u], node_to_index[v]
        edges.append([idx_u, idx_v])
        edges.append([idx_v, idx_u])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # extractting training lead values
    lead_train_vals_list = []

    for n in node_list:
        lead_train_vals_list.append(lead_train_avg.get(n, 0.0))
    lead_train_vals = np.array(lead_train_vals_list)
    
    # getting node degrees for features
    degree_vals_list = []

    for n in node_list:
        degree_vals_list.append(G.degree[n])
    degree_vals = np.array(degree_vals_list)
    
    # getting target values for evaluation
    lead_target_vals_list = []

    for n in node_list:
        lead_target_vals_list.append(lead_target_avg.get(n, np.nan))

    lead_target_vals = np.array(lead_target_vals_list)

    # normalizing features to 0-1 range
    lead_min, lead_max = np.nanmin(lead_train_vals), np.nanmax(lead_train_vals)

    lead_scaled = (lead_train_vals - lead_min) / (lead_max - lead_min + 1e-8)
    degree_scaled = (degree_vals - degree_vals.min()) / (degree_vals.max() - degree_vals.min() + 1e-8)

    # stacking features and creating tensors
    x = torch.tensor(np.stack([lead_scaled, degree_scaled], axis=1), dtype=torch.float)
    y_target = torch.tensor(np.nan_to_num((lead_target_vals - lead_min) / (lead_max - lead_min + 1e-8)), dtype=torch.float, device=device)

    # setting up model and optimizer
    data = Data(x=x.to(device), edge_index=edge_index.to(device))
    model = GAT(in_feats=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    # training the model
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data).squeeze()
        loss = criterion(out, y_target)
        loss.backward()
        optimizer.step()

    # making predictions
    model.eval()
    with torch.no_grad():
        predicted_scaled = model(data).squeeze().cpu().numpy()

    # converting back to original scale
    predicted = predicted_scaled * (lead_max - lead_min) + lead_min

    # cleaning postal codes for matching
    postal_coords['PartialPostalCode'] = postal_coords['PartialPostalCode'].str.strip().str.upper()
    node_list_clean = []
    for n in node_list:
        node_list_clean.append(n.strip().upper())
    # creating prediction dictionary
    pred_dict = dict(zip(node_list_clean, predicted))

    # mapping predictions to postal codes
    postal_gdf_unique = postal_coords[postal_coords['PartialPostalCode'].isin(node_list_clean)].copy()
    postal_gdf_unique['lead_gat'] = postal_gdf_unique['PartialPostalCode'].map(pred_dict)
    postal_gdf_unique['lead_gat'] = postal_gdf_unique['lead_gat'].fillna(0)
    # scaling for visualization purposes
    postal_gdf_unique['lead_scaled'] = postal_gdf_unique['lead_gat'] * 1e5

    postal_gdf_unique = gpd.GeoDataFrame(postal_gdf_unique, geometry=gpd.points_from_xy(postal_gdf_unique['Longitude'], postal_gdf_unique['Latitude']), crs="EPSG:4326")

    # creating the scatter plot map
    vmin, vmax = 0, 10
    postal_gdf_unique['lead_clipped'] = postal_gdf_unique['lead_scaled'].clip(upper=vmax)
    fig, ax = plt.subplots(figsize=(10, 10))
    # plotting pipes in background
    pipes.plot(ax=ax, color='black', linewidth=0.5, alpha=0.3)
    postal_gdf_unique.plot(ax=ax, column='lead_clipped', legend=True, markersize=60, vmin=vmin, vmax=vmax)
    plt.title(f"GAT-Predicted Lead by Postal Code ({y}, Linear Scale, ×10^-5)")
    plt.axis('off')
    plt.savefig(f"out/maps/predicted/GAT_Lead_Prediction_{y}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # thermal heatmap thing : just for cool visuals
    # projecting to web mercator for better visualization
    pipes_proj = pipes.to_crs("EPSG:3857")
    postal_proj = postal_gdf_unique.to_crs("EPSG:3857")
    
    # extracting coordinates and values
    xs = postal_proj.geometry.x.values
    ys = postal_proj.geometry.y.values
    vals_ppm = postal_proj["lead_gat"].values  # predicted values in ppm
    
    if len(xs) > 0:
        # use pipe boundaries to set the plot area
        pxmin, pymin, pxmax, pymax = pipes_proj.total_bounds
        padding = 500  # meters, just to give some space
        xmin = pxmin - padding 
        ymin = pymin - padding
        xmax= pxmax + padding
        ymax = pymax + padding
        
        # create grid for interpolation
        grid_size = 300
        grid_x, grid_y = np.mgrid[
            xmin:xmax:complex(0, grid_size),
            ymin:ymax:complex(0, grid_size)
        ]
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        
        # IDW interpolation using k nearest neighbors
        points_xy = np.column_stack([xs, ys])
        tree_heat = cKDTree(points_xy)
        k = 8  # number of neighbors to use
        power = 2.0  # IDW power parameter
        
        # computing distances and weights
        dists, idxs = tree_heat.query(grid_points, k=k)
        # avoid division by zero
        dists = np.where(dists == 0, 1e-9, dists)
        weights = 1.0 / (dists ** power)
        weights_sum = weights.sum(axis=1)
        neighbor_vals = vals_ppm[idxs]
        # interpolatting values across the grid
        grid_vals = (weights * neighbor_vals).sum(axis=1) / weights_sum
        grid_vals = grid_vals.reshape(grid_x.shape)
        
        # clip to reasonable range
        ppm_vmin = 0.0
        ppm_vmax = np.nanmax(vals_ppm) if len(vals_ppm) > 0 else 10.0
        grid_vals = np.clip(grid_vals, ppm_vmin, ppm_vmax)
        
        # mask to only show areas near pipes (makes it look better)
        pipes_union = pipes_proj.unary_union
        pipe_buffer = pipes_union.buffer(100)  # 100m buffer around pipes
        
        # creatting mask for areas near pipes
        mask_inside = np.array([
            pipe_buffer.contains(Point(x, y))
            for x, y in grid_points
        ]).reshape(grid_x.shape)
        

        grid_vals_masked = np.where(mask_inside, grid_vals, np.nan)
        grid_vals_masked = np.ma.array(grid_vals_masked, mask=np.isnan(grid_vals_masked))
        
        # trying to get turbo colormap, fallback to jet
        try:
            cmap = plt.get_cmap("turbo")  
        except:
            cmap = plt.get_cmap("jet")  # fb
        
        fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
        fig.patch.set_facecolor("white")
        
        # displayng the heatmap
        im = ax.imshow(
            grid_vals_masked.T,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap=cmap,
            vmin=ppm_vmin,
            vmax=ppm_vmax,
            interpolation="bilinear"
        )
        
        # overlay the pipes on top
        pipes_proj.plot(ax=ax, color="k", linewidth=0.3, alpha=0.6)
        
        # add postal points very faintly just to see where data is
        postal_proj.plot(ax=ax, markersize=4, color="white", alpha=0.6)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Predicted Lead (ppm)")
        ax.set_title(f"GAT-Predicted Lead Thermal Map ({y})", fontsize=14)
        ax.set_axis_off()
        plt.tight_layout()
        
        #saving the heatmap
        os.makedirs("out/maps/predicted", exist_ok=True)
        plt.savefig(f"out/maps/predicted/GAT_Lead_Thermal_{y}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        print(f"No postal points for heatmap in year {y}")

    # collectting results for csv output
    output_rows = []
    for i in range(len(node_list_clean)):
        n = node_list_clean[i]
        pred = predicted[i]
        y_true = lead_target_avg.get(n, np.nan)
        # checking if we have actual values for comparison
        if y_true == y_true:  
            y_true_ppm = float(y_true)
        else:
            y_true_ppm = ""
        output_rows.append({"year": y, "PartialPostalCode": n, "y_pred_ppm": float(pred), "y_true_ppm": y_true_ppm})

    # saving predictions to csv
    pred_df = pd.DataFrame(output_rows)
    csv_path = "out/predictions/gat_predictions_by_year.csv"
    # appending if file exists, otherwise creating new
    if os.path.exists(csv_path):
        mode = "a"
        header = False
    else:
        mode = "w"
        header = True
    pred_df.to_csv(csv_path, mode=mode, header=header, index=False)


