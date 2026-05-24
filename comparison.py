import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

import matplotlib.pyplot as plt

# importing the req. csvs
lead = pd.read_csv("leadcont.csv")
lead.columns = lead.columns.str.strip()
lead['Sample Date'] = pd.to_datetime(lead['Sample Date'], errors='coerce', dayfirst=True)
lead['year'] = lead['Sample Date'].dt.year
lead['Lead Amount (ppm)'] = pd.to_numeric(lead['Lead Amount (ppm)'].astype(str).str.replace('<', '', regex=False), errors='coerce')

# grouping the data by pst. codes and yrs.
actual = lead.groupby(['PartialPostalCode', 'year'], as_index=False)['Lead Amount (ppm)'].mean().rename(columns={'Lead Amount (ppm)': 'actual_ppm'})
actual['PartialPostalCode'] = actual['PartialPostalCode'].str.strip().str.upper()

# now we will import the predictions data
preds_df = pd.read_csv("out/predictions/gat_predictions_by_year.csv")
preds_df.columns = preds_df.columns.str.strip()

# cleaning the data (pred. data)
preds_df['PartialPostalCode'] = preds_df['PartialPostalCode'].str.strip().str.upper()
preds_df['year'] = pd.to_numeric(preds_df['year'],errors='coerce')

df = preds_df.merge(actual, on=['PartialPostalCode','year'], how='inner')
df = df.dropna(subset=['y_pred_ppm', 'actual_ppm'])

#creating a list to store the metrics
metrics = []

#calc. the metrics for each year
for i, j in df.groupby('year'):
    y_true_per_yr = j['actual_ppm'].to_numpy()
    y_pred_per_yr = j['y_pred_ppm'].to_numpy()

    #calc. the metrics
    mae_per_yr = mean_absolute_error(y_true_per_yr, y_pred_per_yr)
    rmse_per_yr = root_mean_squared_error(y_true_per_yr, y_pred_per_yr)
    r2_per_yr = r2_score(y_true_per_yr, y_pred_per_yr)
    metrics.append((i, mae_per_yr, rmse_per_yr, r2_per_yr))

y_true_all = df['actual_ppm'].to_numpy()
y_pred_all = df['y_pred_ppm'].to_numpy()
mae_all = mean_absolute_error(y_true_all, y_pred_all)
rmse_all = root_mean_squared_error(y_true_all, y_pred_all)
r2_all = r2_score(y_true_all, y_pred_all)

#plotting graph of actual vs predicted lead levels. 
plt.figure(figsize=(6,6))

years =df['year']

sc = plt.scatter(df['actual_ppm'], df['y_pred_ppm'], c=years, cmap='viridis', s=30, alpha=0.8)

plt.colorbar(sc, label='Year')
lims = [min(df['actual_ppm'].min(), df['y_pred_ppm'].min()), max(df['actual_ppm'].max(), df['y_pred_ppm'].max())]
plt.plot(lims, lims, 'r--', label='Ideal (y=x)')
plt.xlabel("Actual Lead (ppm)")
plt.ylabel("Predicted Lead (ppm)")
plt.title("GAT Predictions vs Actual Lead Levels by Year")
plt.legend()
plt.tight_layout()
plt.savefig(f"out/analysis/comparison/compImg.png",dpi=300,bbox_inches='tight')
plt.close()
