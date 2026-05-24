import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

out_dir = "out/analysis/roc"
os.makedirs(out_dir, exist_ok=True)

predictions_df = pd.read_csv("out/predictions/gat_predictions_by_year.csv")
predictions_df.columns = predictions_df.columns.str.strip()
predictions_df["year"] = pd.to_numeric(predictions_df["year"], errors="coerce")
predictions_df["PartialPostalCode"] = predictions_df["PartialPostalCode"].str.strip().str.upper()

lead_data = pd.read_csv("leadcont.csv")
lead_data.columns = lead_data.columns.str.strip()
lead_data["Sample Date"] = pd.to_datetime(lead_data["Sample Date"], errors="coerce")
lead_data["year"] = lead_data["Sample Date"].dt.year
lead_data["Lead Amount (ppm)"] = pd.to_numeric(lead_data["Lead Amount (ppm)"].astype(str).str.replace("<", "", regex=False), errors="coerce",)
lead_data["PartialPostalCode"] = lead_data["PartialPostalCode"].str.strip().str.upper()

year = [2024, 2025]
results = {}

for y in year:
    year_predictions = predictions_df[predictions_df["year"] == y].copy()
    if len(year_predictions) == 0:
        print(f"[{y}] No data was extracted")
        continue

    between_years = lead_data["year"].between(y - 2, y)
    year_actuals = lead_data[between_years].copy()
    if len(year_actuals) == 0:
        print(f"[{y}] No actual years [{y-2}, {y}]")
        continue

    year_actuals_avg = (
        year_actuals.groupby("PartialPostalCode")["Lead Amount (ppm)"]
        .mean()
        .reset_index()
        .rename(columns={"Lead Amount (ppm)": "y_true_ppm"})
    )

    if "y_true_ppm" in year_predictions.columns:
        year_predictions = year_predictions.drop(columns=["y_true_ppm"])

    year_data = year_predictions.merge(year_actuals_avg, on="PartialPostalCode", how="left")
    year_data = year_data.dropna(subset=["y_pred_ppm", "y_true_ppm"])
    
    # if len(year_data) == 0:
    #     print(f"[{y}] year data is empty.")
    #     continue

    y_true_v = year_data["y_true_ppm"].values
    y_pred_scores = year_data["y_pred_ppm"].values
    threshold = np.median(y_true_v)
    y_true = (y_true_v >= threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        print(f"[{y}] Only one class present")
        continue

    fp_r, tp_r, _ = roc_curve(y_true, y_pred_scores)
    roc_auc = roc_auc_score(y_true, y_pred_scores)

    results[y] = {
        "fp_r": fp_r,
        "tp_r": tp_r,
        "auc": roc_auc,
        "threshold": threshold,
        "len": len(year_data),
    }

    print(f"[{y}] ROC-AUC = {roc_auc:.4f}, threshold = {threshold:.8f}, samples = {len(year_data)}")

valid_years = [y for y in year if y in results]
len = len(valid_years)

f, a = plt.subplots(1, len, figsize=(6 * len, 6), sharex=True, sharey=True)
if len == 1:
    a = [a]

for i, y in zip(a, valid_years):
    r = results[y]
    i.plot(r["fp_r"], r["tp_r"], lw=2.5, label=f"AUC = {r['auc']:.3f}")
    i.plot([0, 1], [0, 1], lw=1.5, linestyle="--", label="Baseline (AUC = 0.50)")
    i.set_xlim([0.0, 1.0])
    i.set_ylim([0.0, 1.02])
    i.set_xlabel("False Positive Rate", fontsize=12)
    i.set_ylabel("True Positive Rate", fontsize=12)
    i.set_title(f"Year {y}\nMedian = {r['threshold']:.8f} ppm\nn = {r['len']}",fontsize=13, fontweight="bold",)
    i.legend(loc="lower right", fontsize=10, frameon=True)
    i.grid(True, alpha=0.3)

f.suptitle("ROC Curve that we got for GAT Lead Risk Prediction\nNote: this is based on ≥ 3 year median per year)",fontsize=16,fontweight="bold",y=1.03,)
plt.tight_layout()
path = os.path.join(out_dir, "ROC_Curves_2024_2025.png")
plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close()
print(f"ROC Curve saved to {path}")