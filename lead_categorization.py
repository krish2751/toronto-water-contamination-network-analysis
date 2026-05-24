import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


lead_data_csv = pd.read_csv('leadcont.csv')

lead_data_csv.columns = lead_data_csv.columns.str.strip()

lead_data_csv['Sample Date'] = pd.to_datetime(lead_data_csv['Sample Date'], errors='coerce')
lead_data_csv['year'] =lead_data_csv['Sample Date'].dt.year
lead_data_csv['Lead Amount (ppm)'] =pd.to_numeric(lead_data_csv['Lead Amount (ppm)'].astype(str).str.replace('<', '', regex=False), errors='coerce')

lead_data_filtered = lead_data_csv.dropna(subset=['Lead Amount (ppm)']).copy()
lead_data_filtered['Lead Amount (ppb)'] = lead_data_filtered['Lead Amount (ppm)'] * 1000

#defining the categories for defining rules according to canadian government 
categories = {
    'Level 0': (0,5, 'None detectable'),
    'Level 1': (5,10, 'Minimal contamination, yet no safe level for children'),
    'Level 2': (10,15, 'Violates trigger level under the Revised Lead Rule; action required'),
    'Level 3': (15,float('inf'), 'Exceeds EPA action level, indicating significant contamination and need for immediate action')
}

# fxn to cateogrize lead values (in ppb)
def categorize_lead(ppb):

    if pd.isna(ppb):
        return None
    elif ppb < 5:
        return 'Level 0'
    elif ppb < 10:
        return 'Level 1'
    elif ppb < 15:
        return 'Level 2'
    else:
        return 'Level 3'

lead_data_filtered['Category'] = lead_data_filtered['Lead Amount (ppb)'].apply(categorize_lead)

#total no. of samples
n = len(lead_data_filtered)

categorization_results = []
all_levels = ['Level 0', 'Level 1', 'Level 2', 'Level 3']

for level in all_levels:

    c = len(lead_data_filtered[lead_data_filtered['Category'] == level])
    prp = (c / n) * 100
    min_val, max_val, desc = categories[level]
    if max_val == float('inf'):
        range_str = f">{min_val}"
    else:
        range_str = f"{min_val} - {max_val}"
    categorization_results.append({'Category': level,'Lead Level (ppb)': range_str,'Count': c,'Proportion (%)': prp,'Implications': desc})

result_df = pd.DataFrame(categorization_results)
result_df.to_csv('out/analysis/categorization/Lead_Categorization_Table.csv', index=False)

fig,axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
category_counts = lead_data_filtered['Category'].value_counts()

category_counts_ordered = pd.Series({cat: category_counts.get(cat, 0) for cat in all_levels})
colors = ['green', 'yellow', 'orange', 'red']
color_map = {}
# mapping clr with levels
for i, cat in enumerate(all_levels):
    color_map[cat] = colors[i]

bars = ax1.bar(category_counts_ordered.index, category_counts_ordered.values,color=[color_map[cat] for cat in category_counts_ordered.index], alpha=0.7, edgecolor='black')
ax1.set_xlabel('Lead Contamination Category', fontsize=12)
ax1.set_ylabel('Number of Samples', fontsize=12)
ax1.set_title('Distribution of Lead Contamination Categories\n(All Years)', fontsize=13, fontweight='bold')
ax1.grid(True, axis='y', alpha=0.3)

for bar, cat in zip(bars, category_counts_ordered.index):
    cnt = category_counts_ordered[cat]
    if cnt>0:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{cnt}\n({cnt/n*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)

ax2 = axes[1]
# proportions of each category for each level .
# this is use to create a pie chart showing the proportion of each category for each level.

proportions = []
for cat in all_levels:
    p = category_counts_ordered.get(cat, 0) / n*100
    proportions.append(p if p > 0 else 0.1)

labels = []
for cat in all_levels:
    p = category_counts_ordered.get(cat, 0)
    labels.append(f'{cat}\n({p} samples)')

wedges, texts , autotexts = ax2.pie(proportions, labels=labels, colors=colors, autopct=lambda pct: f'{pct:.1f}%' if pct > 0.5 else '', startangle=90)
ax2.set_title('Proportion of Lead Contamination Categories', fontsize=13, fontweight='bold')

plt.suptitle('Lead Contamination Categorization Analysis\n(According to EPA Standards)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('out/analysis/categorization/Lead_Categorization_Analysis.png', dpi=300, bbox_inches='tight')
plt.close()
