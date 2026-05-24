import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('Book1.xlsx')

df.columns = df.columns.str.strip()

param_col = None
result_col = None
categories = {
    'Level 0': (0, 5, 'None detectable'),
    'Level 1': (5, 10, 'Minimal contamination, yet no safe level for children'),
    'Level 2': (10, 15, 'Violates trigger level under the Revised Lead Rule; action required'),
    'Level 3': (15, float('inf'), 'Exceeds EPA action level, indicating significant contamination and need for immediate action')
}

for col in df.columns:
    if 'parameter' in col.lower() and 'name' in col.lower():
        param_col = col
    if col.lower() == 'result':
        result_col = col

if param_col is None or result_col is None: exit(1)

lead_data = df[df[param_col].str.contains('lead', case=False, na=False)].copy()
if len(lead_data) == 0: exit(1)

lead_data = lead_data.dropna(subset=[result_col]).copy()
lead_data[result_col] = pd.to_numeric(lead_data[result_col], errors='coerce')
lead_data = lead_data.dropna(subset=[result_col]).copy()

lead_data['Lead Amount (ppb)'] = lead_data[result_col]

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

lead_data['Category'] = lead_data['Lead Amount (ppb)'].apply(categorize_lead)

total_samples = len(lead_data)

categorization_results = []
for level in ['Level 0', 'Level 1', 'Level 2', 'Level 3']:
    count = len(lead_data[lead_data['Category'] == level])
    proportion = (count / total_samples) * 100 if total_samples > 0 else 0
    min_val, max_val, desc = categories[level]
    range_str = f">{min_val}" if max_val == float('inf') else f"{min_val}-{max_val}"
    categorization_results.append({
        'Category': level,
        'Lead Level (ppb)': range_str,
        'Count': count,
        'Proportion (%)': proportion,
        'Implications': desc
    })

results_df = pd.DataFrame(categorization_results)
results_df.to_csv('out/analysis/categorization/Lead_Categorization_Table_Book1.csv', index=False)


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax1 = axes[0]
category_counts = lead_data['Category'].value_counts()
all_categories = ['Level 0', 'Level 1', 'Level 2', 'Level 3']
category_counts_ordered = pd.Series({cat: category_counts.get(cat, 0) for cat in all_categories})
colors = ['green', 'yellow', 'orange', 'red']
color_map = {cat: colors[i] for i, cat in enumerate(all_categories)}
bars = ax1.bar(category_counts_ordered.index, category_counts_ordered.values,color=[color_map[cat] for cat in category_counts_ordered.index], alpha=0.7, edgecolor='black')

ax1.set_xlabel('Lead Contamination Category', fontsize=12)
ax1.set_ylabel('Number of Samples', fontsize=12)
ax1.set_title('Distribution of Lead Contamination Categories\n(Book1.xlsx Data)', fontsize=13, fontweight='bold')
ax1.grid(True, axis='y', alpha=0.3)

for bar, cat in zip(bars, category_counts_ordered.index):
    count = category_counts_ordered[cat]
    if count > 0:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/total_samples*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)

ax2 = axes[1]

proportions = [category_counts_ordered.get(cat, 0) / total_samples * 100 if total_samples > 0 else 0 for cat in all_categories]
filtered_proportions = [p if p > 0 else 0.1 for p in proportions]
wedges, texts, autotexts = ax2.pie(filtered_proportions, labels=all_categories,colors=colors, autopct=lambda pct: f'{pct:.1f}%' if pct > 0.5 else '', startangle=90)
ax2.set_title('Proportion of Lead Contamination Categories', fontsize=13, fontweight='bold')

plt.suptitle('Lead Contamination Categorization Analysis (Book1.xlsx)\n(According to EPA Standards)', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('out/analysis/categorization/Lead_Categorization_Analysis_Book1.png', dpi=300, bbox_inches='tight')
plt.close()
