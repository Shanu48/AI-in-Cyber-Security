"""Generate additional innovative visualizations for the report."""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Load metrics
df = pd.read_csv('figures/metrics_summary.csv', index_col=0)
df = df.sort_values('F2 Score', ascending=False)

# ============================================================
# 1. RADAR / SPIDER CHART comparing all models
# ============================================================
categories = ['Accuracy', 'Precision', 'Recall', 'F2 Score', 'F2-Macro', 'AUC-PR']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

colors = plt.cm.tab10(np.linspace(0, 1, len(df)))

for idx, (model, row) in enumerate(df.iterrows()):
    values = [row[c] for c in categories]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=1.8, label=model, color=colors[idx], markersize=4)
    ax.fill(angles, values, alpha=0.05, color=colors[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0.75, 1.0)
ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
ax.set_yticklabels(['0.80', '0.85', '0.90', '0.95', '1.00'], fontsize=8, color='gray')
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
ax.set_title('Model Performance Radar Chart', fontsize=14, pad=20)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/extra_radar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/extra_radar.png")

# ============================================================
# 2. HEATMAP of all metrics
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Normalize each column to [0,1] for color mapping, but show actual values
display_df = df[categories]
norm_df = (display_df - display_df.min()) / (display_df.max() - display_df.min() + 1e-8)

im = ax.imshow(norm_df.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Annotate with actual values
for i in range(len(df)):
    for j in range(len(categories)):
        val = display_df.iloc[i, j]
        text_color = 'white' if norm_df.iloc[i, j] < 0.3 else 'black'
        ax.text(j, i, f'{val:.4f}', ha='center', va='center', fontsize=10,
                color=text_color, fontweight='bold')

ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, fontsize=11, rotation=30, ha='right')
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df.index, fontsize=11)
ax.set_title('Model Performance Heatmap (sorted by F2 Score)', fontsize=14)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Relative Performance', fontsize=10)

plt.tight_layout()
plt.savefig('figures/extra_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/extra_heatmap.png")

# ============================================================
# 3. PARALLEL COORDINATES plot
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))

# Normalize metrics to [0,1] for parallel coordinates
norm_data = (display_df - display_df.min()) / (display_df.max() - display_df.min() + 1e-8)

x_positions = range(len(categories))

for idx, (model, row) in enumerate(norm_data.iterrows()):
    values = [row[c] for c in categories]
    linewidth = 3 if model in ['SVM', 'Gradient Boosting', 'Random Forest'] else 1.5
    alpha = 0.9 if linewidth == 3 else 0.5
    ax.plot(x_positions, values, 'o-', label=model, color=colors[idx],
            linewidth=linewidth, alpha=alpha, markersize=6)

ax.set_xticks(x_positions)
ax.set_xticklabels(categories, fontsize=11, rotation=15)
ax.set_ylabel('Normalized Score (0 = worst, 1 = best in group)', fontsize=11)
ax.set_title('Parallel Coordinates: Model Comparison', fontsize=14)
ax.legend(loc='lower left', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(-0.05, 1.1)

plt.tight_layout()
plt.savefig('figures/extra_parallel.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/extra_parallel.png")

# ============================================================
# 4. LOLLIPOP CHART for F2 scores
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

f2_sorted = df['F2 Score'].sort_values(ascending=True)
y_pos = range(len(f2_sorted))

# Horizontal lollipop
for i, (model, val) in enumerate(f2_sorted.items()):
    color = '#2ecc71' if val > 0.93 else '#f39c12' if val > 0.90 else '#e74c3c'
    ax.hlines(y=i, xmin=0.85, xmax=val, color=color, linewidth=2.5)
    ax.plot(val, i, 'o', color=color, markersize=10, markeredgecolor='black', markeredgewidth=0.5)
    ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(f2_sorted.index, fontsize=11)
ax.set_xlabel('F2 Score', fontsize=12)
ax.set_title('F2 Score Ranking (Higher = Better Attack Detection)', fontsize=14)
ax.set_xlim(0.85, 1.0)
ax.grid(True, alpha=0.3, axis='x')

# Add threshold annotation
ax.axvline(x=0.93, color='gray', linestyle='--', alpha=0.5)
ax.text(0.931, len(f2_sorted)-0.5, 'F2 = 0.93', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('figures/extra_lollipop.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/extra_lollipop.png")

# ============================================================
# 5. ACCURACY vs TRAINING TIME scatter (log scale)
# ============================================================
# Parse training times from the log
times = {
    'KNN': 0.00, 'Naive Bayes': 0.00, 'Logistic Regression': 0.00,
    'Decision Tree': 0.19, 'Random Forest': 4.30, 'Gradient Boosting': 15.22,
    'DNN': 25.59, 'SVM': 131.37
}
# Use 0.01 for near-zero times for log scale
for k in times:
    if times[k] < 0.01:
        times[k] = 0.01

fig, ax = plt.subplots(figsize=(10, 7))

for model in df.index:
    acc = df.loc[model, 'Accuracy']
    t = times.get(model, 0.01)
    f2 = df.loc[model, 'F2 Score']
    size = f2 * 300  # bubble size proportional to F2
    color_idx = list(df.index).index(model)
    ax.scatter(t, acc, s=size, c=[colors[color_idx]], edgecolors='black',
               linewidth=0.8, alpha=0.8, zorder=5)
    ax.annotate(model, (t, acc), textcoords="offset points",
                xytext=(8, 8), fontsize=9)

ax.set_xscale('log')
ax.set_xlabel('Training Time (seconds, log scale)', fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Accuracy vs. Training Time (bubble size = F2 Score)', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/extra_scatter_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/extra_scatter_time.png")

# ============================================================
# 6. GROUPED BAR: Precision vs Recall side by side
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

models = list(df.index)
x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, df['Precision'].values, width, label='Precision',
               color='#3498db', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, df['Recall'].values, width, label='Recall',
               color='#e74c3c', edgecolor='black', linewidth=0.5)

# Annotate
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, rotation=45)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, rotation=45)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Precision vs. Recall: Per-Model Comparison', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=25, ha='right', fontsize=10)
ax.legend(fontsize=11)
ax.set_ylim(0.82, 1.02)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/extra_prec_recall.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/extra_prec_recall.png")

print("\nAll extra plots generated.")
