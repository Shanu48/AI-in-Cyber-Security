import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             accuracy_score, f1_score)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings
warnings.filterwarnings('ignore')

# Device-agnostic setup: works on CPU, GPU (CUDA), and Apple Silicon (MPS)
print(f"TensorFlow version: {tf.__version__}")
physical_devices = tf.config.list_physical_devices()
print(f"Available devices: {[d.device_type for d in physical_devices]}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU(s) detected: {len(gpus)}. Memory growth enabled.")
else:
    print("No GPU detected. Using CPU.")

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')
PALETTE = sns.color_palette('Set2')
LABEL_COLORS = {0: PALETTE[0], 1: PALETTE[1]}
LABEL_NAMES = {0: 'Normal', 1: 'Attack'}

print(f"Pandas version: {pd.__version__}")

# Load datasets
train_df = pd.read_csv('UNSW_NB15_train_40k.csv')
test_df = pd.read_csv('UNSW_NB15_test_10k.csv')

print(f"Training set shape: {train_df.shape}")
print(f"Test set shape:     {test_df.shape}")
print(f"\nColumn names:\n{list(train_df.columns)}")

# Data types
print("=== Training Set Data Types ===")
print(train_df.dtypes)
print(f"\nCategorical columns: {list(train_df.select_dtypes(include='object').columns)}")
print(f"Numerical columns:   {list(train_df.select_dtypes(include=np.number).columns)}")

# Info for both datasets
print("=== Training Set Info ===")
train_df.info()
print("\n=== Test Set Info ===")
test_df.info()

# Descriptive statistics for training set
train_df.describe(include='all').T

# First rows
print("=== Training Set: First 5 Rows ===")
print(train_df.head())
print("\n=== Test Set: First 5 Rows ===")
print(test_df.head())

# Missing value counts
train_missing = train_df.isnull().sum()
test_missing = test_df.isnull().sum()

print("=== Training Set Missing Values ===")
print(train_missing[train_missing > 0] if train_missing.sum() > 0 else "No missing values found.")
print(f"\nTotal missing cells (train): {train_missing.sum()}")

print("\n=== Test Set Missing Values ===")
print(test_missing[test_missing > 0] if test_missing.sum() > 0 else "No missing values found.")
print(f"\nTotal missing cells (test): {test_missing.sum()}")

# Missing value heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.heatmap(train_df.isnull(), cbar=True, yticklabels=False, cmap='viridis', ax=axes[0])
axes[0].set_title('Missing Values Heatmap (Training Set)', fontsize=13)
axes[0].set_xlabel('Features')

sns.heatmap(test_df.isnull(), cbar=True, yticklabels=False, cmap='viridis', ax=axes[1])
axes[1].set_title('Missing Values Heatmap (Test Set)', fontsize=13)
axes[1].set_xlabel('Features')

plt.tight_layout()
plt.savefig('figures/fig_01.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_01.png')

# Class distribution bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (df, name) in enumerate([(train_df, 'Training'), (test_df, 'Test')]):
    counts = df['label'].value_counts().sort_index()
    bars = axes[idx].bar(
        [LABEL_NAMES[i] for i in counts.index],
        counts.values,
        color=[LABEL_COLORS[i] for i in counts.index],
        edgecolor='black', linewidth=0.8
    )
    for bar, val in zip(bars, counts.values):
        axes[idx].annotate(f'{val:,}\n({val/len(df)*100:.1f}%)',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
    axes[idx].set_title(f'Class Distribution ({name} Set)', fontsize=13)
    axes[idx].set_ylabel('Count')
    axes[idx].set_xlabel('Label')

plt.tight_layout()
plt.savefig('figures/fig_02.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_02.png')

print(f"Training set class ratio (Attack/Normal): {train_df['label'].value_counts()[1] / train_df['label'].value_counts()[0]:.3f}")
print(f"Test set class ratio (Attack/Normal):     {test_df['label'].value_counts()[1] / test_df['label'].value_counts()[0]:.3f}")

# Value counts for categorical features
categorical_cols = ['proto', 'state', 'service']

for col in categorical_cols:
    print(f"\n=== {col} ===")
    print(f"Unique values (train): {train_df[col].nunique()}")
    print(f"Unique values (test):  {test_df[col].nunique()}")
    print(f"\nTop 10 values (train):")
    print(train_df[col].value_counts().head(10))

# Bar charts for categorical feature value counts (top 15)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, col in enumerate(categorical_cols):
    top_vals = train_df[col].value_counts().head(15)
    bars = axes[idx].barh(
        top_vals.index[::-1], top_vals.values[::-1],
        color=PALETTE[idx], edgecolor='black', linewidth=0.5
    )
    for bar, val in zip(bars, top_vals.values[::-1]):
        axes[idx].text(bar.get_width() + max(top_vals.values)*0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:,}', va='center', fontsize=9)
    axes[idx].set_title(f'Top {min(15, len(top_vals))} Values: {col}', fontsize=13)
    axes[idx].set_xlabel('Count')

plt.tight_layout()
plt.savefig('figures/fig_03.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_03.png')

# Grouped bar charts: categorical feature distribution by label
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

for idx, col in enumerate(categorical_cols):
    top_categories = train_df[col].value_counts().head(10).index
    subset = train_df[train_df[col].isin(top_categories)]

    ct = pd.crosstab(subset[col], subset['label'])
    ct.columns = ['Normal', 'Attack']
    ct = ct.loc[top_categories]

    ct.plot(kind='barh', ax=axes[idx], color=[LABEL_COLORS[0], LABEL_COLORS[1]],
            edgecolor='black', linewidth=0.5)
    axes[idx].set_title(f'{col}: Distribution by Label (Top 10)', fontsize=13)
    axes[idx].set_xlabel('Count')
    axes[idx].set_ylabel(col)
    axes[idx].legend(title='Label')

plt.tight_layout()
plt.savefig('figures/fig_04.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_04.png')

# Stacked percentage bar charts: attack rate per category
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

for idx, col in enumerate(categorical_cols):
    top_categories = train_df[col].value_counts().head(10).index
    subset = train_df[train_df[col].isin(top_categories)]

    ct = pd.crosstab(subset[col], subset['label'], normalize='index') * 100
    ct.columns = ['Normal %', 'Attack %']
    ct = ct.loc[top_categories]

    ct.plot(kind='barh', stacked=True, ax=axes[idx],
            color=[LABEL_COLORS[0], LABEL_COLORS[1]],
            edgecolor='black', linewidth=0.5)
    axes[idx].set_title(f'{col}: Attack Rate by Category (Top 10)', fontsize=13)
    axes[idx].set_xlabel('Percentage (%)')
    axes[idx].set_ylabel(col)
    axes[idx].legend(title='Label')
    axes[idx].set_xlim(0, 100)

plt.tight_layout()
plt.savefig('figures/fig_05.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_05.png')

numerical_cols = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'sload', 'dload',
                  'sttl', 'dttl', 'smean', 'dmean', 'sinpkt']

print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")

# Overlaid histograms for all numerical features, colored by label
fig, axes = plt.subplots(4, 3, figsize=(20, 22))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    for label_val in [0, 1]:
        data = train_df[train_df['label'] == label_val][col]
        axes[idx].hist(data, bins=50, alpha=0.6, label=LABEL_NAMES[label_val],
                       color=LABEL_COLORS[label_val], edgecolor='black', linewidth=0.3)
    axes[idx].set_title(f'Distribution of {col}', fontsize=12)
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend(fontsize=9)
    axes[idx].ticklabel_format(style='scientific', axis='y', scilimits=(0, 3))

plt.suptitle('Numerical Feature Distributions by Label (Training Set)', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('figures/fig_06.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_06.png')

# KDE plots for numerical features by label
fig, axes = plt.subplots(4, 3, figsize=(20, 22))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    for label_val in [0, 1]:
        data = train_df[train_df['label'] == label_val][col]
        if data.std() > 0:
            sns.kdeplot(data, ax=axes[idx], label=LABEL_NAMES[label_val],
                        color=LABEL_COLORS[label_val], fill=True, alpha=0.3)
    axes[idx].set_title(f'KDE: {col}', fontsize=12)
    axes[idx].set_xlabel(col)
    axes[idx].legend(fontsize=9)

plt.suptitle('Kernel Density Estimates by Label (Training Set)', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('figures/fig_07.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_07.png')

# Box plots for all numerical features grouped by label
fig, axes = plt.subplots(4, 3, figsize=(20, 22))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    sns.boxplot(x='label', y=col, data=train_df, ax=axes[idx],
                palette=[LABEL_COLORS[0], LABEL_COLORS[1]], showfliers=False)
    axes[idx].set_title(f'Box Plot: {col} by Label', fontsize=12)
    axes[idx].set_xticklabels(['Normal', 'Attack'])
    axes[idx].set_xlabel('Label')
    axes[idx].set_ylabel(col)

plt.suptitle('Numerical Feature Box Plots by Label (Outliers Hidden for Clarity)', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('figures/fig_08.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_08.png')

# Violin plots for all numerical features
fig, axes = plt.subplots(4, 3, figsize=(20, 22))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    sns.violinplot(x='label', y=col, data=train_df, ax=axes[idx],
                   palette=[LABEL_COLORS[0], LABEL_COLORS[1]], inner='quartile', cut=0)
    axes[idx].set_title(f'Violin Plot: {col} by Label', fontsize=12)
    axes[idx].set_xticklabels(['Normal', 'Attack'])
    axes[idx].set_xlabel('Label')
    axes[idx].set_ylabel(col)

plt.suptitle('Numerical Feature Violin Plots by Label (Training Set)', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('figures/fig_09.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_09.png')

# Compute skewness for all numerical features
skew_values = train_df[numerical_cols].skew().sort_values(ascending=True)
print("=== Skewness Values ===")
for col, val in skew_values.items():
    flag = " [highly skewed]" if abs(val) > 2 else ""
    print(f"  {col:>10s}: {val:>10.3f}{flag}")

# Horizontal bar chart of skewness
fig, ax = plt.subplots(figsize=(12, 7))

colors = ['#e74c3c' if abs(v) > 2 else '#3498db' if abs(v) > 1 else '#2ecc71' for v in skew_values.values]
bars = ax.barh(skew_values.index, skew_values.values, color=colors, edgecolor='black', linewidth=0.5)

ax.axvline(x=0, color='black', linewidth=0.8)
ax.axvline(x=2, color='red', linewidth=0.8, linestyle='--', alpha=0.5, label='|skew| = 2')
ax.axvline(x=-2, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axvline(x=1, color='orange', linewidth=0.8, linestyle='--', alpha=0.5, label='|skew| = 1')
ax.axvline(x=-1, color='orange', linewidth=0.8, linestyle='--', alpha=0.5)

for bar, val in zip(bars, skew_values.values):
    ax.text(bar.get_width() + 0.1 if val >= 0 else bar.get_width() - 0.1,
            bar.get_y() + bar.get_height()/2, f'{val:.2f}',
            va='center', ha='left' if val >= 0 else 'right', fontsize=10)

ax.set_title('Skewness of Numerical Features', fontsize=14)
ax.set_xlabel('Skewness')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('figures/fig_10.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_10.png')

# Kurtosis analysis
kurt_values = train_df[numerical_cols].kurtosis().sort_values(ascending=True)
print("=== Kurtosis Values ===")
for col, val in kurt_values.items():
    flag = " [heavy tails]" if val > 7 else ""
    print(f"  {col:>10s}: {val:>12.3f}{flag}")

fig, ax = plt.subplots(figsize=(12, 7))
colors = ['#e74c3c' if v > 7 else '#3498db' for v in kurt_values.values]
ax.barh(kurt_values.index, kurt_values.values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('Kurtosis of Numerical Features', fontsize=14)
ax.set_xlabel('Kurtosis')
plt.tight_layout()
plt.savefig('figures/fig_11.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_11.png')

# Correlation matrix (lower triangle)
corr_cols = numerical_cols + ['label']
corr_matrix = train_df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            mask=mask, square=True, linewidths=0.5, ax=ax,
            vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap (Numerical Features + Label)', fontsize=14)
plt.tight_layout()
plt.savefig('figures/fig_12.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_12.png')

# Full correlation heatmap
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            cbar_kws={'shrink': 0.8})
ax.set_title('Full Correlation Matrix (Numerical Features + Label)', fontsize=14)
plt.tight_layout()
plt.savefig('figures/fig_13.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_13.png')

# Top correlations with label
label_corr = corr_matrix['label'].drop('label').sort_values(ascending=True)

print("=== Correlations with Label ===")
for col, val in label_corr.items():
    print(f"  {col:>10s}: {val:>7.4f}")

fig, ax = plt.subplots(figsize=(12, 7))
colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in label_corr.values]
bars = ax.barh(label_corr.index, label_corr.values, color=colors, edgecolor='black', linewidth=0.5)

for bar, val in zip(bars, label_corr.values):
    ax.text(bar.get_width() + 0.005 if val >= 0 else bar.get_width() - 0.005,
            bar.get_y() + bar.get_height()/2, f'{val:.4f}',
            va='center', ha='left' if val >= 0 else 'right', fontsize=10)

ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_title('Feature Correlation with Label (Target)', fontsize=14)
ax.set_xlabel('Pearson Correlation Coefficient')
plt.tight_layout()
plt.savefig('figures/fig_14.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_14.png')

# Top 15 feature-feature correlations
upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
corr_pairs = upper_tri.stack().reset_index()
corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
corr_pairs['abs_corr'] = corr_pairs['Correlation'].abs()
top_pairs = corr_pairs.sort_values('abs_corr', ascending=False).head(15)

print("=== Top 15 Feature Correlations (by absolute value) ===")
for _, row in top_pairs.iterrows():
    print(f"  {row['Feature 1']:>10s} <-> {row['Feature 2']:<10s}: {row['Correlation']:>7.4f}")

# Select top 5 features most correlated with label
top_5_features = label_corr.abs().sort_values(ascending=False).head(5).index.tolist()
print(f"Top 5 features correlated with label: {top_5_features}")

# Pair plot colored by label (sample for speed)
plot_df = train_df[top_5_features + ['label']].sample(n=min(5000, len(train_df)), random_state=42)
plot_df['label'] = plot_df['label'].map(LABEL_NAMES)

g = sns.pairplot(plot_df, hue='label', palette={'Normal': PALETTE[0], 'Attack': PALETTE[1]},
                 diag_kind='kde', plot_kws={'alpha': 0.4, 's': 15, 'edgecolor': 'none'},
                 height=2.5)
g.figure.suptitle('Pairwise Scatter Matrix: Top 5 Label-Correlated Features', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/fig_15.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_15.png')

# Scatter plots for the top 3 most correlated feature pairs
top_3_pairs = corr_pairs.sort_values('abs_corr', ascending=False).head(3)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, (_, row) in enumerate(top_3_pairs.iterrows()):
    f1, f2, corr_val = row['Feature 1'], row['Feature 2'], row['Correlation']
    sample = train_df.sample(n=min(5000, len(train_df)), random_state=42)

    for label_val in [0, 1]:
        m = sample['label'] == label_val
        axes[idx].scatter(sample.loc[m, f1], sample.loc[m, f2],
                         alpha=0.3, s=10, label=LABEL_NAMES[label_val],
                         color=LABEL_COLORS[label_val], edgecolors='none')
    axes[idx].set_xlabel(f1, fontsize=11)
    axes[idx].set_ylabel(f2, fontsize=11)
    axes[idx].set_title(f'{f1} vs {f2} (r={corr_val:.3f})', fontsize=12)
    axes[idx].legend(fontsize=9)

plt.suptitle('Top 3 Correlated Feature Pairs, Colored by Label', fontsize=14)
plt.tight_layout()
plt.savefig('figures/fig_16.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_16.png')

# Grouped statistics by label
print("=== Mean Values by Label ===")
grouped_means = train_df.groupby('label')[numerical_cols].mean().T
grouped_means.columns = ['Normal (mean)', 'Attack (mean)']
grouped_means['Difference'] = grouped_means['Attack (mean)'] - grouped_means['Normal (mean)']
grouped_means['Ratio'] = grouped_means['Attack (mean)'] / grouped_means['Normal (mean)'].replace(0, np.nan)
print(grouped_means.round(3).to_string())

# Heatmap of normalized mean feature values by class
norm_means = train_df.groupby('label')[numerical_cols].mean()
scaler_viz = MinMaxScaler()
norm_means_scaled = pd.DataFrame(
    scaler_viz.fit_transform(norm_means.T).T,
    index=['Normal', 'Attack'],
    columns=numerical_cols
)

fig, ax = plt.subplots(figsize=(16, 4))
sns.heatmap(norm_means_scaled, annot=True, fmt='.2f', cmap='YlOrRd',
            linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Normalized Mean Feature Values by Class', fontsize=14)
ax.set_ylabel('Class')
plt.tight_layout()
plt.savefig('figures/fig_17.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_17.png')

# Make working copies to preserve originals
train = train_df.copy()
test = test_df.copy()

# Check missing values in both sets
train_missing = train.isnull().sum()
test_missing = test.isnull().sum()

missing_report = pd.DataFrame({
    'Train Missing': train_missing[train_missing > 0],
    'Test Missing': test_missing[test_missing > 0]
}).fillna(0).astype(int)

if missing_report.empty:
    print("No missing values found in either set.")
else:
    print("Missing values before imputation:")
    print(missing_report)

categorical_cols = ['proto', 'state', 'service']
numerical_cols = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts',
                  'sload', 'dload', 'sttl', 'dttl', 'smean', 'dmean', 'sinpkt']

# Impute numerical columns with median (robust to outliers/skew)
for col in numerical_cols:
    if train[col].isnull().any() or test[col].isnull().any():
        median_val = train[col].median()
        train[col].fillna(median_val, inplace=True)
        test[col].fillna(median_val, inplace=True)

# Impute categorical columns with mode
for col in categorical_cols:
    if train[col].isnull().any() or test[col].isnull().any():
        mode_val = train[col].mode()[0]
        train[col].fillna(mode_val, inplace=True)
        test[col].fillna(mode_val, inplace=True)

# Verify
total_after = train.isnull().sum().sum() + test.isnull().sum().sum()
print(f"Total missing values after imputation: {total_after}")

# Inspect unique values in both sets
for col in categorical_cols:
    train_unique = set(train[col].unique())
    test_unique = set(test[col].unique())
    unseen = test_unique - train_unique
    print(f"[{col}]  Train unique: {len(train_unique)}  |  Test unique: {len(test_unique)}  |  Unseen in test: {len(unseen)}")
    if unseen:
        print(f"         Unseen categories: {unseen}")
    print()

from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    known_classes = list(train[col].unique())
    le.fit(known_classes + ['__unseen__'])

    # Map unseen test categories to the placeholder before transforming
    test[col] = test[col].apply(lambda x: x if x in known_classes else '__unseen__')

    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

print("Encoded dtypes:")
print(train[categorical_cols].dtypes)
print()
print("Sample encoded values (train):")
print(train[categorical_cols].head())

# Box plots BEFORE capping
plot_features = ['dur', 'sbytes', 'dbytes', 'sload', 'dload', 'smean']

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Numerical Features BEFORE Outlier Capping', fontsize=14)
for ax, feat in zip(axes.ravel(), plot_features):
    ax.boxplot(train[feat].dropna(), vert=True)
    ax.set_title(feat)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figures/fig_18.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_18.png')

# IQR-based capping (fitted on train statistics only)
clip_bounds = {}

for col in numerical_cols:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    clip_bounds[col] = (lower, upper)

    train[col] = train[col].clip(lower=lower, upper=upper)
    test[col] = test[col].clip(lower=lower, upper=upper)

print("Clip bounds (lower, upper) per feature:")
for col, (lo, hi) in clip_bounds.items():
    print(f"  {col:>10s}: [{lo:.4f}, {hi:.4f}]")

# Box plots AFTER capping
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Numerical Features AFTER Outlier Capping', fontsize=14)
for ax, feat in zip(axes.ravel(), plot_features):
    ax.boxplot(train[feat].dropna(), vert=True)
    ax.set_title(feat)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figures/fig_19.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_19.png')

from sklearn.preprocessing import StandardScaler

# Store unscaled copies for before/after comparison
train_unscaled = train[numerical_cols].copy()

scaler = StandardScaler()
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])

print("Post-scaling train statistics (should be ~0 mean, ~1 std):")
print(train[numerical_cols].describe().loc[['mean', 'std']].round(4))

# Before/after distribution comparison for selected features
compare_feats = ['dur', 'sbytes', 'sload', 'smean']

fig, axes = plt.subplots(len(compare_feats), 2, figsize=(14, 3.5 * len(compare_feats)))

for i, feat in enumerate(compare_feats):
    axes[i, 0].hist(train_unscaled[feat], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[i, 0].set_title(f'{feat} - Before Scaling')
    axes[i, 0].set_ylabel('Count')

    axes[i, 1].hist(train[feat], bins=50, color='darkorange', edgecolor='black', alpha=0.7)
    axes[i, 1].set_title(f'{feat} - After Scaling')
    axes[i, 1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('figures/fig_20.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_20.png')

from sklearn.feature_selection import mutual_info_classif

feature_cols = [c for c in train.columns if c != 'label']

mi_scores = mutual_info_classif(
    train[feature_cols], train['label'], discrete_features='auto', random_state=42
)

mi_df = pd.DataFrame({
    'Feature': feature_cols,
    'MI Score': mi_scores
}).sort_values('MI Score', ascending=False).reset_index(drop=True)

print(mi_df.to_string(index=False))

# Bar chart of mutual information scores
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(mi_df['Feature'], mi_df['MI Score'], color='teal', edgecolor='black')
ax.set_xlabel('Mutual Information Score')
ax.set_title('Feature Importance (Mutual Information with Target)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('figures/fig_21.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_21.png')

# Drop features with negligible MI (threshold: 0.01)
mi_threshold = 0.01
low_mi = mi_df[mi_df['MI Score'] < mi_threshold]['Feature'].tolist()

if low_mi:
    print(f"Dropping {len(low_mi)} feature(s) with MI < {mi_threshold}: {low_mi}")
    train.drop(columns=low_mi, inplace=True)
    test.drop(columns=low_mi, inplace=True)
else:
    print(f"All features have MI >= {mi_threshold}. Keeping all {len(feature_cols)} features.")

selected_features = [c for c in train.columns if c != 'label']
print(f"\nFeatures retained: {len(selected_features)}")

from imblearn.over_sampling import SMOTE

X_train_pre = train.drop(columns=['label'])
y_train_pre = train['label']

# Class distribution before SMOTE
before_counts = y_train_pre.value_counts().sort_index()
print("Class distribution BEFORE SMOTE:")
print(before_counts)
print()

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pre, y_train_pre)

after_counts = pd.Series(y_train_resampled).value_counts().sort_index()
print("Class distribution AFTER SMOTE:")
print(after_counts)

# Side-by-side bar chart: before vs after SMOTE
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

axes[0].bar(['Normal (0)', 'Attack (1)'], before_counts.values,
            color=['#4c72b0', '#dd8452'], edgecolor='black')
axes[0].set_title('Before SMOTE')
axes[0].set_ylabel('Sample Count')
for j, v in enumerate(before_counts.values):
    axes[0].text(j, v + 100, str(v), ha='center', fontweight='bold')

axes[1].bar(['Normal (0)', 'Attack (1)'], after_counts.values,
            color=['#4c72b0', '#dd8452'], edgecolor='black')
axes[1].set_title('After SMOTE')
for j, v in enumerate(after_counts.values):
    axes[1].text(j, v + 100, str(v), ha='center', fontweight='bold')

fig.suptitle('Training Set Class Distribution: Before vs After SMOTE', fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('figures/fig_22.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_22.png')

# Prepare final arrays
X_test = test.drop(columns=['label'])
y_test = test['label']

X_train = pd.DataFrame(X_train_resampled, columns=X_test.columns)
y_train = pd.Series(y_train_resampled, name='label')

print("=" * 55)
print("  PREPROCESSED DATA SUMMARY")
print("=" * 55)
print(f"  X_train shape : {X_train.shape}")
print(f"  y_train shape : {y_train.shape}")
print(f"  X_test shape  : {X_test.shape}")
print(f"  y_test shape  : {y_test.shape}")
print(f"  Features      : {list(X_train.columns)}")
print(f"  Target classes : {sorted(y_train.unique())}")
print("=" * 55)
print()
print("X_train dtypes:")
print(X_train.dtypes)

# Sanity checks
assert X_train.isnull().sum().sum() == 0, "NaNs found in X_train"
assert X_test.isnull().sum().sum() == 0, "NaNs found in X_test"
assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch between train and test"
assert set(X_train.columns) == set(X_test.columns), "Feature name mismatch between train and test"

print("All sanity checks passed. Data is ready for modeling.")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n=== Feature Engineering ===")

# Create interaction features that capture network flow characteristics
# bytes_ratio: ratio of source to total bytes (directional asymmetry)
X_train['bytes_ratio'] = X_train['sbytes'] / (X_train['sbytes'] + X_train['dbytes'] + 1e-8)
X_test['bytes_ratio'] = X_test['sbytes'] / (X_test['sbytes'] + X_test['dbytes'] + 1e-8)

# pkt_ratio: ratio of source to total packets
X_train['pkt_ratio'] = X_train['spkts'] / (X_train['spkts'] + X_train['dpkts'] + 1e-8)
X_test['pkt_ratio'] = X_test['spkts'] / (X_test['spkts'] + X_test['dpkts'] + 1e-8)

# load_ratio: ratio of source to total load
X_train['load_ratio'] = X_train['sload'] / (X_train['sload'] + X_train['dload'] + 1e-8)
X_test['load_ratio'] = X_test['sload'] / (X_test['sload'] + X_test['dload'] + 1e-8)

# bytes_per_pkt_src: average bytes per source packet
X_train['bytes_per_pkt_src'] = X_train['sbytes'] / (X_train['spkts'] + 1e-8)
X_test['bytes_per_pkt_src'] = X_test['sbytes'] / (X_test['spkts'] + 1e-8)

# ttl_diff: difference between source and destination TTL
X_train['ttl_diff'] = X_train['sttl'] - X_train['dttl']
X_test['ttl_diff'] = X_test['sttl'] - X_test['dttl']

# mean_ratio: ratio of source to total mean packet size
X_train['mean_ratio'] = X_train['smean'] / (X_train['smean'] + X_train['dmean'] + 1e-8)
X_test['mean_ratio'] = X_test['smean'] / (X_test['smean'] + X_test['dmean'] + 1e-8)

print(f"Features after engineering: {X_train.shape[1]} (added 6 interaction features)")
print(f"New features: {['bytes_ratio', 'pkt_ratio', 'load_ratio', 'bytes_per_pkt_src', 'ttl_diff', 'mean_ratio']}")

# Handle any NaN/inf from division
X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)

print(f"Final X_train shape: {X_train.shape}")
print(f"Final X_test shape: {X_test.shape}")

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, fbeta_score, average_precision_score,
    precision_recall_curve
)
from sklearn.model_selection import cross_val_score
import time

results = {}
training_times = {}

# Tune LR regularization strength
from sklearn.model_selection import GridSearchCV

print("Tuning Logistic Regression C parameter...")
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs', penalty='l2'),
    param_grid={'C': [0.01, 0.1, 1.0, 10.0, 100.0]},
    cv=3, scoring='f1', n_jobs=-1, verbose=0
)
lr_grid.fit(X_train, y_train)
print(f"Best C: {lr_grid.best_params_['C']} (CV F1: {lr_grid.best_score_:.4f})")

start = time.time()
lr = lr_grid.best_estimator_
training_times['Logistic Regression'] = time.time() - start

y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

results['Logistic Regression'] = {'y_pred': y_pred_lr, 'y_prob': y_prob_lr, 'model': lr}
print(f"Logistic Regression training complete. Time: {training_times['Logistic Regression']:.2f}s")
print(f"Test accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")

# Tune var_smoothing for NB
print("Tuning Naive Bayes var_smoothing...")
nb_grid = GridSearchCV(
    GaussianNB(),
    param_grid={'var_smoothing': np.logspace(-12, -3, 10)},
    cv=3, scoring='f1', n_jobs=-1, verbose=0
)
nb_grid.fit(X_train, y_train)
print(f"Best var_smoothing: {nb_grid.best_params_['var_smoothing']:.2e} (CV F1: {nb_grid.best_score_:.4f})")

start = time.time()
gnb = nb_grid.best_estimator_
training_times['Naive Bayes'] = time.time() - start

y_pred_gnb = gnb.predict(X_test)
y_prob_gnb = gnb.predict_proba(X_test)[:, 1]

results['Naive Bayes'] = {'y_pred': y_pred_gnb, 'y_prob': y_prob_gnb, 'model': gnb}
print(f"Gaussian Naive Bayes training complete. Time: {training_times['Naive Bayes']:.2f}s")
print(f"Test accuracy: {accuracy_score(y_test, y_pred_gnb):.4f}")

# Find optimal k using cross-validation on a small range
from sklearn.model_selection import cross_val_score

print("Tuning KNN with GridSearchCV...")
knn_grid = GridSearchCV(
    KNeighborsClassifier(n_jobs=-1),
    param_grid={
        'n_neighbors': [1, 3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski', 'manhattan'],
    },
    cv=3, scoring='f1', n_jobs=-1, verbose=0
)
knn_grid.fit(X_train, y_train)
print(f"Best KNN params: {knn_grid.best_params_} (CV F1: {knn_grid.best_score_:.4f})")

start = time.time()
knn = knn_grid.best_estimator_
training_times['KNN'] = time.time() - start

y_pred_knn = knn.predict(X_test)
y_prob_knn = knn.predict_proba(X_test)[:, 1]

results['KNN'] = {'y_pred': y_pred_knn, 'y_prob': y_prob_knn, 'model': knn}
print(f"KNN training complete. Time: {training_times['KNN']:.2f}s")
print(f"Test accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")

start = time.time()
svm = SVC(kernel='rbf', probability=True, random_state=42, C=10.0, gamma='scale', cache_size=1000)
svm.fit(X_train, y_train)
training_times['SVM'] = time.time() - start

y_pred_svm = svm.predict(X_test)
y_prob_svm = svm.predict_proba(X_test)[:, 1]

results['SVM'] = {'y_pred': y_pred_svm, 'y_prob': y_prob_svm, 'model': svm}
print(f"SVM training complete (C=10, RBF). Time: {training_times['SVM']:.2f}s")
print(f"Test accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")

start = time.time()
dt = DecisionTreeClassifier(
    random_state=42, max_depth=25, min_samples_split=10,
    min_samples_leaf=5, criterion='gini', class_weight='balanced'
)
dt.fit(X_train, y_train)
training_times['Decision Tree'] = time.time() - start

y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

results['Decision Tree'] = {'y_pred': y_pred_dt, 'y_prob': y_prob_dt, 'model': dt}
print(f"Decision Tree training complete (depth=25, balanced). Time: {training_times['Decision Tree']:.2f}s")
print(f"Test accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")

start = time.time()
rf = RandomForestClassifier(
    n_estimators=500, random_state=42, max_depth=None,
    min_samples_split=2, min_samples_leaf=1,
    max_features='sqrt', bootstrap=True, n_jobs=-1,
    class_weight='balanced_subsample'
)
rf.fit(X_train, y_train)
training_times['Random Forest'] = time.time() - start

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

results['Random Forest'] = {'y_pred': y_pred_rf, 'y_prob': y_prob_rf, 'model': rf}
print(f"Random Forest training complete (500 trees, unlimited depth). Time: {training_times['Random Forest']:.2f}s")
print(f"Test accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# Feature importance from Random Forest
rf_importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nRandom Forest Feature Importances:")
for feat, imp in rf_importance.items():
    print(f"  {feat:>10s}: {imp:.4f}")

start = time.time()
gb = GradientBoostingClassifier(
    n_estimators=500, random_state=42, max_depth=7,
    learning_rate=0.05, subsample=0.8, min_samples_split=10,
    min_samples_leaf=5, max_features='sqrt',
    n_iter_no_change=20, validation_fraction=0.1, tol=1e-4
)
gb.fit(X_train, y_train)
training_times['Gradient Boosting'] = time.time() - start

y_pred_gb = gb.predict(X_test)
y_prob_gb = gb.predict_proba(X_test)[:, 1]

results['Gradient Boosting'] = {'y_pred': y_pred_gb, 'y_prob': y_prob_gb, 'model': gb}
print(f"Gradient Boosting training complete (500 trees, lr=0.05, depth=7). Time: {training_times['Gradient Boosting']:.2f}s")
print(f"Test accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
print(f"Gradient Boosting used {gb.n_estimators_} estimators (early stopping may have triggered)")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split as tts

tf.random.set_seed(42)
np.random.seed(42)

# Shuffle and create a proper stratified validation split from SMOTE'd training data
# This avoids the Keras validation_split problem where the last N% is not representative
X_tr_dnn, X_val_dnn, y_tr_dnn, y_val_dnn = tts(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)
print(f"DNN train: {X_tr_dnn.shape}, DNN val: {X_val_dnn.shape}")
print(f"DNN val class distribution: {pd.Series(y_val_dnn).value_counts().to_dict()}")

input_dim = X_train.shape[1]

dnn = Sequential([
    Input(shape=(input_dim,)),
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.25),
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(16, activation='relu', kernel_initializer='he_normal'),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.001)
dnn.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

dnn.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

start = time.time()
history = dnn.fit(
    X_tr_dnn, y_tr_dnn,
    epochs=150,
    batch_size=512,
    validation_data=(X_val_dnn, y_val_dnn),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
training_times['DNN'] = time.time() - start
print(f"\nDNN training time: {training_times['DNN']:.2f}s")
print(f"Best val_loss: {min(history.history['val_loss']):.4f} at epoch {np.argmin(history.history['val_loss'])+1}")

# Training and validation loss / accuracy curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title('DNN Training and Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Training Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_title('DNN Training and Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig_23.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_23.png')

y_prob_dnn = dnn.predict(X_test, verbose=0).flatten()
y_pred_dnn = (y_prob_dnn >= 0.5).astype(int)

results['DNN'] = {'y_pred': y_pred_dnn, 'y_prob': y_prob_dnn, 'model': dnn}
print("Deep Neural Network training and prediction complete.")

# Compute all metrics for every model
model_names = list(results.keys())
metrics_data = []

for name in model_names:
    y_pred = results[name]['y_pred']
    y_prob = results[name]['y_prob']

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
    f2_macro = fbeta_score(y_test, y_pred, beta=2, average='macro', zero_division=0)
    auc_pr = average_precision_score(y_test, y_prob)

    metrics_data.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F2 Score': f2,
        'F2-Macro': f2_macro,
        'AUC-PR': auc_pr
    })

metrics_df = pd.DataFrame(metrics_data)
metrics_df.set_index('Model', inplace=True)
print("Metrics computed for all models.")

for cm_idx, name in enumerate(model_names):
    y_pred = results[name]['y_pred']

    print("=" * 70)
    print(f"  {name}")
    print("=" * 70)
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'], ax=ax)
    ax.set_title(f'Confusion Matrix: {name}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    cm_fig_name = f'figures/cm_{cm_idx+1:02d}_{name.lower().replace(" ", "_")}.png'
    plt.savefig(cm_fig_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {cm_fig_name}')
    print()

metric_cols = ['Accuracy', 'Precision', 'Recall', 'F2 Score', 'F2-Macro', 'AUC-PR']

x = np.arange(len(model_names))
width = 0.12
fig, ax = plt.subplots(figsize=(18, 7))

for i, metric in enumerate(metric_cols):
    offset = (i - len(metric_cols) / 2 + 0.5) * width
    bars = ax.bar(x + offset, metrics_df[metric].values, width, label=metric)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison: All Evaluation Metrics', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
ax.legend(loc='lower right', fontsize=9)
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig_25.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_25.png')

fig, ax = plt.subplots(figsize=(10, 7))

colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

for idx, name in enumerate(model_names):
    y_prob = results[name]['y_prob']
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)
    ax.plot(recall_vals, precision_vals, color=colors[idx],
            label=f'{name} (AUC-PR={auc_pr:.4f})', linewidth=1.5)

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves: All Models', fontsize=14)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('figures/fig_26.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_26.png')

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()

for idx, name in enumerate(model_names):
    y_pred = results[name]['y_pred']
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'], ax=axes[idx])
    axes[idx].set_title(name, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

for idx in range(len(model_names), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Confusion Matrices: All Models', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('figures/fig_27.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_27.png')

summary_df = metrics_df.copy()
summary_df = summary_df.sort_values(by='F2 Score', ascending=False)

print("\n=== Model Performance Summary (sorted by F2 Score) ===")
print(summary_df.round(4).to_string())

# Feature importance from Random Forest
rf_importance = pd.Series(
    results['Random Forest']['model'].feature_importances_,
    index=X_train.columns
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
rf_importance.plot(kind='barh', color='forestgreen', edgecolor='black', ax=ax)
ax.set_title('Random Forest Feature Importances', fontsize=14)
ax.set_xlabel('Importance (Gini)')
plt.tight_layout()
plt.savefig('figures/fig_28.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_28.png')

# Training time comparison
time_df = pd.Series(training_times).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(time_df)))
time_df.plot(kind='barh', color=colors, edgecolor='black', ax=ax)

for i, (name, val) in enumerate(time_df.items()):
    ax.text(val + max(time_df) * 0.01, i, f'{val:.1f}s', va='center', fontsize=10)

ax.set_title('Model Training Time Comparison', fontsize=14)
ax.set_xlabel('Time (seconds)')
plt.tight_layout()
plt.savefig('figures/fig_29.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved figures/fig_29.png')

# Save metrics to CSV for the report
metrics_df_sorted = metrics_df.sort_values('F2 Score', ascending=False)
metrics_df_sorted.to_csv('figures/metrics_summary.csv')
print("\nFinal Metrics Summary:")
print(metrics_df_sorted.round(4).to_string())
print("\nTraining Times:")
for name, t in sorted(training_times.items(), key=lambda x: x[1]):
    print(f"  {name:>20s}: {t:.2f}s")
