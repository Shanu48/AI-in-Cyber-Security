# AI in Cybersecurity — Internal Assessment 4

**Network Intrusion Detection System Using Machine Learning and Deep Learning**

| | |
|---|---|
| **Name** | Shravani Sanjay Sawant |
| **Registration No.** | 230953006 |
| **Course** | ICT4416 — AI in Cybersecurity |
| **Department** | CCE, School of Computer Engineering, MIT Manipal |

## Overview

Binary classification of network traffic as **Normal** or **Attack** using the [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) dataset. Eight classifiers are trained and compared:

| Model | F2 Score | Accuracy |
|---|---|---|
| SVM | 0.9706 | 0.9217 |
| Gradient Boosting | 0.9336 | 0.9358 |
| Random Forest | 0.9326 | 0.9362 |
| DNN (Keras) | 0.9308 | 0.9134 |
| KNN | 0.9252 | 0.9308 |
| Decision Tree | 0.9249 | 0.9274 |
| Logistic Regression | 0.9194 | 0.8829 |
| Naive Bayes | 0.8849 | 0.8314 |

## Repository Structure

```
├── IA4_NIDS.ipynb          # Main Jupyter notebook (complete pipeline)
├── run_all.py              # Standalone Python script (same pipeline)
├── extra_plots.py          # Additional visualization scripts
├── UNSW_NB15_train_40k.csv # Training dataset (40,000 samples)
├── UNSW_NB15_test_10k.csv  # Test dataset (10,000 samples)
├── figures/                # All generated plots and confusion matrices
│   ├── fig_01–fig_29.png   # EDA and model evaluation plots
│   ├── cm_01–cm_08.png     # Individual confusion matrices
│   ├── extra_*.png         # Radar, heatmap, parallel coords, etc.
│   └── metrics_summary.csv # Tabulated model metrics
└── report.pdf              # Final report
```

## Pipeline

1. **Exploratory Data Analysis** — class distribution, feature distributions, correlation analysis
2. **Preprocessing** — label encoding, IQR outlier capping, StandardScaler, mutual information feature selection
3. **Feature Engineering** — 6 interaction features (byte/packet/load ratios, TTL difference)
4. **Class Balancing** — SMOTE on training set only
5. **Model Training** — GridSearchCV (3-fold CV) for hyperparameter tuning
6. **Evaluation** — Accuracy, Precision, Recall, F2, F2-Macro, AUC-PR on held-out test set

## Requirements

- Python 3.10+
- TensorFlow, scikit-learn, pandas, numpy, matplotlib, seaborn, imbalanced-learn

Install dependencies:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn imbalanced-learn scipy
```

## Usage

Run the notebook:
```bash
jupyter notebook IA4_NIDS.ipynb
```

Or run the standalone script:
```bash
python run_all.py
```
