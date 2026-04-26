# Socioeconomic Drivers of Climate Change: Multi-Target ML Analysis (1900–2023)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)

A comprehensive machine learning study analyzing how socioeconomic factors—such as GDP, population, and energy usage—drive global climate indicators (Temperature Anomaly and CO₂ Emissions) across 195 countries over 123 years.

## 🌟 Project Overview
This project leverages multi-target regression models to simultaneously predict environmental outcomes and socioeconomic patterns. By integrating historical datasets from 1900 to 2023, the study identifies critical drivers of climate change and simulates policy scenarios to provide actionable insights for sustainable development.

### Key Features
- **Multi-Target Learning**: Simultaneous prediction of Temperature Anomaly and CO₂ Emissions.
- **Advanced Feature Engineering**: Time-aware features including lags, rolling statistics, and per-capita indicators.
- **Interpretable ML**: Global and local feature importance analysis using SHAP (SHapley Additive exPlanations).
- **Policy Simulator**: A "Digital Twin Lite" framework to project the impact of economic and energy policy shifts.
- **Robust Validation**: Expanded-window time-series cross-validation to ensure zero data leakage.

## 📂 Project Structure
```text
.
├── data/                   # Dataset storage
│   ├── raw/                # Original source files
│   └── processed/          # Cleaned, engineered panel data
├── notebooks/              # Modular Jupyter Workflows
│   ├── 01_eda_visualizations.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_model_training_tuning.ipynb
│   └── 04_interpretation.ipynb
├── src/                    # Reusable Python Modules
│   ├── data_loader.py      # Pipeline for loading and splitting
│   ├── feature_engineering.py # Lag/rolling window logic
│   ├── models.py           # Model definitions (XGB, LGBM, etc.)
│   └── visualization.py    # Custom plotting utilities
├── outputs/                # Generated artifacts
│   ├── figures/            # Plots, SHAP charts, and maps
│   └── models/             # Saved model checkpoints
├── Report/                 # LaTeX source for the research paper
├── requirements.txt        # Dependency specification
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/socioeconomic-climate-ml.git
   cd socioeconomic-climate-ml
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Usage
Run the modular notebooks in order to reproduce the full pipeline:
1.  **Exploratory Data Analysis**: `notebooks/01_eda_visualizations.ipynb`
2.  **Preprocessing**: `notebooks/02_preprocessing_pipeline.ipynb`
3.  **Model Training**: `notebooks/03_model_training_tuning.ipynb`
4.  **Interpretability**: `notebooks/04_interpretation.ipynb`

## 🧠 Methodology

### Data Pipeline
We utilize a panel dataset spanning 1900–2023 across 195 countries. The preprocessing pipeline handles:
- **Missing Value Imputation**: Time-aware KNN imputer.
- **Scaling**: Robust scaling to handle outliers in economic data.
- **Feature Engineering**: Creation of 10+ derivate features (e.g., Energy Intensity, Policy-GDP interaction).

### Modeling Approach
We benchmark several architectures for multi-target regression:
- **Gradient Boosting**: XGBoost, LightGBM (optimized via Optuna).
- **Ensemble**: Random Forest Regressor.
- **Neural Networks**: MLP (Multi-Layer Perceptron).
- **Baseline**: Ridge Regression.

## 📊 Key Results
- **Predictability Asymmetry**: Socioeconomic indicators provide a strong signal for CO₂ emissions ($R^2 \approx 0.9+$ in some regions) but exhibit near-zero linear correlation with local temperature anomalies, highlighting the complexity of climate systems.
- **Primary Drivers**: Fossil fuel usage and population growth remain the dominant predictors for emissions, while policy interventions show non-linear "threshold" effects when combined with high urbanization.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors
- **[Your Name]** - *Lead Researcher/Developer*
- **[Collaborator Name]** - *Data Scientist*

---
*Developed for [Course Name/Project Context] at [Institution Name].*
