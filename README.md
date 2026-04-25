# NOT COMPLETED! IN PROGRESS CURRENTLY
---

# 📋 Complete Project Specification

## _Socioeconomic Drivers of Climate Change: Multi-Target ML Analysis (1900–2023)_

---

## 🎯 1. Problem Definition & Objectives

### Research Problem

> How do socioeconomic factors (GDP, population, energy use, policy) drive global temperature anomalies and CO₂ emissions, and can machine learning models accurately predict both targets simultaneously across 195 countries over 123 years?

### Clear Objectives

| #   | Objective                                                                                     | Success Metric                                                       |
| --- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 1   | Build a clean, time-aligned panel dataset linking socioeconomic indicators to climate outputs | 0 missing values in final feature matrix; documented transformations |
| 2   | Engineer time-aware features capturing lagged effects and structural shifts                   | ≥10 engineered features with documented rationale                    |
| 3   | Train ≥4 ML models for multi-target regression with rigorous time-series validation           | All models converge; no data leakage in CV                           |
| 4   | Compare model performance and interpret feature contributions                                 | Statistical significance test (Diebold-Mariano) + SHAP analysis      |
| 5   | Simulate policy/economic scenarios to generate actionable insights                            | ≥3 scenario analyses with clear visualizations                       |

---

## 📊 2. Dataset Schema & Loading

### Confirmed Structure (26 Columns)

```python
import pandas as pd

# Load data
df = pd.read_csv('climate_socioeconomic_1900_2023.csv')

# Target variables (what you predict)
TARGETS = ['Temperature_Anomaly', 'CO2_Emissions']

# Feature categories
FEATURES = {
    'environmental': [
        'Methane_Emissions', 'Sea_Level_Rise', 'Arctic_Ice_Extent',
        'Deforestation_Rate', 'Extreme_Weather_Events', 'Average_Rainfall',
        'Solar_Energy_Potential', 'Air_Pollution_Index', 'Biodiversity_Index',
        'Ocean_Acidification', 'Average_Temperature'
    ],
    'socioeconomic': [
        'Population', 'GDP', 'Renewable_Energy_Usage', 'Urbanization',
        'Waste_Management', 'Industrial_Activity', 'Fossil_Fuel_Usage',
        'Energy_Consumption_Per_Capita', 'Policy_Score'
    ],
    'derived': [
        'Forest_Area', 'Per_Capita_Emissions'  # Will engineer more
    ]
}

# Identifiers (exclude from modeling)
ID_COLS = ['Country', 'Year']

print(f"Dataset shape: {df.shape}")  # Expected: ~100,000 × 26
print(f"Year range: {df['Year'].min()}–{df['Year'].max()}")
print(f"Countries: {df['Country'].nunique()}")
```

### Quick Data Quality Checks

```python
# Check for duplicates
print(f"Duplicate (Country, Year) pairs: {df.duplicated(subset=ID_COLS).sum()}")

# Check missing values
missing = df.isnull().sum()
print(f"Columns with missing values:\n{missing[missing > 0]}")

# Basic stats for targets
print(df[TARGETS + ['Year']].groupby('Year').mean().describe())
```

---

## ⚙️ 3. Preprocessing Pipeline (Reusable Function)

```python
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import KNNImputer
import numpy as np

def prepare_climate_data(df, targets, id_cols, test_year_start=2010):
    """
    Full preprocessing pipeline for climate-economy panel data.

    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    df = df.copy()

    # 1. Remove exact duplicates
    df = df.drop_duplicates(subset=id_cols)

    # 2. Sort for time-series operations
    df = df.sort_values(id_cols).reset_index(drop=True)

    # 3. Feature Engineering
    df = engineer_features(df)

    # 4. Define final feature set (exclude targets + IDs)
    feature_cols = [c for c in df.columns if c not in targets + id_cols]

    # 5. Handle missing values (time-aware)
    imputer = KNNImputer(n_neighbors=5)
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    # 6. Create lag/rolling features (AFTER imputation to avoid leakage)
    df = add_time_features(df, targets, id_cols)

    # Update feature list after engineering
    feature_cols = [c for c in df.columns if c not in targets + id_cols]

    # 7. Train/test split by time (no future leakage)
    train_df = df[df['Year'] < test_year_start].copy()
    test_df = df[df['Year'] >= test_year_start].copy()

    # 8. Scale features
    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    y_train = train_df[targets].values
    y_test = test_df[targets].values

    return X_train, X_test, y_train, y_test, feature_cols, scaler, train_df, test_df


def engineer_features(df):
    """Create domain-informed derived features."""
    # Per-capita metrics
    df['CO2_per_Capita'] = df['CO2_Emissions'] / (df['Population'] + 1e-6)
    df['GDP_per_Capita'] = df['GDP'] / (df['Population'] + 1e-6)

    # Energy efficiency & transition
    df['Renewable_Ratio'] = df['Renewable_Energy_Usage'] / (df['Fossil_Fuel_Usage'] + df['Renewable_Energy_Usage'] + 1e-6)
    df['Energy_Intensity'] = df['Energy_Consumption_Per_Capita'] / (df['GDP_per_Capita'] + 1e-6)

    # Policy interaction
    df['Policy_GDP_Interaction'] = df['Policy_Score'] * df['GDP_per_Capita']

    # Decade indicator for structural breaks
    df['Decade'] = (df['Year'] // 10) * 10

    return df


def add_time_features(df, targets, id_cols):
    """Add lag and rolling window features per country."""
    df = df.copy()

    for col in targets + ['GDP', 'Renewable_Energy_Usage', 'Fossil_Fuel_Usage']:
        if col in df.columns:
            # Lag features (1yr, 5yr)
            for lag in [1, 5]:
                df[f'{col}_lag{lag}'] = df.groupby(id_cols[0])[col].shift(lag)

            # Rolling statistics (5-year window)
            df[f'{col}_roll5_mean'] = df.groupby(id_cols[0])[col].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )
            df[f'{col}_roll5_std'] = df.groupby(id_cols[0])[col].transform(
                lambda x: x.rolling(5, min_periods=1).std()
            )

    # Drop rows with NaN from lagging (only affects earliest years)
    df = df.dropna(subset=[c for c in df.columns if 'lag' in c])

    return df
```

---

## 🔍 4. Exploratory Data Analysis (EDA) Plan

### Required Visualizations for Paper

| Figure                                                      | Purpose                                     | Code Snippet                                                                        |
| ----------------------------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Fig 1**: Global Temp & CO₂ Trends (1900–2023)             | Show baseline climate trajectory            | `df.groupby('Year')[TARGETS].mean().plot(subplots=True)`                            |
| **Fig 2**: Correlation Heatmap (Socioeconomic vs Climate)   | Identify multicollinearity & key drivers    | `sns.heatmap(df[FEATURES_flat + TARGETS].corr(), cmap='coolwarm')`                  |
| **Fig 3**: GDP vs CO₂ Scatter with LOESS                    | Test Environmental Kuznets Curve hypothesis | `sns.regplot(x='GDP_per_Capita', y='CO2_Emissions', data=df, lowess=True)`          |
| **Fig 4**: Regional Boxplots (Policy Score vs Temp Anomaly) | Show policy effectiveness variation         | `sns.boxplot(x='Region', y='Temperature_Anomaly', hue='Policy_Score_Bin', data=df)` |
| **Fig 5**: STL Decomposition of Global Temp                 | Separate trend/seasonality/residual         | `from statsmodels.tsa.seasonal import seasonal_decompose`                           |

### Summary Statistics Table (Include in Paper)

```python
# Generate descriptive stats for key variables
summary_table = df.groupby('Year')[TARGETS + ['GDP', 'Population', 'Renewable_Energy_Usage']].agg([
    'mean', 'std', 'min', 'max'
]).round(3)

# Export for paper
summary_table.to_latex('tables/summary_stats.tex')  # Or .to_csv()
```

---

## 🤖 5. Model Implementation (4+ Models)

### Multi-Target Architecture

```python
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

# Model registry
MODELS = {
    'RandomForest': MultiOutputRegressor(RandomForestRegressor(random_state=42)),
    'XGBoost': MultiOutputRegressor(XGBRegressor(random_state=42, tree_method='hist')),
    'LightGBM': MultiOutputRegressor(LGBMRegressor(random_state=42)),
    'MLP': MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, early_stopping=True)),
    'Ridge_Baseline': MultiOutputRegressor(Ridge(alpha=1.0))  # Simple linear baseline
}
```

### Time-Series Cross-Validation (Critical!)

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def time_series_cv_evaluate(model, X, y, feature_names, targets, n_splits=5):
    """Expanding window CV for time-series multi-target regression."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {t: {'rmse': [], 'mae': [], 'r2': []} for t in targets}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        for i, target in enumerate(targets):
            results[target]['rmse'].append(np.sqrt(mean_squared_error(y_val[:, i], y_pred[:, i])))
            results[target]['mae'].append(mean_absolute_error(y_val[:, i], y_pred[:, i]))
            results[target]['r2'].append(r2_score(y_val[:, i], y_pred[:, i]))

        print(f"Fold {fold} complete")

    # Aggregate results
    summary = {}
    for target in targets:
        summary[target] = {
            'RMSE_mean': np.mean(results[target]['rmse']),
            'RMSE_std': np.std(results[target]['rmse']),
            'MAE_mean': np.mean(results[target]['mae']),
            'R2_mean': np.mean(results[target]['r2'])
        }

    return summary, results
```

### Hyperparameter Tuning with Optuna (XGBoost Example)

```python
import optuna

def optimize_xgboost(X, y, targets, n_trials=30):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42
        }

        model = MultiOutputRegressor(XGBRegressor(**params))
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            # Macro-average RMSE across targets
            rmse = np.mean([
                np.sqrt(mean_squared_error(y[val_idx][:, i], preds[:, i]))
                for i in range(len(targets))
            ])
            scores.append(-rmse)  # Negative for maximization

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params
```

---

## 📐 6. Evaluation & Results Framework

### Required Metrics Table for Paper

```python
def generate_results_table(models_dict, X_train, X_test, y_train, y_test, targets):
    """Generate comparative results table for all models."""
    results = []

    for name, model in models_dict.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        row = {'Model': name}
        for i, target in enumerate(targets):
            rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            row[f'{target}_RMSE'] = rmse
            row[f'{target}_MAE'] = mae
            row[f'{target}_R2'] = r2

        # Macro averages
        row['Macro_RMSE'] = np.mean([row[f'{t}_RMSE'] for t in targets])
        row['Macro_R2'] = np.mean([row[f'{t}_R2'] for t in targets])

        results.append(row)

    return pd.DataFrame(results).sort_values('Macro_RMSE')

# Usage
results_df = generate_results_table(MODELS, X_train, X_test, y_train, y_test, TARGETS)
print(results_df.to_markdown(index=False))  # Copy-paste into paper
```

### Model Comparison Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_comparison(results_df, targets):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RMSE comparison
    rmse_cols = [f'{t}_RMSE' for t in targets] + ['Macro_RMSE']
    results_df_melted = results_df.melt(id_vars='Model', value_vars=rmse_cols,
                                       var_name='Metric', value_name='RMSE')
    sns.barplot(data=results_df_melted, x='Model', y='RMSE', hue='Metric', ax=axes[0])
    axes[0].set_title('RMSE by Model & Target')
    axes[0].tick_params(axis='x', rotation=45)

    # R² comparison
    r2_cols = [f'{t}_R2' for t in targets] + ['Macro_R2']
    results_df_melted_r2 = results_df.melt(id_vars='Model', value_vars=r2_cols,
                                          var_name='Metric', value_name='R2')
    sns.barplot(data=results_df_melted_r2, x='Model', y='R2', hue='Metric', ax=axes[1])
    axes[1].set_title('R² by Model & Target')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
```

---

## 🔬 7. Interpretability & Scenario Analysis (Bonus Points)

### SHAP Feature Importance (Global)

```python
import shap

def explain_model(model, X_test, feature_names, targets, sample_size=1000):
    """Generate SHAP explanations for multi-target model."""
    # Use a subset for speed
    X_sample = X_test[:sample_size]

    explanations = {}
    for i, target in enumerate(targets):
        # Extract single-target estimator if wrapped
        estimator = model.estimators_[i] if hasattr(model, 'estimators_') else model

        explainer = shap.TreeExplainer(estimator) if hasattr(estimator, 'tree_') else shap.KernelExplainer(estimator.predict, X_sample)
        shap_values = explainer.shap_values(X_sample)

        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                         title=f'SHAP Importance: {target}', show=False)
        plt.savefig(f'figures/shap_{target}.png', dpi=300, bbox_inches='tight')
        plt.close()

        explanations[target] = shap_values

    return explanations
```

### Policy Scenario Simulator (Digital Twin Lite)

```python
def simulate_policy_scenario(base_features, feature_names, model, scaler,
                            gdp_change_pct=0, renewable_increase_pct=0, policy_boost=0):
    """
    Simulate impact of policy/economic changes on climate targets.

    Args:
        base_features: dict of baseline feature values (one country-year)
        feature_names: list of feature column names
        model: trained multi-target model
        scaler: fitted RobustScaler
        gdp_change_pct: % change in GDP
        renewable_increase_pct: absolute % point increase in renewable usage
        policy_boost: increase in Policy_Score (0-10 scale)

    Returns:
        dict with predicted changes in Temperature_Anomaly and CO2_Emissions
    """
    import numpy as np

    # Create baseline vector
    base_vec = np.array([[base_features.get(f, 0) for f in feature_names]])

    # Create scenario vector with modifications
    scenario_vec = base_vec.copy()
    for i, feat in enumerate(feature_names):
        if feat == 'GDP':
            scenario_vec[0, i] *= (1 + gdp_change_pct/100)
        elif feat == 'Renewable_Energy_Usage':
            scenario_vec[0, i] = min(100, scenario_vec[0, i] + renewable_increase_pct)
        elif feat == 'Policy_Score':
            scenario_vec[0, i] = min(10, scenario_vec[0, i] + policy_boost)

    # Scale both
    base_scaled = scaler.transform(base_vec)
    scenario_scaled = scaler.transform(scenario_vec)

    # Predict
    base_pred = model.predict(base_scaled)[0]
    scenario_pred = model.predict(scenario_scaled)[0]

    return {
        'baseline': dict(zip(TARGETS, base_pred)),
        'scenario': dict(zip(TARGETS, scenario_pred)),
        'delta': dict(zip(TARGETS, scenario_pred - base_pred))
    }

# Example usage
# result = simulate_policy_scenario(baseline_dict, feature_cols, best_model, scaler,
#                                  gdp_change_pct=-2, renewable_increase_pct=15, policy_boost=2)
# print(f"CO2 change: {result['delta']['CO2_Emissions']:.2f} Mt")
```

---

## 📝 8. Research Paper Structure (8–12 Pages)

```
1. Abstract (150 words)
   - Problem, methods, key findings, contribution

2. Introduction
   - Climate change urgency + socioeconomic complexity
   - Gap: lack of multi-target ML studies with interpretable drivers
   - Our contribution: (1) panel dataset pipeline, (2) comparative ML framework, (3) scenario simulator

3. Literature Review (1.5 pages)
   - Traditional econometric climate models (IPCC, DICE)
   - ML in climate science (random forests for downscaling, LSTM for forecasting)
   - Multi-target learning applications in environmental science

4. Methodology (3 pages) ★ CORE SECTION
   4.1 Data Sources & Preprocessing
       - Dataset description, variable definitions, quality checks
       - Figure: Data processing flowchart
   4.2 Feature Engineering
       - Table: Engineered features with rationale
       - Lag/rolling window methodology
   4.3 Model Architecture
       - Multi-output regression framework diagram
       - Model selection rationale (bias-variance tradeoff)
   4.4 Validation Strategy
       - TimeSeriesSplit diagram, no-leakage guarantee
       - Metrics justification (RMSE for magnitude, R² for explained variance)

5. Results (2.5 pages)
   5.1 EDA Insights
       - Fig 1-3 with captions interpreting climate-economy linkages
   5.2 Model Performance
       - Table: Comparative metrics (RMSE, MAE, R²) with std dev
       - Fig: Model comparison bar charts
   5.3 Interpretability
       - Fig: SHAP summary plots for both targets
       - Key driver identification (e.g., "Fossil_Fuel_Usage contributes 3.2× more to CO₂ prediction than Policy_Score")
   5.4 Scenario Analysis
       - Table: Policy simulation results (baseline vs. intervention)
       - Fig: Interactive-style scenario visualization

6. Discussion (1.5 pages)
   - Interpretation: Which socioeconomic levers matter most?
   - Policy implications: e.g., "Renewable transition shows stronger marginal impact than GDP moderation"
   - Limitations: temporal resolution, unobserved confounders, extrapolation risks
   - Future work: integration with satellite data, causal inference extensions

7. Conclusion (0.5 page)
   - Recap contributions + actionable takeaway

8. References (IEEE format, ≥20 sources)
   - Mix of climate science papers + ML methodology papers

Appendix (Optional)
   - Full hyperparameter grids
   - Additional country-level case studies
   - Reproducibility: requirements.txt, data access instructions
```

---

## 📦 9. Deliverables Checklist

| Deliverable         | Required Content                                                                    | Pro Tips                                                                 |
| ------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Code**            | Jupyter notebook OR modular `.py` files + `requirements.txt`                        | Use functions, add docstrings, include `set_random_state(42)` everywhere |
| **Dataset**         | Cleaned CSV + `metadata.csv` (source, units, transformations)                       | Add a `data_dictionary.pdf` explaining each column                       |
| **Paper (PDF)**     | 8–12 pages, IEEE/ACM format, figures + tables + references                          | Use LaTeX Overleaf template; export figures at 300 DPI                   |
| **Slides**          | 10–15 slides: Problem → Data → Methods → Results → Policy → Q&A                     | One key message per slide; use speaker notes for details                 |
| **Reproducibility** | `environment.yml` or `requirements.txt`, fixed random seeds, CV split indices saved | Add a `RUN_ME_FIRST.md` with setup instructions                          |

---

## 🌟 10. Bonus Optimization Strategy (+4 Marks)

Since your project is in **Climate Change**, you qualify for the bonus. Maximize points with:

### ✅ High-Impact Additions

| Strategy                             | Implementation Effort                                  | Bonus Impact |
| ------------------------------------ | ------------------------------------------------------ | ------------ |
| **Digital Twin Lite**                | Medium (Gradio/Streamlit dashboard)                    | 🔥🔥🔥🔥     |
| **Uncertainty Quantification**       | Low (add prediction intervals via quantile regression) | 🔥🔥🔥       |
| **Policy Effectiveness Metric**      | Low (SHAP + policy dummy interaction analysis)         | 🔥🔥🔥🔥     |
| **Regional Stratification Analysis** | Low (run models on OECD vs. non-OECD subsets)          | 🔥🔥🔥       |

### 🚀 Quick Win: Add Prediction Intervals

```python
from sklearn.ensemble import GradientBoostingRegressor

# Use quantile regression for uncertainty bounds
class MultiTargetQuantileRegressor:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.models_lower = {}
        self.models_median = {}
        self.models_upper = {}

    def fit(self, X, y, targets):
        for i, target in enumerate(targets):
            self.models_median[target] = MultiOutputRegressor(
                GradientBoostingRegressor(loss='ls', random_state=42)
            ).fit(X, y[:, [i]])
            self.models_lower[target] = MultiOutputRegressor(
                GradientBoostingRegressor(loss='quantile', alpha=self.alpha/2, random_state=42)
            ).fit(X, y[:, [i]])
            self.models_upper[target] = MultiOutputRegressor(
                GradientBoostingRegressor(loss='quantile', alpha=1-self.alpha/2, random_state=42)
            ).fit(X, y[:, [i]])

    def predict_with_interval(self, X, targets):
        predictions = {}
        for target in targets:
            median = self.models_median[target].predict(X).flatten()
            lower = self.models_lower[target].predict(X).flatten()
            upper = self.models_upper[target].predict(X).flatten()
            predictions[target] = {
                'median': median,
                'lower': lower,
                'upper': upper,
                'width': upper - lower
            }
        return predictions
```

### 🎨 Presentation Slide Template (Key Slides)

```
Slide 1: Title + Your Name + Course
Slide 2: The Problem (1-sentence hook + why it matters)
Slide 3: Research Questions (3 bullet points)
Slide 4: Data Snapshot (map + key stats table)
Slide 5: Methodology Flowchart (visual pipeline)
Slide 6: Key EDA Insight (1 powerful figure)
Slide 7: Model Comparison (bar chart + winner highlight)
Slide 8: What Drives Climate Change? (SHAP plot)
Slide 9: Policy Simulator Demo (before/after scenario)
Slide 10: Limitations & Future Work (honest + forward-looking)
Slide 11: Conclusion (1-sentence takeaway)
Slide 12: Q&A + GitHub Link
```

---

## 🗓️ 11. Execution Timeline (11-Day Sprint)

| Day    | Task                                                       | Output                                                         |
| ------ | ---------------------------------------------------------- | -------------------------------------------------------------- |
| **1**  | Load data, audit quality, define targets/features          | Cleaned `df_processed.csv`, feature list                       |
| **2**  | Implement preprocessing + feature engineering functions    | `preprocess.py` module, engineered feature report              |
| **3**  | EDA: generate 5 core visualizations + summary stats        | `figures/eda_*.png`, `tables/summary_stats.tex`                |
| **4**  | Implement 4 models + baseline; set up TimeSeriesSplit      | `models.py`, baseline results                                  |
| **5**  | Hyperparameter tuning (Optuna) for top 2 models            | Best params JSON, tuned model checkpoints                      |
| **6**  | Final evaluation: metrics table + model comparison plots   | `results/comparison_table.csv`, `figures/model_comparison.png` |
| **7**  | SHAP analysis + scenario simulator implementation          | `figures/shap_*.png`, interactive demo script                  |
| **8**  | Draft paper Sections 1–4 (Intro through Methodology)       | `paper_draft_v1.tex`                                           |
| **9**  | Draft paper Sections 5–7 (Results, Discussion, Conclusion) | Complete paper draft                                           |
| **10** | Build slides + finalize code documentation                 | `presentation.pptx`, `README.md`, `requirements.txt`           |
| **11** | Peer review, fix leaks, export deliverables, submit        | Final ZIP with all 4 deliverables                              |

---

## 🚀 Immediate Next Steps (Start Today)

1. **Download & Explore**

   ```python
   df = pd.read_csv('your_dataset.csv')
   print(df.info())
   print(df.groupby('Country')['Year'].count().describe())  # Check panel balance
   ```

2. **Run the Preprocessing Pipeline**  
   Copy the `prepare_climate_data()` function above and execute:

   ```python
   X_train, X_test, y_train, y_test, feats, scaler, train_df, test_df = prepare_climate_data(
       df, TARGETS, ID_COLS, test_year_start=2010
   )
   print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
   ```

3. **Baseline Model First**  
   Before complex models, validate your pipeline with Ridge regression:

   ```python
   from sklearn.linear_model import Ridge
   baseline = MultiOutputRegressor(Ridge()).fit(X_train, y_train)
   print("Baseline R²:", r2_score(y_test, baseline.predict(X_test), multioutput='uniform_average'))
   ```

4. **Save Your Environment**
   ```bash
   pip freeze > requirements.txt
   # Or for conda: conda env export > environment.yml
   ```

---

## 💡 Need Help With?

✅ I can generate for you right now:

- [ ] A complete starter Jupyter Notebook with all code above pre-loaded
- [ ] LaTeX paper template with sections pre-formatted
- [ ] Gradio dashboard code for the policy simulator
- [ ] SHAP + scenario analysis visualization scripts

**Which would accelerate your progress most?** Let me know and I'll deliver it immediately. 🎯
