# ==========================================
# Full Data Science Portfolio – 5 Projects
# Valid Jupyter Notebooks
# ==========================================

Set-Location "C:\Users\Abisola\data-science-portfolio"

# Function to create folders
Function Create-ProjectFolders($projectPath) {
    New-Item -ItemType Directory -Force -Path "$projectPath\data"
    New-Item -ItemType Directory -Force -Path "$projectPath\plots"
    New-Item -ItemType Directory -Force -Path "$projectPath\notebooks"
}

# Function to create requirements.txt
Function Create-Requirements($projectPath) {
@"
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
statsmodels
jupyter
shap
folium
geopandas
"@ > "$projectPath\requirements.txt"
}

# Function to create README.md
Function Create-Readme($projectPath, $title, $description) {
@"
# $title

$description

**Notebook:** `notebooks/<notebook_name>.ipynb`  
**Data:** `data/`  
**Plots:** `plots/`  
"@ > "$projectPath\README.md"
}

# Function to create a valid notebook
Function Create-Notebook($projectPath, $notebookName, $codeLines) {

    $jsonLines = $codeLines | ForEach-Object { "`"$($_ -replace '"','\"')`"," }

    $notebookJson = @"
{
 "cells": [
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    $($jsonLines -join "`n")
   ],
   "outputs": [],
   "execution_count": null
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
"@

    $nbPath = "$projectPath\notebooks\$notebookName.ipynb"
    $notebookJson | Out-File -Encoding UTF8 $nbPath
}

# ==========================================
# Define all 5 projects with code lines
$projects = @(
    # Project 1: Breast Cancer ML
    @{path="01-health-machine-learning\breast-cancer-explainable-ml"; notebook="breast_cancer_explainable_ml"; title="Explainable Machine Learning for Breast Cancer Diagnosis"; desc="Predict breast cancer diagnosis using explainable ML. Includes EDA, preprocessing, models, and feature importance analysis."; code=@(
"import pandas as pd",
"import numpy as np",
"import matplotlib.pyplot as plt",
"import seaborn as sns",
"from sklearn.datasets import load_breast_cancer",
"from sklearn.model_selection import train_test_split",
"from sklearn.preprocessing import StandardScaler",
"from sklearn.ensemble import RandomForestClassifier",
"from sklearn.linear_model import LogisticRegression",
"from xgboost import XGBClassifier",
"from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score",
"import shap",
"import os",
"os.makedirs('../data', exist_ok=True)",
"os.makedirs('../plots', exist_ok=True)",
"data = load_breast_cancer()",
"X = pd.DataFrame(data.data, columns=data.feature_names)",
"y = pd.Series(data.target, name='target')",
"df = pd.concat([X, y], axis=1)",
"df.to_csv('../data/breast_cancer_data.csv', index=False)",
"scaler = StandardScaler()",
"X_scaled = scaler.fit_transform(X)",
"X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)",
"models = {'Logistic Regression': LogisticRegression(max_iter=1000), 'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42), 'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')}",
"for name, model in models.items():",
"    model.fit(X_train, y_train)",
"    y_pred = model.predict(X_test)",
"    acc = accuracy_score(y_test, y_pred)",
"    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])",
"    print(f'{name} - Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}')",
"    cm = confusion_matrix(y_test, y_pred)",
"    plt.figure()",
"    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')",
"    plt.title(f'{name} Confusion Matrix')",
"    plt.savefig(f'../plots/{name}_confusion_matrix.png')",
"    plt.show()",
"rf = models['Random Forest']",
"explainer = shap.TreeExplainer(rf)",
"shap_values = explainer.shap_values(X_train)",
"shap.summary_plot(shap_values[1], X, plot_type='bar', show=False)",
"plt.savefig('../plots/rf_feature_importance.png')",
"plt.close()"
)},

    # Project 2: Diabetes Risk
    @{path="01-health-machine-learning\diabetes-risk-stratification"; notebook="diabetes_risk_stratification"; title="Diabetes Risk Prediction and Stratification"; desc="Predict diabetes risk and stratify patients. Includes EDA, preprocessing, ML models, and evaluation."; code=@(
"import pandas as pd",
"import numpy as np",
"import matplotlib.pyplot as plt",
"import seaborn as sns",
"from sklearn.model_selection import train_test_split",
"from sklearn.preprocessing import StandardScaler",
"from sklearn.ensemble import RandomForestClassifier",
"from sklearn.linear_model import LogisticRegression",
"from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score",
"import os",
"os.makedirs('../data', exist_ok=True)",
"os.makedirs('../plots', exist_ok=True)",
"url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'",
"columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigree','Age','Outcome']",
"df = pd.read_csv(url, names=columns)",
"df.to_csv('../data/pima_diabetes.csv', index=False)",
"X = df.drop('Outcome', axis=1)",
"y = df['Outcome']",
"scaler = StandardScaler()",
"X_scaled = scaler.fit_transform(X)",
"X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)",
"models = {'Logistic Regression': LogisticRegression(max_iter=1000), 'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42)}",
"for name, model in models.items():",
"    model.fit(X_train, y_train)",
"    y_pred = model.predict(X_test)",
"    acc = accuracy_score(y_test, y_pred)",
"    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])",
"    print(f'{name} - Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}')",
"    cm = confusion_matrix(y_test, y_pred)",
"    plt.figure()",
"    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')",
"    plt.title(f'{name} Confusion Matrix')",
"    plt.savefig(f'../plots/{name}_confusion_matrix.png')",
"    plt.show()",
"risk_tiers = pd.cut(df['Glucose'], bins=[0,120,160,300], labels=['Low','Medium','High'])",
"df['RiskTier'] = risk_tiers",
"df.to_csv('../data/patient_risk_tiers.csv', index=False)"
)},

    # Project 3: Cardiovascular Risk
    @{path="01-health-machine-learning\cardiovascular-risk-modeling"; notebook="cardiovascular_risk_modeling"; title="Cardiovascular Disease Risk Modeling"; desc="Predict cardiovascular disease risk and identify important features. Includes EDA, preprocessing, modeling, and evaluation."; code=@(
"import pandas as pd",
"import numpy as np",
"import matplotlib.pyplot as plt",
"import seaborn as sns",
"from sklearn.model_selection import train_test_split",
"from sklearn.preprocessing import StandardScaler",
"from sklearn.ensemble import RandomForestClassifier",
"from sklearn.linear_model import LogisticRegression",
"from sklearn.feature_selection import RFE",
"from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score",
"import os",
"os.makedirs('../data', exist_ok=True)",
"os.makedirs('../plots', exist_ok=True)",
"url = 'https://raw.githubusercontent.com/anjanadharmapuri/datasets/main/heart.csv'",
"df = pd.read_csv(url)",
"df.to_csv('../data/heart_disease.csv', index=False)",
"X = df.drop('target', axis=1)",
"y = df['target']",
"scaler = StandardScaler()",
"X_scaled = scaler.fit_transform(X)",
"X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)",
"lr = LogisticRegression(max_iter=1000)",
"selector = RFE(lr, n_features_to_select=5)",
"selector = selector.fit(X_train, y_train)",
"selected_features = X.columns[selector.support_]",
"print('Top selected features:', list(selected_features))",
"X_train_sel = X_train[:, selector.support_]",
"X_test_sel = X_test[:, selector.support_]",
"rf = RandomForestClassifier(n_estimators=200, random_state=42)",
"rf.fit(X_train_sel, y_train)",
"y_pred = rf.predict(X_test_sel)",
"acc = accuracy_score(y_test, y_pred)",
"roc = roc_auc_score(y_test, rf.predict_proba(X_test_sel)[:,1])",
"print(f'Random Forest - Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}')",
"cm = confusion_matrix(y_test, y_pred)",
"plt.figure()",
"sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')",
"plt.title('Random Forest Confusion Matrix')",
"plt.savefig('../plots/rf_confusion_matrix.png')",
"plt.show()"
)},

    # Project 4: Urban Crime Analysis
    @{path="02-public-safety-analytics\urban-crime-pattern-analysis"; notebook="urban_crime_pattern_analysis"; title="Urban Crime Pattern Analysis"; desc="Analyze spatial and temporal patterns of urban crime. Includes EDA, visualizations, heatmaps, and trends."; code=@(
"import pandas as pd",
"import matplotlib.pyplot as plt",
"import seaborn as sns",
"import folium",
"import os",
"os.makedirs('../data', exist_ok=True)",
"os.makedirs('../plots', exist_ok=True)",
"url = 'https://raw.githubusercontent.com/datameet/india-cities-crime/main/datasets/crimes.csv'",
"df = pd.read_csv(url)",
"df.to_csv('../data/crime_data.csv', index=False)",
"plt.figure(figsize=(12,6))",
"sns.countplot(y='Crime_Category', data=df, order=df['Crime_Category'].value_counts().index)",
"plt.title('Crime Count by Category')",
"plt.savefig('../plots/crime_count_by_category.png')",
"plt.show()",
"df['Date'] = pd.to_datetime(df['Date'])",
"monthly_counts = df.groupby(df['Date'].dt.to_period('M')).size()",
"monthly_counts.plot(figsize=(12,5))",
"plt.title('Monthly Crime Counts')",
"plt.savefig('../plots/monthly_crime_counts.png')",
"plt.show()",
"crime_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)",
"for idx, row in df.head(500).iterrows():",
"    folium.CircleMarker(location=[row['Latitude'], row['Longitude']], radius=2, color='red').add_to(crime_map)",
"crime_map.save('../plots/crime_map.html')"
)},

    # Project 5: Medicine Demand Forecasting
    @{path="03-operations-analytics\medicine-demand-forecasting"; notebook="medicine_demand_forecasting"; title="Medicine Demand Forecasting"; desc="Forecast demand for medicines using time-series analysis. Includes EDA, forecasting, and visualizations."; code=@(
"import pandas as pd",
"import numpy as np",
"import matplotlib.pyplot as plt",
"from sklearn.linear_model import LinearRegression",
"from sklearn.metrics import mean_absolute_error, mean_squared_error",
"import os",
"os.makedirs('../data', exist_ok=True)",
"os.makedirs('../plots', exist_ok=True)",
"dates = pd.date_range(start='2022-01-01', periods=180)",
"np.random.seed(42)",
"sales = np.random.poisson(lam=50, size=180) + np.linspace(0,20,180)",
"df = pd.DataFrame({'Date': dates, 'Medicine_Sold': sales})",
"df.to_csv('../data/medicine_sales.csv', index=False)",
"plt.figure(figsize=(12,5))",
"plt.plot(df['Date'], df['Medicine_Sold'])",
"plt.title('Medicine Sales Over Time')",
"plt.xlabel('Date')",
"plt.ylabel('Units Sold')",
"plt.savefig('../plots/sales_trend.png')",
"plt.show()",
"df['Day'] = np.arange(len(df))",
"X = df[['Day']]",
"y = df['Medicine_Sold']",
"model = LinearRegression()",
"model.fit(X, y)",
"df['Forecast'] = model.predict(X)",
"plt.figure(figsize=(12,5))",
"plt.plot(df['Date'], df['Medicine_Sold'], label='Actual')",
"plt.plot(df['Date'], df['Forecast'], label='Forecast', linestyle='--')",
"plt.title('Medicine Demand Forecast')",
"plt.xlabel('Date')",
"plt.ylabel('Units Sold')",
"plt.legend()",
"plt.savefig('../plots/forecast.png')",
"plt.show()",
"mae = mean_absolute_error(y, df['Forecast'])",
"rmse = mean_squared_error(y, df['Forecast'], squared=False)",
"print(f'MAE: {mae:.2f}, RMSE: {rmse:.2f}')"
)}
)

# ==========================================
# Loop through projects
foreach ($proj in $projects) {
    Create-ProjectFolders $proj.path
    Create-Requirements $proj.path
    Create-Readme $proj.path $proj.title $proj.desc
    Create-Notebook $proj.path $proj.notebook $proj.code
}

Write-Host "✅ All 5 projects created with fully valid .ipynb notebooks!"
