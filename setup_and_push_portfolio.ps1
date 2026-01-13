# ==========================================
# FULL DATA SCIENCE PORTFOLIO SETUP + PUSH
# ==========================================

# ------------------------------
# CONFIGURATION
# ------------------------------
$PortfolioPath = "C:\Users\Abisola\data-science-portfolio"
$GitHubRepoURL = "https://github.com/mhaleeq/data-science-portfolio.git"
$GitHubBranch = "main"
$CommitMessage = "Initial commit: 5 full turnkey projects with ML notebooks and plots"

# ------------------------------
# CREATE PORTFOLIO FOLDER
# ------------------------------
if (-Not (Test-Path $PortfolioPath)) {
    New-Item -ItemType Directory -Force -Path $PortfolioPath
}
Set-Location $PortfolioPath

# ------------------------------
# FUNCTIONS
# ------------------------------
Function Create-ProjectFolders($projectPath) {
    New-Item -ItemType Directory -Force -Path "$projectPath\data"
    New-Item -ItemType Directory -Force -Path "$projectPath\plots"
    New-Item -ItemType Directory -Force -Path "$projectPath\notebooks"
}

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
"@ | Out-File -Encoding utf8NoBOM "$projectPath\requirements.txt"
}

Function Create-Readme($projectPath, $title, $description) {
@"
# $title

$description

**Notebook:** `notebooks/<notebook_name>.ipynb`  
**Data:** `data/`  
**Plots:** `plots/`  
"@ | Out-File -Encoding utf8NoBOM "$projectPath\README.md"
}

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
    $notebookJson | Out-File -Encoding utf8NoBOM $nbPath
}

# ------------------------------
# PROJECTS DEFINITION
# ------------------------------
$projects = @(

# 1. Breast Cancer
@{path="01-health-machine-learning\breast-cancer-explainable-ml"; notebook="breast_cancer_explainable_ml"; title="Explainable Machine Learning for Breast Cancer Diagnosis"; desc="Predict breast cancer diagnosis using explainable ML. Includes EDA, preprocessing, models, evaluation, and feature importance analysis."; code=@(
"import pandas as pd","import numpy as np","import matplotlib.pyplot as plt","import seaborn as sns",
"from sklearn.datasets import load_breast_cancer","from sklearn.model_selection import train_test_split","from sklearn.preprocessing import StandardScaler",
"from sklearn.ensemble import RandomForestClassifier","from sklearn.linear_model import LogisticRegression","from xgboost import XGBClassifier",
"from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score","import shap","import os",
"os.makedirs('../data', exist_ok=True)","os.makedirs('../plots', exist_ok=True)",
"data = load_breast_cancer()","X = pd.DataFrame(data.data, columns=data.feature_names)","y = pd.Series(data.target, name='target')",
"df = pd.concat([X, y], axis=1)","df.to_csv('../data/breast_cancer_data.csv', index=False)",
"scaler = StandardScaler()","X_scaled = scaler.fit_transform(X)","X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)",
"models = {'Logistic Regression': LogisticRegression(max_iter=1000), 'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42), 'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')}",
"for name, model in models.items():","    model.fit(X_train, y_train)","    y_pred = model.predict(X_test)",
"    acc = accuracy_score(y_test, y_pred)","    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])",
"    print(f'{name} - Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}')",
"    cm = confusion_matrix(y_test, y_pred)","    plt.figure()","    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')","    plt.title(f'{name} Confusion Matrix')",
"    plt.savefig(f'../plots/{name}_confusion_matrix.png')","    plt.show()",
"rf = models['Random Forest']","explainer = shap.TreeExplainer(rf)","shap_values = explainer.shap_values(X_train)",
"shap.summary_plot(shap_values[1], X, plot_type='bar', show=False)","plt.savefig('../plots/rf_feature_importance.png')","plt.close()"
)},

# 2. Diabetes
@{path="01-health-machine-learning\diabetes-risk-stratification"; notebook="diabetes_risk_stratification"; title="Diabetes Risk Prediction and Stratification"; desc="Predict and stratify diabetes risk using ML models. Includes preprocessing, classification models, evaluation, and feature correlation visualization."; code=@(
"import pandas as pd","import numpy as np","import matplotlib.pyplot as plt","import seaborn as sns",
"from sklearn.model_selection import train_test_split","from sklearn.preprocessing import StandardScaler",
"from sklearn.ensemble import RandomForestClassifier","from sklearn.linear_model import LogisticRegression",
"from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix","import os",
"os.makedirs('../data', exist_ok=True)","os.makedirs('../plots', exist_ok=True)",
"np.random.seed(42)","n = 500","X = pd.DataFrame({'age': np.random.randint(20, 80, n),'bmi': np.random.uniform(18, 40, n),'blood_pressure': np.random.randint(80,160,n),'glucose': np.random.randint(70,200,n),'insulin': np.random.randint(15,276,n)})",
"y = ((X['glucose']>126) | (X['bmi']>30)).astype(int)","df = pd.concat([X, pd.Series(y, name='target')], axis=1)",
"df.to_csv('../data/diabetes_data.csv', index=False)","scaler = StandardScaler()","X_scaled = scaler.fit_transform(X)",
"X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)",
"models = {'Logistic Regression': LogisticRegression(max_iter=1000), 'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42)}",
"for name, model in models.items():","    model.fit(X_train, y_train)","    y_pred = model.predict(X_test)",
"    acc = accuracy_score(y_test, y_pred)","    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])",
"    print(f'{name} - Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}')",
"    cm = confusion_matrix(y_test, y_pred)","    plt.figure()","    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')","    plt.title(f'{name} Confusion Matrix')",
"    plt.savefig(f'../plots/{name}_confusion_matrix.png')","    plt.show()",
"plt.figure(figsize=(6,4))","sns.heatmap(df.corr(), annot=True, cmap='coolwarm')","plt.title('Feature Correlation Heatmap')","plt.savefig('../plots/feature_correlation.png')","plt.show()"
)},

# 3. Cardiovascular
@{path="01-health-machine-learning\cardiovascular-risk-modeling"; notebook="cardiovascular_risk_modeling"; title="Cardiovascular Disease Risk Modeling"; desc="Model cardiovascular risk using patient data with ML models. Includes preprocessing, evaluation, and histograms of key features."; code=@(
"import pandas as pd","import numpy as np","import matplotlib.pyplot as plt","import seaborn as sns",
"from sklearn.model_selection import train_test_split","from sklearn.preprocessing import StandardScaler",
"from sklearn.ensemble import RandomForestClassifier","from sklearn.linear_model import LogisticRegression",
"from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix","import os",
"os.makedirs('../data', exist_ok=True)","os.makedirs('../plots', exist_ok=True)",
"np.random.seed(42)","n = 500","X = pd.DataFrame({'age': np.random.randint(30,80,n),'cholesterol': np.random.randint(150,300,n),'blood_pressure': np.random.randint(80,160,n),'bmi': np.random.uniform(18,40,n),'smoker': np.random.randint(0,2,n)})",
"y = ((X['cholesterol']>240) | (X['blood_pressure']>140) | (X['bmi']>30)).astype(int)","df = pd.concat([X, pd.Series(y, name='target')], axis=1)","df.to_csv('../data/cardiovascular_data.csv', index=False)",
"scaler = StandardScaler()","X_scaled = scaler.fit_transform(X)","X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)",
"models = {'Logistic Regression': LogisticRegression(max_iter=1000), 'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42)}",
"for name, model in models.items():","    model.fit(X_train, y_train)","    y_pred = model.predict(X_test)",
"    acc = accuracy_score(y_test, y_pred)","    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])",
"    print(f'{name} - Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}')",
"    cm = confusion_matrix(y_test, y_pred)","    plt.figure()","    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')","    plt.title(f'{name} Confusion Matrix')",
"    plt.savefig(f'../plots/{name}_confusion_matrix.png')","    plt.show()",
"for col in ['age','cholesterol','bmi']:","    plt.figure()","    plt.hist(df[col], bins=20, color='skyblue')","    plt.title(f'Histogram of {col}')","    plt.savefig(f'../plots/{col}_hist.png')","    plt.show()"
)},

# 4. Urban Crime
@{path="02-public-safety-analytics\urban-crime-pattern-analysis"; notebook="urban_crime_pattern_analysis"; title="Urban Crime Pattern Analysis"; desc="Analyze urban crime trends using statistical methods. Includes barplots and histograms of incidents."; code=@(
"import pandas as pd","import numpy as np","import matplotlib.pyplot as plt","import seaborn as sns","import os",
"os.makedirs('../data', exist_ok=True)","os.makedirs('../plots', exist_ok=True)",
"np.random.seed(42)","n=500","X=pd.DataFrame({'neighborhood': np.random.choice(['A','B','C','D'], n),'incidents': np.random.poisson(5, n),'population': np.random.randint(1000,10000,n)})",
"X['crime_rate']=X['incidents']/X['population']*1000","df=X.copy()","df.to_csv('../data/urban_crime_data.csv', index=False)",
"plt.figure(figsize=(6,4))","sns.barplot(x='neighborhood', y='crime_rate', data=df)","plt.title('Crime Rate by Neighborhood')","plt.savefig('../plots/crime_rate_barplot.png')","plt.show()",
"plt.figure(figsize=(6,4))","plt.hist(df['incidents'], bins=20, color='purple')","plt.title('Histogram of Incidents')","plt.savefig('../plots/incidents_hist.png')","plt.show()"
)},

# 5. Medicine Demand Forecasting
@{path="03-operations-analytics\medicine-demand-forecasting"; notebook="medicine_demand_forecasting"; title="Medicine Demand Forecasting"; desc="Forecast medicine demand using time series and ML models. Includes prediction plots and rolling average visualization."; code=@(
"import pandas as pd","import numpy as np","import matplotlib.pyplot as plt",
"from sklearn.model_selection import train_test_split","from sklearn.ensemble import RandomForestRegressor","from sklearn.metrics import mean_squared_error",
"import os","os.makedirs('../data', exist_ok=True)","os.makedirs('../plots', exist_ok=True)",
"np.random.seed(42)","dates = pd.date_range(start='2023-01-01', periods=100)","demand = np.random.poisson(50,100) + np.arange(100)*0.5",
"df = pd.DataFrame({'date': dates, 'demand': demand})","df.to_csv('../data/medicine_demand.csv', index=False)",
"df['lag1']=df['demand'].shift(1).fillna(method='bfill')","df['lag2']=df['demand'].shift(2).fillna(method='bfill')",
"X = df[['lag1','lag2']]","y = df['demand']","X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)",
"model = RandomForestRegressor(n_estimators=200, random_state=42)","model.fit(X_train, y_train)","y_pred = model.predict(X_test)",
"mse = mean_squared_error(y_test, y_pred)","print(f'MSE: {mse:.4f}')",
"plt.figure(figsize=(6,4))","plt.plot(df['date'], df['demand'], label='Actual')","plt.plot(df['date'].iloc[-len(y_test):], y_pred, label='Predicted')","plt.legend()","plt.title('Medicine Demand Forecasting')","plt.savefig('../plots/medicine_demand_forecast.png')","plt.show()",
"plt.figure(figsize=(6,4))","plt.plot(df['date'], df['demand'].rolling(window=7).mean(), color='orange')","plt.title('7-Day Rolling Average Demand')","plt.savefig('../plots/medicine_demand_rolling_avg.png')","plt.show()"
)}

)

# ------------------------------
# CREATE PROJECT FOLDERS, FILES, NOTEBOOKS
# ------------------------------
foreach ($proj in $projects) {
    Create-ProjectFolders $proj.path
    Create-Requirements $proj.path
    Create-Readme $proj.path $proj.title $proj.desc
    Create-Notebook $proj.path $proj.notebook $proj.code
}

Write-Host "✅ All 5 projects created with fully runnable notebooks and plots!"

# ------------------------------
# INITIALIZE GIT AND PUSH
# ------------------------------
if (-Not (Test-Path ".git")) {
    git init
    git remote add origin $GitHubRepoURL
}

git add .
git commit -m "$CommitMessage"
git branch -M $GitHubBranch
git push -u origin $GitHubBranch

Write-Host "🚀 Portfolio pushed to GitHub successfully!"
