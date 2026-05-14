import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Load & Clean Data
df = pd.read_csv('/Users/apple/Downloads/Summary of Weather.csv')

df['Precip'] = df['Precip'].replace('T', 0)
df['Precip'] = pd.to_numeric(df['Precip'], errors='coerce')

FEATURES = ['MaxTemp', 'MinTemp', 'Precip', 'Snowfall', 'YR', 'MO', 'DA']
TARGET = 'MeanTemp'

df = df[FEATURES + [TARGET]].copy()
df = df.apply(pd.to_numeric, errors='coerce')
df['Snowfall'] = df['Snowfall'].fillna(0)
df.dropna(inplace=True)

print(f"Clean dataset: {len(df):,} rows, {len(FEATURES)} features -> target: {TARGET}\n")

# 2. Correlation & Heatmap
corr = df.corr()
print("Correlation with MeanTemp:")
print(corr[TARGET].sort_values(ascending=False).to_string())
print()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('/Users/apple/reg-ana/correlation_heatmap.png', dpi=150)
plt.close()
print("Saved: correlation_heatmap.png")

# 3. Visualisations
# Histogram
plt.figure(figsize=(8, 5))
plt.hist(df[TARGET], bins=50, color='steelblue', edgecolor='white')
plt.title('Distribution of Mean Temperature')
plt.xlabel('MeanTemp (C)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('/Users/apple/reg-ana/histogram.png', dpi=150)
plt.close()
print("Saved: histogram.png")

# Bar plot — avg MeanTemp by month
monthly_avg = df.groupby('MO')[TARGET].mean()
plt.figure(figsize=(8, 5))
monthly_avg.plot(kind='bar', color='coral', edgecolor='white')
plt.title('Average Mean Temperature by Month')
plt.xlabel('Month')
plt.ylabel('Avg MeanTemp (C)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('/Users/apple/reg-ana/barplot.png', dpi=150)
plt.close()
print("Saved: barplot.png")

# Line chart — MeanTemp trend by year
yearly_avg = df.groupby('YR')[TARGET].mean()
plt.figure(figsize=(10, 5))
plt.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=1.5, color='teal')
plt.title('Mean Temperature Trend by Year')
plt.xlabel('Year')
plt.ylabel('Avg MeanTemp (C)')
plt.tight_layout()
plt.savefig('/Users/apple/reg-ana/linechart.png', dpi=150)
plt.close()
print("Saved: linechart.png\n")

# 4. Train / Test Split (80 / 20)
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}\n")

# 5. Build & Train Models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree':     DecisionTreeRegressor(random_state=42),
    'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
}

header = f"{'Model':<22} {'R2 Score':>10} {'MSE':>12} {'MAE':>10}"
print(header)
print('-' * len(header))

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2  = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"{name:<22} {r2:>10.4f} {mse:>12.4f} {mae:>10.4f}")

print("\nDone. All plots saved to /Users/apple/reg-ana/")
