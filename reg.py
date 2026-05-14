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

# ── 1. Load & Clean ──────────────────────────────────────────────────────────
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

# ── 2. Correlation & Heatmap ─────────────────────────────────────────────────
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
