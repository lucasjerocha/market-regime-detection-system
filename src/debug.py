import numpy as np
import pandas as pd
import yfinance as yf
from regime_model import build_features


df1 = yf.download('SPY', start='2000-01-01', end=None)
if df1 is None or df1.empty:
    print("Failed to download data for SPY.")
df1 = df1.dropna()

# Feature Engineering 
df = build_features(df1)
X = df.copy()

print("========== BASIC INFO ==========")
print("Shape:", X.shape)
print()

print("========== NaN CHECK ==========")
print("Total NaNs:", np.isnan(X).sum().sum())
print()

print("========== INF CHECK ==========")
print("Total +inf/-inf:", np.isinf(X).sum().sum())
print()

print("========== EXTREME VALUES ==========")
print("Max value:", np.nanmax(X.values))
print("Min value:", np.nanmin(X.values))
print()

print("========== ROWS WITH INF ==========")
inf_rows = X[np.isinf(X).any(axis=1)]
print(inf_rows.head())
print()

print("========== ROWS WITH NaN ==========")
nan_rows = X[X.isna().any(axis=1)]
print(nan_rows.head())