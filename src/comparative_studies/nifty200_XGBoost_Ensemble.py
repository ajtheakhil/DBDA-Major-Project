import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import os
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("🚀 XGBoost + Ensemble Tomorrow Prediction - Nifty 200 (1-Year Data)")
print("📦 pip install xgboost scikit-learn pandas numpy")

# Load data
df = pd.read_csv("nifty200_data/nifty200_complete.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# 1-Year data
one_year_ago = df.index[-252]
df_1year = df.loc[one_year_ago:]
print(f"📊 1-Year data: {len(df_1year)} trading days")

open_cols = [col for col in df_1year.columns if col.startswith('Open_')]
print(f"🎯 Processing {len(open_cols)} stocks")

forecast_dir = "nifty200_xgboost_ensemble"
os.makedirs(forecast_dir, exist_ok=True)

tomorrow_date = df_1year.index[-1] + timedelta(days=1)

def create_features(stock_data):
    """Create lag features + technical indicators"""
    df_feat = pd.DataFrame(index=stock_data.index)
    df_feat['price'] = stock_data
    
    # Lag features (1-5 days)
    for lag in range(1, 6):
        df_feat[f'lag_{lag}'] = df_feat['price'].shift(lag)
    
    # Rolling statistics (5, 10, 20 days)
    df_feat['ma_5'] = df_feat['price'].rolling(5).mean()
    df_feat['ma_10'] = df_feat['price'].rolling(10).mean()
    df_feat['std_5'] = df_feat['price'].rolling(5).std()
    
    # RSI (simple 14-day)
    delta = df_feat['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df_feat['rsi'] = 100 - (100 / (1 + rs))
    
    # Returns
    df_feat['return_1d'] = df_feat['price'].pct_change()
    df_feat['return_5d'] = df_feat['price'].pct_change(5)
    
    return df_feat.dropna()

def train_xgboost_ensemble(X_train, y_train):
    """XGBoost + RF + GBR ensemble"""
    # XGBoost
    xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    # Gradient Boosting
    gbr = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
    gbr.fit(X_train, y_train)
    
    return xgb, rf, gbr

def predict_tomorrow_ensemble(stock_data):
    """Predict tomorrow using ensemble"""
    if len(stock_data) < 100:
        return None, None
    
    # Create features
    df_feat = create_features(stock_data)
    
    feature_cols = [col for col in df_feat.columns if col != 'price']
    X = df_feat[feature_cols]
    y = df_feat['price']
    
    # Time series split (80% train)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train ensemble
    xgb, rf, gbr = train_xgboost_ensemble(X_train, y_train)
    
    # Predict tomorrow using latest features
    X_tomorrow = X.iloc[-1:].values
    xgb_pred = xgb.predict(X_tomorrow)[0]
    rf_pred = rf.predict(X_tomorrow)[0]
    gbr_pred = gbr.predict(X_tomorrow)[0]
    
    # Ensemble average
    tomorrow_price = (xgb_pred + rf_pred + gbr_pred) / 3
    last_price = stock_data.iloc[-1]
    
    return tomorrow_price, last_price

# ========================================
# MAIN PREDICTION LOOP
# ========================================
print("\n🔮 XGBoost Ensemble Predictions...")
results = []

for i, stock_col in enumerate(open_cols[:100]):  # Limit to 100 for speed
    print(f"[{i+1:3d}] {stock_col}", end=" ")
    
    stock_data = df_1year[stock_col].dropna()
    tomorrow_price, last_price = predict_tomorrow_ensemble(stock_data)
    
    if tomorrow_price is not None:
        change_pct = ((tomorrow_price / last_price) - 1) * 100
        
        results.append({
            'Stock': stock_col.replace('Open_', ''),
            'Last_Close': round(last_price, 2),
            'Tomorrow_XGB_Ensemble': round(tomorrow_price,
