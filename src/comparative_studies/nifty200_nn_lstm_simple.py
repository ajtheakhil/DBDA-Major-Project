import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("🚀 TOMORROW PREDICTION - Nifty 200 (1-Year Data) - FIXED")

# Load data
df = pd.read_csv("nifty200_data/nifty200_complete.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# 1-Year data (~252 trading days)
one_year_ago = df.index[-252]
df_1year = df.loc[one_year_ago:]
print(f"📊 1-Year data: {len(df_1year)} days")

open_cols = [col for col in df_1year.columns if col.endswith('_Open')]
print(f"🎯 {len(open_cols)} stocks")

forecast_dir = "nifty200_lstm_fixed"
os.makedirs(forecast_dir, exist_ok=True)

seq_length = 60
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ FIXED LSTM MODEL
class SingleStockLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_length, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Last timestep

def predict_tomorrow(stock_data):
    """Train + predict tomorrow"""
    if len(stock_data) < seq_length + 10:
        return None, None
    
    # Scale
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data.values.reshape(-1, 1))
    
    # ✅ FIXED: Create sequences (batch, seq_len, features=1)
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])  # Shape: (60, 1)
        y.append(scaled_data[i])
    
    X = torch.FloatTensor(np.array(X))  # (samples, 60, 1)
    y = torch.FloatTensor(np.array(y))  # (samples, 1)
    
    # Train
    model = SingleStockLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    
    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        pred = model(X.to(device))
        loss = criterion(pred.squeeze(), y.to(device).squeeze())
        loss.backward()
        optimizer.step()
    
    # Predict tomorrow ✅ FIXED SHAPE
    model.eval()
    last_seq = torch.FloatTensor(scaled_data[-seq_length:]).unsqueeze(0).to(device)  # (1, 60, 1)
    
    with torch.no_grad():
        tomorrow_scaled = model(last_seq).cpu().numpy()[0, 0]
    
    tomorrow_price = scaler.inverse_transform([[tomorrow_scaled]])[0, 0]
    last_price = stock_data.iloc[-1]
    
    return tomorrow_price, last_price

# ========================================
# MAIN PREDICTION LOOP
# ========================================
print("\n🔮 Predicting TOMORROW for all stocks...")
results = []

for i, stock_col in enumerate(open_cols):
    stock_data = df_1year[stock_col].dropna()
    tomorrow_price, last_price = predict_tomorrow(stock_data)
    
    if tomorrow_price is not None:
        change_pct = ((tomorrow_price / last_price) - 1) * 100
        
        results.append({
            'Stock': stock_col.replace('Open_', ''),
            'Last_Close': round(last_price, 2),
            'Tomorrow_Open': round(tomorrow_price, 2),
            'Change_Pct': round(change_pct, 2),
            'Signal': '🟢 BUY' if change_pct > 1.0 else '🟡 HOLD' if change_pct > -1.0 else '🔴 SELL'
        })
    
    if (i + 1) % 25 == 0:
        print(f"Processed {i+1}/{len(open_cols)} stocks")

# ========================================
# SAVE RESULTS
# ========================================
tomorrow_date = df_1year.index[-1] + timedelta(days=1)
results_df = pd.DataFrame(results)
results_df['Forecast_Date'] = tomorrow_date.strftime('%Y-%m-%d')

# Save all files
results_df.to_csv(f"{forecast_dir}/TOMORROW_NIFTY200.csv", index=False)
results_df.nlargest(10, 'Change_Pct').to_csv(f"{forecast_dir}/TOP10_GAINERS.csv", index=False)
results_df.nsmallest(10, 'Change_Pct').to_csv(f"{forecast_dir}/TOP10_LOSERS.csv", index=False)

print(f"\n🎉 SUCCESS! Tomorrow: {tomorrow_date.strftime('%Y-%m-%d')}")
print(f"📁 Files saved to: {forecast_dir}/")

print("\n🔥 TOP 10 GAINERS:")
print(results_df.nlargest(10, 'Change_Pct')[['Stock', 'Change_Pct', 'Signal']].to_string(index=False))

print("\n📉 TOP 10 LOSERS:")
print(results_df.nsmallest(10, 'Change_Pct')[['Stock', 'Change_Pct', 'Signal']].to_string(index=False))

print(f"\n📊 SIGNALS: 🟢BUY={len(results_df[results_df.Signal.str.contains('BUY')])} | 🟡HOLD={len(results_df[results_df.Signal.str.contains('HOLD')])} | 🔴SELL={len(results_df[results_df.Signal.str.contains('SELL')])}")
