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

print("🚀 CNN-LSTM TOMORROW PREDICTION - Nifty 200 (1-Year Data)")

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
print(f"🎯 Predicting {len(open_cols)} stocks")

forecast_dir = "nifty200_cnn_lstm_tomorrow"
os.makedirs(forecast_dir, exist_ok=True)

seq_length = 60
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================================
# ✅ CNN-LSTM MODEL (Single Stock)
# ========================================
class CNNLSTMSingleStock(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CNN: Extract local patterns from 60-day sequence
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)
        
        # CNN output: 64 channels * 15 timesteps = 960 features
        self.cnn_features = 64 * 15
        
        # LSTM on CNN features
        self.lstm = nn.LSTM(self.cnn_features, 64, batch_first=True, dropout=0.2)
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq=60, features=1)
        batch_size = x.size(0)
        
        # CNN: (batch, 1, 60)
        x = x.transpose(1, 2)  # (batch, 1, 60)
        
        # CNN layers
        x = torch.relu(self.conv1(x))  # (batch, 32, 60)
        x = self.pool(x)               # (batch, 32, 30)
        x = torch.relu(self.conv2(x))  # (batch, 64, 30)
        x = self.pool(x)               # (batch, 64, 15)
        x = self.dropout(x)
        
        # Flatten for LSTM
        x = x.transpose(1, 2).contiguous().view(batch_size, 1, -1)  # (batch, 1, 960)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        final_features = lstm_out[:, -1, :]  # (batch, 64)
        
        return self.fc(final_features)

def predict_tomorrow_cnn_lstm(stock_data):
    """CNN-LSTM prediction for single stock"""
    if len(stock_data) < seq_length + 10:
        return None, None
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data.values.reshape(-1, 1))
    
    # Create sequences: (samples, 60, 1)
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])  # (60, 1)
        y.append(scaled_data[i])
    
    X = torch.FloatTensor(np.array(X))  # (samples, 60, 1)
    y = torch.FloatTensor(np.array(y))  # (samples, 1)
    
    # Train CNN-LSTM
    model = CNNLSTMSingleStock().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    model.train()
    for epoch in range(40):  # More epochs for CNN-LSTM
        optimizer.zero_grad()
        pred = model(X.to(device))
        loss = criterion(pred.squeeze(), y.to(device).squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Predict tomorrow
    model.eval()
    last_seq = torch.FloatTensor(scaled_data[-seq_length:]).unsqueeze(0).to(device)  # (1, 60, 1)
    
    with torch.no_grad():
        tomorrow_scaled = model(last_seq).detach().cpu().numpy()[0, 0]
    
    tomorrow_price = scaler.inverse_transform([[tomorrow_scaled]])[0, 0]
    last_price = stock_data.iloc[-1]
    
    return tomorrow_price, last_price

# ========================================
# MAIN PREDICTION LOOP
# ========================================
print("\n🔮 CNN-LSTM Tomorrow Predictions...")
results = []

for i, stock_col in enumerate(open_cols):
    stock_data = df_1year[stock_col].dropna()
    tomorrow_price, last_price = predict_tomorrow_cnn_lstm(stock_data)
    
    if tomorrow_price is not None:
        change_pct = ((tomorrow_price / last_price) - 1) * 100
        
        results.append({
            'Stock': stock_col.replace('Open_', ''),
            'Last_Close': round(last_price, 2),
            'Tomorrow_Open_CNNLSTM': round(tomorrow_price, 2),
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

# Save files
results_df.to_csv(f"{forecast_dir}/CNNLSTM_TOMORROW_NIFTY200.csv", index=False)
results_df.nlargest(10, 'Change_Pct').to_csv(f"{forecast_dir}/CNNLSTM_TOP10_GAINERS.csv", index=False)
results_df.nsmallest(10, 'Change_Pct').to_csv(f"{forecast_dir}/CNNLSTM_TOP10_LOSERS.csv", index=False)

print(f"\n🎉 CNN-LSTM COMPLETE! Tomorrow: {tomorrow_date.strftime('%Y-%m-%d')}")
print(f"📁 Files: {forecast_dir}/")

print("\n🔥 CNN-LSTM TOP 10 GAINERS:")
print(results_df.nlargest(10, 'Change_Pct')[['Stock', 'Change_Pct', 'Signal']].to_string(index=False))

print("\n📉 CNN-LSTM TOP 10 LOSERS:")
print(results_df.nsmallest(10, 'Change_Pct')[['Stock', 'Change_Pct', 'Signal']].to_string(index=False))

print(f"\n📊 CNN-LSTM SIGNALS:")
buy_count = len(results_df[results_df['Signal'] == '🟢 BUY'])
hold_count = len(results_df[results_df['Signal'] == '🟡 HOLD'])
sell_count = len(results_df[results_df['Signal'] == '🔴 SELL'])
print(f"🟢 BUY: {buy_count} | 🟡 HOLD: {hold_count} | 🔴 SELL: {sell_count}")
