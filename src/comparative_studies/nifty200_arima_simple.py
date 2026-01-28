import pandas as pd
import numpy as np
from pmdarima import auto_arima
import os
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("🚀 ARIMA TOMORROW PREDICTION - Nifty 200 (1-Year Data)")
print("📦 Install: pip install pmdarima")

# Load data
df = pd.read_csv("nifty200_data/nifty200_complete.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# Filter 1-year data (~252 trading days)
one_year_ago = df.index[-252]
df_1year = df.loc[one_year_ago:]
print(f"📊 1-Year data: {len(df_1year)} trading days")

open_cols = [col for col in df_1year.columns if col.startswith('Open_')]
print(f"🎯 Processing {len(open_cols)} stocks")

# Output directory
forecast_dir = "nifty200_arima_tomorrow"
os.makedirs(forecast_dir, exist_ok=True)

tomorrow_date = df_1year.index[-1] + timedelta(days=1)
results = []

print("\n🔮 ARIMA Auto-Fitting + Tomorrow Predictions...")

for i, stock_col in enumerate(open_cols):
    stock_data = df_1year[stock_col].dropna()
    
    if len(stock_data) < 100:  # ARIMA needs sufficient data
        continue
    
    try:
        # Auto-ARIMA (automatically finds best p,d,q)
        model = auto_arima(
            stock_data.values,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            seasonal=False,  # Daily data
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            maxiter=50,
            trace=False
        )
        
        # Predict tomorrow (1 step ahead)
        tomorrow_price = model.predict(n_periods=1)[0]
        last_price = stock_data.iloc[-1]
        
        change_pct = ((tomorrow_price / last_price) - 1) * 100
        
        results.append({
            'Stock': stock_col.replace('Open_', ''),
            'Last_Close': round(last_price, 2),
            'Tomorrow_Open_ARIMA': round(tomorrow_price, 2),
            'Change_Pct': round(change_pct, 2),
            'ARIMA_Order': f"({model.order[0]},{model.order[1]},{model.order[2]})",
            'Signal': '🟢 BUY' if change_pct > 1.0 else '🟡 HOLD' if change_pct > -1.0 else '🔴 SELL',
            'AIC': round(model.aic(), 2)
        })
        
    except Exception as e:
        print(f"❌ {stock_col}: {str(e)[:40]}")
        continue
    
    if (i + 1) % 25 == 0:
        print(f"Processed {i+1}/{len(open_cols)} stocks")

# ========================================
# SAVE RESULTS
# ========================================
results_df = pd.DataFrame(results)
results_df['Forecast_Date'] = tomorrow_date.strftime('%Y-%m-%d')

# MAIN FILES
results_df.to_csv(f"{forecast_dir}/ARIMA_TOMORROW_NIFTY200.csv", index=False)
results_df.nlargest(15, 'Change_Pct').to_csv(f"{forecast_dir}/ARIMA_TOP15_GAINERS.csv", index=False)
results_df.nsmallest(15, 'Change_Pct').to_csv(f"{forecast_dir}/ARIMA_TOP15_LOSERS.csv", index=False)

# SIGNAL SUMMARY
signals_summary = results_df['Signal'].value_counts()
print(f"\n🎉 ARIMA COMPLETE! Tomorrow: {tomorrow_date.strftime('%Y-%m-%d')}")
print(f"📁 Files saved: {forecast_dir}/")
print(f"📊 Stocks predicted: {len(results_df)}")

print("\n🔥 ARIMA TOP 10 GAINERS:")
print(results_df.nlargest(10, 'Change_Pct')[['Stock', 'Tomorrow_Open_ARIMA', 'Change_Pct', 'Signal', 'ARIMA_Order']].to_string(index=False))

print("\n📉 ARIMA TOP 10 LOSERS:")
print(results_df.nsmallest(10, 'Change_Pct')[['Stock', 'Tomorrow_Open_ARIMA', 'Change_Pct', 'Signal', 'ARIMA_Order']].to_string(index=False))

print(f"\n📈 ARIMA SIGNALS:")
for signal, count in signals_summary.items():
    print(f"{signal}: {count}")

print(f"\n⚙️  ARIMA Parameters Example:")
print(results_df[['Stock', 'ARIMA_Order', 'AIC']].head().to_string(index=False))
