import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error
import subprocess
import io
import warnings

warnings.filterwarnings('ignore')

print("🚀 ARIMA PIPELINE - WITH BACKTEST ACCURACY")

# 1. LOAD DATA
hdfs_path = "/user/hdoop/nifty_project/raw/nifty_raw.csv"
try:
    cat = subprocess.Popen(["hdfs", "dfs", "-cat", hdfs_path], stdout=subprocess.PIPE)
    df = pd.read_csv(io.BytesIO(cat.communicate()[0]))
except Exception as e:
    print(f"❌ Load Error: {e}"); exit()

# 2. MAP POSITIONS
target_indices = [i for i, col in enumerate(df.columns) if col.endswith('_Close') and 'Adj' not in col]
results = []

# 3. ANALYSIS LOOP WITH VALIDATION
for i, col_idx in enumerate(target_indices):
    stock_name = df.columns[col_idx]
    print(f"[{i+1}/{len(target_indices)}] Validating: {stock_name}...", end=" ", flush=True)
    
    try:
        series = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').ffill().dropna().values
        
        if len(series) < 110:
            print("⏭️ Short"); continue
            
        # --- BACKTEST LOGIC ---
        # We split the data: Train on everything EXCEPT the last 5 days
        train = series[:-5]
        actual_last_5 = series[-5:]
        
        # Train model on historical data
        model = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True)
        
        # Predict the 5 days we 'hid'
        backtest_preds = model.predict(n_periods=5)
        
        # Calculate Accuracy (100 - MAPE)
        error = mean_absolute_percentage_error(actual_last_5, backtest_preds)
        accuracy_score = max(0, 100 - (error * 100))
        
        # Now train on the FULL dataset to get tomorrow's real prediction
        final_model = auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True)
        tomorrow_pred = final_model.predict(n_periods=1)[0]
        change = ((tomorrow_pred / series[-1]) - 1) * 100
        
        results.append({
            'Stock': stock_name.replace('_Close', ''),
            'Price': round(series[-1], 2),
            'Pred_Tomorrow': round(tomorrow_pred, 2),
            'Change_%': round(change, 2),
            'Accuracy_Score': round(accuracy_score, 2),
            'Model_Order': str(final_model.order)
        })
        print(f"✅ Accuracy: {round(accuracy_score, 2)}%")
        
    except:
        print("❌ Error")

# 4. SAVE RESULTS
if results:
    res_df = pd.DataFrame(results).sort_values(by='Change_%', ascending=False)
    local_path = "/tmp/arima_accuracy_results.csv"
    res_df.to_csv(local_path, index=False)
    
    # Push back to HDFS
    hdfs_out = "/user/hdoop/nifty_project/forecasts/arima_accuracy_report.csv"
    subprocess.run(["hdfs", "dfs", "-put", "-f", local_path, hdfs_out])
    print(f"\n🌟 ACCURACY REPORT SAVED TO HDFS: {hdfs_out}")
