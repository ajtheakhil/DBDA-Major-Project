from pyspark.sql import SparkSession
import pandas as pd
import yfinance as yf
import subprocess
import os

# --------------------------------------------------
# 1. Spark Session WITH Hive Support
# --------------------------------------------------

warehouse_path = os.path.expanduser("~/spark-warehouse-v2")
#metastore_path = os.path.join(warehouse_path, "metastore_db")
#os.makedirs(metastore_path, exist_ok=True)

metastore_db = "jdbc:derby:;databaseName=/home/ds_ubuntu/metastore_db;create=true"
warehouse_path = "/home/ds_ubuntu/hive/warehouse"

spark = SparkSession.builder \
    .appName("Nifty_Hive_Sync") \
    .config("spark.sql.warehouse.dir", warehouse_path) \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("javax.jdo.option.ConnectionURL", metastore_db) \
    .enableHiveSupport() \
    .getOrCreate()

# --------------------------------------------------
# 2. Read NIFTY 200 symbols from CSV
# --------------------------------------------------
symbols_path = "/media/sf_ubuntu_project_folder/ind_nifty200list.csv"
symbols_df = pd.read_csv(symbols_path)
symbols = symbols_df['Symbol'].astype(str).str.strip()

# Convert to Yahoo Finance format
nifty_symbols = [f"{sym}.NS" for sym in symbols]
nifty_symbols.append("^NSEI")
print(f"Total symbols loaded: {len(nifty_symbols)}")

# --------------------------------------------------
# 3. Download 1 Year Daily Data
# --------------------------------------------------
print("Downloading data from Yahoo Finance...")
data = yf.download(
    tickers=nifty_symbols,
    period="1y",
    interval="1d",
    group_by="ticker",
    threads=True,
    auto_adjust=False
)

# --------------------------------------------------
# 4. Flatten MultiIndex columns
# --------------------------------------------------
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in data.columns
    ]
data.reset_index(inplace=True)

# --------------------------------------------------
# 5. Save CSV locally
# --------------------------------------------------
local_csv_path = "/home/ds_ubuntu/nifty_project/nifty_raw.csv"
os.makedirs(os.path.dirname(local_csv_path), exist_ok=True)
data.to_csv(local_csv_path, index=False)
print(f"CSV saved locally at: {local_csv_path}")

# --------------------------------------------------
# 6. Copy CSV to HDFS
# --------------------------------------------------
hdfs_raw_path = "/user/hdoop/nifty_project/raw/nifty_raw.csv"
subprocess.run(["hdfs", "dfs", "-put", "-f", local_csv_path, hdfs_raw_path], check=True)
print(f"CSV copied to HDFS at: {hdfs_raw_path}")

# --------------------------------------------------
# 7. Load CSV from HDFS into Spark DataFrame
# --------------------------------------------------
spark_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(hdfs_raw_path)

spark_df.cache()
print("Spark DataFrame Schema:")
spark_df.printSchema()
spark_df.show(5, truncate=False)

# --------------------------------------------------
# 8. Create Hive Database and Table
# --------------------------------------------------
spark.sql("CREATE DATABASE IF NOT EXISTS nifty_db")

spark_df.write \
    .mode("overwrite") \
    .format("parquet") \
    .saveAsTable("nifty_db.nifty_raw")

print("Hive table nifty_db.nifty_raw created successfully")

# --------------------------------------------------
# 9. Stop Spark
# --------------------------------------------------
spark.stop()
print("Process completed successfully.")

