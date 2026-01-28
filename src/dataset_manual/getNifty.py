from pyspark.sql import SparkSession
import yfinance as yf
import pandas as pd
from io import StringIO

# Create PySpark session (scale with .master() for cluster)
spark = SparkSession.builder \
    .appName("Nifty200YahooFinanceToCSV") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Full Nifty 200 symbols from NSE (Yahoo format: SYMBOL.NS) + Nifty index
nifty200_symbols = [
    '360ONE.NS', 'ABB.NS', 'ACC.NS', 'APLAPOLLO.NS', 'AUBANK.NS',
    'ADANIENSOL.NS', 'ADANIENT.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 'ADANIPOWER.NS',
    'ATGL.NS', 'ABCAPITAL.NS', 'ALKEM.NS', 'AMBUJACEM.NS', 'APOLLOHOSP.NS',
    'ASHOKLEY.NS', 'ASIANPAINT.NS', 'ASTRAL.NS', 'AUROPHARMA.NS', 'DMART.NS',
    'AXISBANK.NS', 'BSE.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS',
    'BAJAJHLDNG.NS', 'BAJAJHFL.NS', 'BANKBARODA.NS', 'BANKINDIA.NS', 'BDL.NS',
    'BEL.NS', 'BHARATFORG.NS', 'BHEL.NS', 'BPCL.NS', 'BHARTIARTL.NS',
    'BHARTIHEXA.NS', 'BIOCON.NS', 'BLUESTARCO.NS', 'BOSCHLTD.NS', 'BRITANNIA.NS',
    'CGPOWER.NS', 'CANBK.NS', 'CHOLAFIN.NS', 'CIPLA.NS', 'COALINDIA.NS',
    'COCHINSHIP.NS', 'COFORGE.NS', 'COLPAL.NS', 'CONCOR.NS', 'COROMANDEL.NS',
    'CUMMINSIND.NS', 'DLF.NS', 'DABUR.NS', 'DIVISLAB.NS', 'DIXON.NS',
    'DRREDDY.NS', 'EICHERMOT.NS', 'ETERNAL.NS', 'EXIDEIND.NS', 'NYKAA.NS',
    'FEDERALBNK.NS', 'FORTIS.NS', 'GAIL.NS', 'GMRAIRPORT.NS', 'GLENMARK.NS',
    'GODFRYPHLP.NS', 'GODREJCP.NS', 'GODREJPROP.NS', 'GRASIM.NS', 'HCLTECH.NS',
    'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HAVELLS.NS', 'HEROMOTOCO.NS',
    'HINDALCO.NS', 'HAL.NS', 'HINDPETRO.NS', 'HINDUNILVR.NS', 'HINDZINC.NS',
    'POWERINDIA.NS', 'HUDCO.NS', 'HYUNDAI.NS', 'ICICIBANK.NS', 'ICICIGI.NS',
    'IDFCFIRSTB.NS', 'IRB.NS', 'ITCHOTELS.NS', 'ITC.NS', 'INDIANB.NS',
    'INDHOTEL.NS', 'IOC.NS', 'IRCTC.NS', 'IRFC.NS', 'IREDA.NS',
    'IGL.NS', 'INDUSTOWER.NS', 'INDUSINDBK.NS', 'NAUKRI.NS', 'INFY.NS',
    'INDIGO.NS', 'JSWENERGY.NS', 'JSWSTEEL.NS', 'JINDALSTEL.NS', 'JIOFIN.NS',
    'JUBLFOOD.NS', 'KEI.NS', 'KPITTECH.NS', 'KALYANKJIL.NS', 'KOTAKBANK.NS',
    'LTF.NS', 'LICHSGFIN.NS', 'LTIM.NS', 'LT.NS', 'LICI.NS',
    'LODHA.NS', 'LUPIN.NS', 'MRF.NS', 'M&MFIN.NS', 'M&M.NS',
    'MANKIND.NS', 'MARICO.NS', 'MARUTI.NS', 'MFSL.NS', 'MAXHEALTH.NS',
    'MAZDOCK.NS', 'MOTILALOFS.NS', 'MPHASIS.NS', 'MUTHOOTFIN.NS', 'NHPC.NS',
    'NMDC.NS', 'NTPCGREEN.NS', 'NTPC.NS', 'NATIONALUM.NS', 'NESTLEIND.NS',
    'OBEROIRLTY.NS', 'ONGC.NS', 'OIL.NS', 'PAYTM.NS', 'OFSS.NS',
    'POLICYBZR.NS', 'PIIND.NS', 'PAGEIND.NS', 'PATANJALI.NS', 'PERSISTENT.NS',
    'PHOENIXLTD.NS', 'PIDILITIND.NS', 'POLYCAB.NS', 'PFC.NS', 'POWERGRID.NS',
    'PREMIERENE.NS', 'PRESTIGE.NS', 'PNB.NS', 'RECLTD.NS', 'RVNL.NS',
    'RELIANCE.NS', 'SBICARD.NS', 'SBILIFE.NS', 'SRF.NS', 'MOTHERSON.NS',
    'SHREECEM.NS', 'SHRIRAMFIN.NS', 'ENRIN.NS', 'SIEMENS.NS', 'SOLARINDS.NS',
    'SONACOMS.NS', 'SBIN.NS', 'SAIL.NS', 'SUNPHARMA.NS', 'SUPREMEIND.NS',
    'SUZLON.NS', 'SWIGGY.NS', 'TVSMOTOR.NS', 'TATACOMM.NS', 'TCS.NS',
    'TATACONSUM.NS', 'TATAELXSI.NS', 'TMPV.NS', 'TATAPOWER.NS', 'TATASTEEL.NS',
    'TATATECH.NS', 'TECHM.NS', 'TITAN.NS', 'TORNTPHARM.NS', 'TORNTPOWER.NS',
    'TRENT.NS', 'TIINDIA.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'UNIONBANK.NS',
    'UNITDSPR.NS', 'VBL.NS', 'VEDL.NS', 'VMM.NS', 'IDEA.NS',
    'VOLTAS.NS', 'WAAREEENER.NS', 'WIPRO.NS', 'YESBANK.NS', 'ZYDUSLIFE.NS'
]

tickers = nifty200_symbols + ['^NSEI']  # Add Nifty index

# Download data (1y daily; adjust period/interval as needed)
print("Downloading Nifty 200 data...")
data = yf.download(tickers, period='1y', group_by='ticker', threads=True)

# Handle multi-index: concatenate levels (e.g., 'Adj Close_RELIANCE.NS')
if data.columns.nlevels > 1:
    data.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in data.columns]
data.reset_index(inplace=True)

# To CSV buffer
csv_buffer = StringIO()
data.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)

# PySpark DataFrame
df = spark.read.option("header", "true").option("inferSchema", "true").csv(csv_buffer)
df = df.cache()

# Inspect
df.printSchema()
df.show(5, truncate=False)

# Save as partitioned CSV (by Date for efficiency; ~200 cols x 250 rows)
df.write.partitionBy("Date").mode("overwrite").option("header", "true").csv("nifty200_yahoo_data")

spark.stop()
print("Data saved to 'nifty200_yahoo_data/' (use spark.read.csv('nifty200_yahoo_data/*') to reload)")
