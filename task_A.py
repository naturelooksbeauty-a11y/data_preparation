
import pandas as pd
import numpy as np

# ==========================================
# STEP 1: LOAD THE DATASETS
# ==========================================
# Replace 'historical_data.csv' and 'fear_greed.csv' with your actual file names
df_trades = pd.read_csv('historical_data.csv')
df_sentiment = pd.read_csv('fear_greed.csv')

print("--- DATA LOADED ---")
print(f"Trades Shape: {df_trades.shape}")
print(f"Sentiment Shape: {df_sentiment.shape}")

# Documenting missing values (Task A.1)
print("\nMissing Values in Trades:\n", df_trades.isnull().sum())
# ==========================================
# STEP 2: CLEAN & ALIGN DATES (Task A.2)
# ==========================================

# 1. Fix Trades Timestamp (The scientific notation column 'Timestamp')
# We use unit='ms' because your data is 1.73E+12 (milliseconds)
df_trades['datetime'] = pd.to_datetime(df_trades['Timestamp'], unit='ms')

# 2. Create a 'join_date' column (stripping the time) to match sentiment
df_trades['join_date'] = df_trades['datetime'].dt.normalize()

# 3. Fix Sentiment Date
# The image shows the column is named 'date' (e.g., 2/1/2018)
df_sentiment['join_date'] = pd.to_datetime(df_sentiment['date'])

# 4. Merge the datasets
# We assume we want to analyze the trades, so we 'left join' sentiment onto trades
df_merged = pd.merge(df_trades, df_sentiment, on='join_date', how='left')

print(f"\nMerged Data Shape: {df_merged.shape}")

# ==========================================
# STEP 3: CREATE KEY METRICS (Task A.3)
# ==========================================

# --- Metric 1: Leverage ---

#  We add a small number (1e-9) to avoid dividing by zero error if Start Position is 0.
df_merged['Calculated_Leverage'] = df_merged['Size USD'] / (df_merged['Start Position'] + 1e-9)
print("\n1. Leverage Distribution (Top 5000 rows):")
print(df_merged[['Account', 'Size USD', 'Start Position', 'Calculated_Leverage']].head(5000))

# --- Metric 2: Long/Short Ratio ---
# We use the unique values you showed in the image to classify direction.
def classify_side(direction):
    direction = str(direction).lower()
    if 'long' in direction or 'buy' in direction:
        return 'Long'
    elif 'short' in direction or 'sell' in direction:
        return 'Short'
    else:
        return 'Other'

df_merged['Trade_Side'] = df_merged['Direction'].apply(classify_side)

long_short_counts = df_merged['Trade_Side'].value_counts()
print("\n2. Long/Short Ratio:")
print(long_short_counts)

# --- Metric 3: Win Rate & Avg Trade Size ---
# We define a function to calculate stats for each trader
def calculate_trader_stats(x):
    total_trades = len(x)
    # A "Win" is when Closed PnL is positive
    wins = x[x['Closed PnL'] > 0]
    
    return pd.Series({
        'Win_Rate': len(wins) / total_trades if total_trades > 0 else 0,
        'Avg_Trade_Size_USD': x['Size USD'].mean(),
        'Total_Trades': total_trades,
        'Total_PnL': x['Closed PnL'].sum()
    })

# Group by Account to get stats per trader
trader_performance = df_merged.groupby('Account').apply(calculate_trader_stats)
print("\n3. Trader Performance (Sample):")
print(trader_performance.head(5000))

# --- Metric 4: Daily Activity ---
daily_stats = df_merged.groupby('join_date').agg({
    'Account': 'count',          # Number of trades per day
    'Closed PnL': 'sum'          # Daily PnL
}).rename(columns={'Account': 'Trade_Count', 'Closed PnL': 'Daily_PnL'})

print("\n4. Daily Activity (Sample):")
print(daily_stats.head())
#final observation,duplicated column removing
print(df_merged.drop(columns=["timestamp"],inplace=True))
# ==========================================
#  Saving the file
# ==========================================
#df_merged.to_csv("part__a__completed.csv", index=False)
   









