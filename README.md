# Trader-sentiment
Trader-sentiment-analysis
**Datasets Used**
Hyperliquid Historical Trader Data
Trade-level execution data including price, size, side, PnL
Bitcoin Fear & Greed Index
Daily market sentiment classification
**Objective**
To discover how trader:
Profitability
Risk-taking
Trade frequency
Directional bias
change depending on overall market sentimen.
**Methodology**
Clean and preprocess trade-level data
Aggregate trading metrics per day
Merge with daily sentiment classification
Compare trader behavior across Fear vs Greed regimes.
**Key Findings**
Insight
Observation
Trading Activity
Traders trade more frequently during Fear
Risk Behavior
Larger position sizes during Fear
Profitability
Higher total daily profits during Fear
Accuracy
Higher win rates during Greed
Trade Style.

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load Datasets
sns.set(style="whitegrid")
trades = pd.read_csv("data/historical_data.csv")
sentiment = pd.read_csv("data/fear_greed_index.csv")
# Data Cleaning
# Convert timestamps
trades['Timestamp'] = pd.to_datetime(trades['Timestamp'])
trades['Date'] = trades['Timestamp'].dt.date

sentiment['Date'] = pd.to_datetime(sentiment['Date']).dt.date

# Standardize sentiment labels
sentiment['Classification'] = sentiment['Classification'].str.capitalize()
# Create Trading Metrics Per Day
daily_metrics = trades.groupby('Date').agg(
    total_trades=('Closed PnL', 'count'),
    total_volume_usd=('Size USD', 'sum'),
    total_pnl=('Closed PnL', 'sum'),
    avg_pnl=('Closed PnL', 'mean'),
    win_rate=('Closed PnL', lambda x: (x > 0).mean())
).reset_index()
# Merge With Sentiment Data
merged = daily_metrics.merge(sentiment, on='Date', how='inner')
merged.head()
# Compare Fear vs Greed
comparison = merged.groupby('Classification').agg(
    avg_trades_per_day=('total_trades', 'mean'),
    avg_daily_volume=('total_volume_usd', 'mean'),
    avg_daily_pnl=('total_pnl', 'mean'),
    avg_win_rate=('win_rate', 'mean')
)
print(comparison)
# Visualization — Profitability by Sentiment
plt.figure(figsize=(8,5))
sns.barplot(data=merged, x='Classification', y='total_pnl')
plt.title("Average Daily PnL by Market Sentiment")
plt.show()
# Trade Direction Behavior
direction_analysis = trades.merge(sentiment, on='Date', how='inner')

sns.countplot(data=direction_analysis, x='Classification', hue='Side')
plt.title("Trade Direction by Sentiment")
plt.show()
# Key Insight Summary
print("Fear markets show higher activity and larger volume.")
print("Greed markets show better win rates and higher efficiency per trade.")
print("Traders adapt behavior depending on volatility and sentiment.")
