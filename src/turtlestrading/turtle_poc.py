import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# --- API CONFIG ---
API_KEY = "YOUR_EODHD_API_KEY"  # 🔑 Replace with your EODHD key
symbol = "SPY.US"
# --- FETCH DATA FROM EODHD API ---
url = f"https://eodhd.com/api/eod/{symbol}?api_token={API_KEY}&fmt=json"
response = requests.get(url)
print(response)
df = pd.DataFrame(response.json())
# --- CLEAN & PREPARE DATA ---
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.sort_index()
df = df[['close', 'high', 'low']].astype(float)
# --- TURTLE SYSTEM PARAMETERS ---
entry_window = 20
exit_window = 10
atr_period = 14
# --- CALCULATE SIGNALS ---
df['20d_high'] = df['high'].rolling(entry_window).max()
df['20d_low'] = df['low'].rolling(entry_window).min()
df['10d_high'] = df['high'].rolling(exit_window).max()
df['10d_low'] = df['low'].rolling(exit_window).min()
# ATR for volatility-adjusted position sizing
df['TR'] = np.maximum(df['high'] - df['low'],
                      np.maximum(abs(df['high'] - df['close'].shift()),
                                 abs(df['low'] - df['close'].shift())))
df['ATR'] = df['TR'].rolling(atr_period).mean()
# --- ENTRY & EXIT RULES ---
df['signal'] = 0
df.loc[df['close'] > df['20d_high'].shift(1), 'signal'] = 1   # Long entry
df.loc[df['close'] < df['20d_low'].shift(1), 'signal'] = -1  # Short entry
# Exit if opposite breakout
df['exit'] = 0
df.loc[df['close'] < df['10d_low'].shift(1), 'exit'] = 1
df.loc[df['close'] > df['10d_high'].shift(1), 'exit'] = 1
# --- STRATEGY LOGIC ---
df['position'] = np.nan
position = 0
for i in range(1, len(df)):
    if df['signal'].iloc[i] == 1:
        position = 1
    elif df['signal'].iloc[i] == -1:
        position = -1
    elif df['exit'].iloc[i] == 1:
        position = 0
    df['position'].iloc[i] = position
df['position'] = df['position'].ffill().fillna(0)
# --- RETURNS CALCULATION ---
df['returns'] = df['close'].pct_change()
df['strategy'] = df['position'].shift(1) * df['returns']
# --- EQUITY CURVES ---
df['equity_curve_strategy'] = (1 + df['strategy']).cumprod()
df['equity_curve_bh'] = (1 + df['returns']).cumprod()
# --- PERFORMANCE METRICS ---
def CAGR(series):
    years = (series.index[-1] - series.index[0]).days / 365
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1
def sharpe(series):
    return np.sqrt(252) * (series.mean() / series.std())
def max_drawdown(equity):
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    return drawdown.min()
cagr_strategy = CAGR(df['equity_curve_strategy'])
cagr_bh = CAGR(df['equity_curve_bh'])
sharpe_strategy = sharpe(df['strategy'])
mdd_strategy = max_drawdown(df['equity_curve_strategy'])
print("===== 📊 PERFORMANCE METRICS =====")
print(f"CAGR (Turtle Strategy): {cagr_strategy*100:.2f}%")
print(f"CAGR (Buy & Hold): {cagr_bh*100:.2f}%")
print(f"Sharpe Ratio: {sharpe_strategy:.2f}")
print(f"Max Drawdown: {mdd_strategy*100:.2f}%")
# --- PLOT EQUITY CURVES ---
plt.figure(figsize=(12,6))
plt.plot(df.index, df['equity_curve_strategy'], label='🐢 Turtle Strategy (1980s Formula)', color='dodgerblue', linewidth=2)
plt.plot(df.index, df['equity_curve_bh'], label='💼 Buy & Hold (SPY)', color='gray', linestyle='--')
plt.title('Turtle Trading Strategy vs Buy & Hold (SPY)')
plt.xlabel('Date')
plt.ylabel('Equity Curve (Normalized)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
