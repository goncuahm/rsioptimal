import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide")

# =====================================================
# SIDEBAR INPUTS
# =====================================================

st.sidebar.title("RSI Pivot Strategy Settings")

TICKER = st.sidebar.text_input("Ticker", value="ISDMR.IS")

START = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2024-01-01")
)

END = datetime.today().strftime('%Y-%m-%d')

RISK_FREE_RATE_ANNUAL = st.sidebar.number_input(
    "Annual Risk-Free Rate (e.g. 0.30 = 30%)",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.01
)

TRANSACTION_COST = 0.002

# =====================================================
# DOWNLOAD FUNCTION
# =====================================================

@st.cache_data
def download_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[['Open', 'High', 'Low', 'Close']].dropna()
    return data

# =====================================================
# RSI FUNCTION
# =====================================================

def compute_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# =====================================================
# BACKTEST FUNCTION
# =====================================================

def backtest_strategy(df, rsi_len, pivot_lb, bottom_th, peak_th, risk_free_rate):

    df = df.copy()
    df['RSI'] = compute_rsi(df['Close'], rsi_len)
    df['Bottom'] = False
    df['Peak'] = False

    for i in range(pivot_lb, len(df)):
        window_past = df['RSI'].iloc[i - pivot_lb : i + 1]
        current = df['RSI'].iloc[i]

        if current == window_past.min() and current < bottom_th:
            df.iloc[i, df.columns.get_loc('Bottom')] = True

        if current == window_past.max() and current > peak_th:
            df.iloc[i, df.columns.get_loc('Peak')] = True

    initial_cash = 10000
    cash = initial_cash
    shares = 0.0
    position_size = 2000

    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    portfolio = []

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        cash *= (1 + daily_rf)

        if df['Bottom'].iloc[i] and cash >= position_size:
            shares_to_buy = position_size / price
            cost = shares_to_buy * price * (1 + TRANSACTION_COST)
            if cash >= cost:
                shares += shares_to_buy
                cash -= cost

        if df['Peak'].iloc[i] and shares > 0:
            shares_to_sell = min(position_size / price, shares)
            proceeds = shares_to_sell * price * (1 - TRANSACTION_COST)
            shares -= shares_to_sell
            cash += proceeds

        portfolio.append(cash + shares * price)

    df['Portfolio'] = portfolio
    returns = df['Portfolio'].pct_change().fillna(0)

    total_return = df['Portfolio'].iloc[-1] / initial_cash - 1
    sharpe = 0 if returns.std() == 0 else np.sqrt(252) * returns.mean() / returns.std()
    max_dd = (df['Portfolio'] / df['Portfolio'].cummax() - 1).min()

    return df, total_return, sharpe, max_dd

# =====================================================
# MAIN APP
# =====================================================

st.title("📊 RSI Pivot Strategy Optimizer")

data = download_stock_data(TICKER, START, END)

if data is None:
    st.error("No data found for this ticker.")
    st.stop()

st.success(f"Downloaded {len(data)} trading days")

# Split
split_index = int(len(data) * 0.75)
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

# Grid search
rsi_range = [9,10,11]
pivot_range = [3,5,7,10]
bottom_range = [25,30,33,37,40]
peak_range = [66,71,75,77,80]

best_score = -999
best_params = None

with st.spinner("Optimizing parameters..."):

    for rsi_len in rsi_range:
        for pivot_lb in pivot_range:
            for bottom_th in bottom_range:
                for peak_th in peak_range:
                    if peak_th <= bottom_th:
                        continue

                    _, _, sharpe, _ = backtest_strategy(
                        train_data,
                        rsi_len,
                        pivot_lb,
                        bottom_th,
                        peak_th,
                        risk_free_rate=0.0
                    )

                    if sharpe > best_score:
                        best_score = sharpe
                        best_params = (rsi_len, pivot_lb, bottom_th, peak_th)

st.subheader("Optimal Parameters (Train Set)")
st.write(f"RSI Length: {best_params[0]}")
st.write(f"Pivot Lookback: {best_params[1]}")
st.write(f"Bottom Threshold: {best_params[2]}")
st.write(f"Peak Threshold: {best_params[3]}")
st.write(f"Train Sharpe: {best_score:.2f}")

# Backtest
train_df, train_ret, train_sharpe, train_dd = backtest_strategy(
    train_data, *best_params, risk_free_rate=RISK_FREE_RATE_ANNUAL
)

test_df, test_ret, test_sharpe, test_dd = backtest_strategy(
    test_data, *best_params, risk_free_rate=RISK_FREE_RATE_ANNUAL
)

# Buy & Hold
initial_cash = 10000
buy_hold_train = initial_cash * (train_data['Close'] / train_data['Close'].iloc[0])
buy_hold_test = initial_cash * (test_data['Close'] / test_data['Close'].iloc[0])

# =====================================================
# PLOTS
# =====================================================

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,10), sharex=True)

# Portfolio plot
ax1.plot(train_df.index, train_df['Portfolio'], label="Strategy (Train)", color='blue')
ax1.plot(test_df.index, test_df['Portfolio'], label="Strategy (Test)", color='red')
ax1.plot(train_data.index, buy_hold_train, '--', color='lightblue', label="Buy & Hold (Train)")
ax1.plot(test_data.index, buy_hold_test, '--', color='lightcoral', label="Buy & Hold (Test)")

split_date = train_data.index[-1]
ax1.axvline(split_date, color='black')
ax1.legend()
ax1.set_title("Portfolio Value Comparison")

# Price + signals
ax2.plot(data.index, data['Close'], color='gray', label="Price")

bottoms = pd.concat([train_df, test_df])
ax2.scatter(bottoms[bottoms['Bottom']].index,
            bottoms[bottoms['Bottom']]['Close'],
            marker='^', color='green', s=100)

ax2.scatter(bottoms[bottoms['Peak']].index,
            bottoms[bottoms['Peak']]['Close'],
            marker='v', color='red', s=100)

ax2.axvline(split_date, color='black')
ax2.legend()

st.pyplot(fig)



# =====================================================
# CURRENT POSITION + SIGNAL
# =====================================================

st.subheader("📌 Current Portfolio Status")

# Re-run full backtest to reconstruct actual position
initial_cash = 10000
cash = initial_cash
shares = 0.0
position_size = 2000
daily_rf = (1 + RISK_FREE_RATE_ANNUAL) ** (1/252) - 1

for i in range(len(full_df)):
    price = full_df['Close'].iloc[i]
    cash *= (1 + daily_rf)

    if full_df['Bottom'].iloc[i] and cash >= position_size:
        shares_to_buy = position_size / price
        cost = shares_to_buy * price * (1 + TRANSACTION_COST)
        if cash >= cost:
            shares += shares_to_buy
            cash -= cost

    if full_df['Peak'].iloc[i] and shares > 0:
        shares_to_sell = min(position_size / price, shares)
        proceeds = shares_to_sell * price * (1 - TRANSACTION_COST)
        shares -= shares_to_sell
        cash += proceeds

# Current values
last_price = data['Close'].iloc[-1]
stock_value = shares * last_price
total_value = cash + stock_value

stock_pct = (stock_value / total_value * 100) if total_value > 0 else 0
cash_pct = 100 - stock_pct

current_rsi = full_df['RSI'].iloc[-1]
is_bottom = full_df['Bottom'].iloc[-1]
is_peak = full_df['Peak'].iloc[-1]

# Display nicely
col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Portfolio Value", f"${total_value:,.2f}")
col2.metric("📈 Stock Allocation", f"{stock_pct:.1f}%")
col3.metric("🏦 Cash Allocation", f"{cash_pct:.1f}%")

st.write(f"Current RSI: {current_rsi:.2f}")

st.markdown("---")
st.subheader("📢 Trading Recommendation")

if is_bottom and cash >= position_size:
    st.success(f"""
    🟢 **BUY SIGNAL**
    
    Suggested Action:
    Buy approximately ${position_size:,.0f} of {TICKER}
    
    Reason:
    RSI bottom pivot detected at {current_rsi:.2f}
    """)
elif is_peak and shares > 0:
    shares_to_sell = min(position_size / last_price, shares)
    sell_value = shares_to_sell * last_price

    st.error(f"""
    🔴 **SELL SIGNAL**
    
    Suggested Action:
    Sell approximately ${sell_value:,.0f} of {TICKER}
    
    Reason:
    RSI peak pivot detected at {current_rsi:.2f}
    """)
else:
    st.info("""
    ⚪ **HOLD**
    
    No confirmed RSI pivot signal.
    Maintain current allocation.
    """)

st.markdown("---")
st.write("✅ Analysis complete.")



# # =====================================================
# # CURRENT SIGNAL
# # =====================================================

# full_df, _, _, _ = backtest_strategy(
#     data, *best_params, risk_free_rate=RISK_FREE_RATE_ANNUAL
# )

# current_rsi = full_df['RSI'].iloc[-1]
# is_bottom = full_df['Bottom'].iloc[-1]
# is_peak = full_df['Peak'].iloc[-1]
# last_price = data['Close'].iloc[-1]

# st.subheader("Current Status")
# st.write(f"Last Close: {last_price:.2f}")
# st.write(f"Current RSI: {current_rsi:.2f}")

# if is_bottom:
#     st.success(f"🟢 BUY SIGNAL for {TICKER}")
# elif is_peak:
#     st.error(f"🔴 SELL SIGNAL for {TICKER}")
# else:
#     st.info("⚪ HOLD – No active signal")

# st.write("Analysis complete.")
