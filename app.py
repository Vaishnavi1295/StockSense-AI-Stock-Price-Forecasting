from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_stock_data, load_demo_data, normalize_ticker
from src.indicators import add_indicators
from src.preprocessing import chronological_split, make_sequences, scale_series
from src.model_builder import tensorflow_available, build_model
from src.evaluate import calculate_metrics, naive_predictions, directional_accuracy
from src.train import train_model, recursive_forecast
from src.visualization import price_chart, comparison_chart, loss_chart

st.set_page_config(page_title="StockSense AI | Stock Forecasting", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .stApp {background: linear-gradient(180deg, #f6f9ff 0%, #ffffff 42%);}
    [data-testid="stMetric"] {background: white; border: 1px solid #e5eaf3; padding: 15px; border-radius: 14px;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 20px; color: white;
           background: linear-gradient(115deg, #101a35 0%, #174a70 60%, #0d9488 100%); margin-bottom: 1rem;}
    .hero h1 {margin:0; font-size:2.15rem;}
    .hero p {margin:.4rem 0 0 0; color:#d9f1ff;}
    .small-note {color:#64748b; font-size:.9rem;}
    div[data-testid="stSidebar"] {background:#f0f5fb;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📈 StockSense AI</h1>
  <p>Stock Price Forecasting • RNN vs LSTM • Historical Analysis • Model Evaluation</p>
</div>
""", unsafe_allow_html=True)

st.caption("Educational forecasting project. Predictions are estimates, not financial advice or guarantees.")

with st.sidebar:
    st.header("⚙️ Forecast settings")
    market = st.selectbox("Market", ["India (NSE)", "United States"], index=0)
    default_ticker = "RELIANCE.NS" if market == "India (NSE)" else "AAPL"
    ticker = st.text_input("Stock ticker", value=default_ticker, help="Examples: RELIANCE.NS, TCS.NS, INFY.NS, AAPL, MSFT").strip()
    period = st.selectbox("Historical period", ["2y", "5y", "10y", "max"], index=1)
    interval = st.selectbox("Data interval", ["1d"], index=0)
    lookback = st.select_slider("Lookback window (trading sessions)", options=[20, 30, 45, 60, 90], value=60)
    horizon = st.select_slider("Forecast horizon (future sessions)", options=[1, 3, 5, 10], value=5)
    model_choice = st.selectbox("Deep learning model", ["LSTM", "Simple RNN"], index=0)
    use_demo = st.checkbox("Use built-in demo data (offline)", value=False)
    st.divider()
    train_button = st.button("🚀 Train & forecast", type="primary", use_container_width=True)
    st.caption("First training may take a few minutes. Internet is needed for live market data.")

@st.cache_data(ttl=1800, show_spinner=False)
def get_data(ticker_value, period_value, demo):
    if demo:
        return load_demo_data()
    return load_stock_data(ticker_value, period_value)

try:
    with st.spinner("Loading historical market data..."):
        data = get_data(ticker, period, use_demo)
    data = add_indicators(data)
except Exception as exc:
    st.error(f"Could not load market data: {exc}")
    st.info("Check the ticker and internet connection, or select **Use built-in demo data (offline)** in the sidebar.")
    st.stop()

if data.empty or len(data) < 100:
    st.error("Not enough valid historical data. Choose a longer period or another ticker.")
    st.stop()

display_ticker = "DEMO (synthetic data)" if use_demo else normalize_ticker(ticker)
latest = data.iloc[-1]
previous = data.iloc[-2]
change = float(latest["Close"] - previous["Close"])
change_pct = (change / previous["Close"] * 100) if previous["Close"] else 0

st.subheader(f"Market overview · {display_ticker}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest available close", f"{latest['Close']:,.2f}", f"{change:+,.2f} ({change_pct:+.2f}%)")
c2.metric("Trading sessions loaded", f"{len(data):,}")
c3.metric("Latest volume", f"{latest['Volume']:,.0f}" if pd.notna(latest["Volume"]) else "N/A")
c4.metric("Latest RSI (14)", f"{latest['RSI_14']:.1f}" if pd.notna(latest["RSI_14"]) else "N/A")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧠 Forecast Lab", "🔬 Model Evaluation", "📁 Data & About"])

with tab1:
    st.subheader("Historical price analysis")
    chart_mode = st.radio("Chart view", ["Close price", "Candlestick"], horizontal=True)
    fig = price_chart(data, chart_mode)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Technical indicators")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", line=dict(width=1.7)))
        fig_ma.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], name="SMA 20", line=dict(width=1.4)))
        fig_ma.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], name="SMA 50", line=dict(width=1.4)))
        fig_ma.update_layout(title="Closing price and moving averages", height=350, template="plotly_white", margin=dict(l=10,r=10,t=45,b=10))
        st.plotly_chart(fig_ma, use_container_width=True)
    with col_b:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data["RSI_14"], name="RSI (14)"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(title="Relative Strength Index (RSI)", height=350, template="plotly_white", yaxis_range=[0,100], margin=dict(l=10,r=10,t=45,b=10))
        st.plotly_chart(fig_rsi, use_container_width=True)

with tab2:
    st.subheader("Train a forecasting model")
    st.write(f"Selected model: **{model_choice}** · Lookback: **{lookback} sessions** · Forecast horizon: **{horizon} sessions**")
    if not tensorflow_available():
        st.warning("TensorFlow is not installed in this Python environment. Install dependencies using run_windows.bat or `pip install -r requirements.txt` with Python 3.11.")
    if train_button:
        if not tensorflow_available():
            st.error("The RNN/LSTM model needs TensorFlow. Follow the setup instructions in README.md, then restart the app.")
        else:
            try:
                with st.spinner(f"Training {model_choice} model and evaluating on unseen historical data..."):
                    feature_cols = ["Close"]
                    values = data[feature_cols].astype(float).values
                    split = chronological_split(values, train_ratio=0.70, val_ratio=0.15)
                    scaler, train_scaled, val_scaled, test_scaled = scale_series(split["train"], split["validation"], split["test"])
                    # Build all sequences without leakage: each target belongs to its own split.
                    X_train, y_train = make_sequences(train_scaled, lookback)
                    val_context = np.vstack([train_scaled[-lookback:], val_scaled])
                    X_val, y_val = make_sequences(val_context, lookback)
                    test_context = np.vstack([val_scaled[-lookback:], test_scaled])
                    X_test, y_test = make_sequences(test_context, lookback)
                    if min(len(X_train), len(X_val), len(X_test)) < 5:
                        st.error("Not enough samples for this lookback window. Choose a longer historical period.")
                    else:
                        model = build_model(model_choice, lookback, len(feature_cols))
                        model, history = train_model(model, X_train, y_train, X_val, y_val)
                        pred_scaled = model.predict(X_test, verbose=0)
                        pred = scaler.inverse_transform(pred_scaled).ravel()
                        actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
                        baseline_scaled = test_context[lookback-1:-1, 0]
                        baseline = scaler.inverse_transform(baseline_scaled.reshape(-1,1)).ravel()
                        metrics = calculate_metrics(actual, pred)
                        baseline_metrics = calculate_metrics(actual, baseline)
                        test_dates = data.index[-len(actual):]
                        future_scaled = recursive_forecast(model, train_scaled, val_scaled, test_scaled, lookback, horizon)
                        future_prices = scaler.inverse_transform(future_scaled.reshape(-1,1)).ravel()
                        st.session_state["forecast_result"] = {
                            "model_name": model_choice, "metrics": metrics, "baseline_metrics": baseline_metrics,
                            "actual": actual, "pred": pred, "baseline": baseline, "dates": test_dates,
                            "future": future_prices, "history": history.history, "last_close": float(data["Close"].iloc[-1]),
                            "ticker": display_ticker, "horizon": horizon
                        }
                        st.success("Training and evaluation completed.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.code(traceback.format_exc())

    result = st.session_state.get("forecast_result")
    if result:
        st.markdown("### Future forecast")
        forecast_dates = pd.bdate_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(result["future"]))
        # Use business days as an estimate; exchange holidays are not removed.
        forecast_df = pd.DataFrame({"Forecast session": np.arange(1, len(result["future"])+1), "Estimated date (business day)": forecast_dates.date, "Predicted close": result["future"]})
        st.warning("Forecast dates are approximate business days and do not account for exchange holidays. Forecast values are model estimates, not guarantees.")
        st.dataframe(forecast_df.style.format({"Predicted close":"{:,.2f}"}), use_container_width=True, hide_index=True)
        future_fig = go.Figure()
        recent = data.tail(120)
        future_fig.add_trace(go.Scatter(x=recent.index, y=recent["Close"], name="Historical close", line=dict(color="#0f766e", width=2)))
        future_fig.add_trace(go.Scatter(x=forecast_dates, y=result["future"], name="Model forecast", line=dict(color="#f59e0b", width=3, dash="dash"), mode="lines+markers"))
        future_fig.update_layout(title="Recent history and future forecast", template="plotly_white", height=430, xaxis_title="Date", yaxis_title="Price", margin=dict(l=10,r=10,t=45,b=10))
        st.plotly_chart(future_fig, use_container_width=True)
        st.download_button("⬇️ Download forecast CSV", forecast_df.to_csv(index=False).encode("utf-8"), file_name="stock_forecast.csv", mime="text/csv")
        st.markdown("### Training history")
        st.plotly_chart(loss_chart(result["history"]), use_container_width=True)

with tab3:
    st.subheader("Model evaluation")
    result = st.session_state.get("forecast_result")
    if not result:
        st.info("Open Forecast Lab and click **Train & forecast** to calculate real test metrics.")
        st.markdown("""
        **How evaluation works**
        - The data is split chronologically: first 70% for training, next 15% for validation, final 15% for testing.
        - The scaler is fitted on the training partition only.
        - Predictions are compared with actual prices on the held-out test period.
        - The naive baseline predicts the next close using the last observed close.
        """)
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("LSTM/RNN MAE", f"{result['metrics']['MAE']:.4f}")
        m2.metric("LSTM/RNN RMSE", f"{result['metrics']['RMSE']:.4f}")
        m3.metric("Directional accuracy", f"{directional_accuracy(result['actual'], result['pred']):.1f}%")
        baseline_metrics = result["baseline_metrics"]
        comparison = pd.DataFrame([
            {"Model": result["model_name"], "MAE": result["metrics"]["MAE"], "RMSE": result["metrics"]["RMSE"]},
            {"Model": "Naive baseline", "MAE": baseline_metrics["MAE"], "RMSE": baseline_metrics["RMSE"]},
        ])
        st.dataframe(comparison.style.format({"MAE":"{:.4f}", "RMSE":"{:.4f}"}), use_container_width=True, hide_index=True)
        st.plotly_chart(comparison_chart(result["dates"], result["actual"], result["pred"], result["baseline"]), use_container_width=True)
        if result["metrics"]["RMSE"] < baseline_metrics["RMSE"]:
            st.success("The selected neural network had lower RMSE than the naive baseline on this test period.")
        else:
            st.warning("The selected neural network did not beat the naive baseline on RMSE for this test period. This is a valid result; do not hide it.")
        eval_df = pd.DataFrame({"Date": pd.to_datetime(result["dates"]), "Actual close": result["actual"], "Predicted close": result["pred"], "Naive baseline": result["baseline"]})
        st.download_button("⬇️ Download test predictions", eval_df.to_csv(index=False).encode("utf-8"), file_name="test_predictions.csv", mime="text/csv")

with tab4:
    st.subheader("Dataset preview")
    st.dataframe(data.tail(20).reset_index(), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download historical data CSV", data.reset_index().to_csv(index=False).encode("utf-8"), file_name="historical_stock_data.csv", mime="text/csv")
    st.markdown("""
    **Project details**
    - **Primary model:** Long Short-Term Memory (LSTM)
    - **Comparison model:** Simple Recurrent Neural Network (RNN)
    - **Target:** Next-session closing price
    - **Evaluation:** Chronological holdout, MAE, RMSE and directional accuracy
    - **Indicators:** SMA, EMA, RSI and MACD
    - **Interface:** Streamlit and Plotly
    """)
    st.info("If demo mode is enabled, the chart uses clearly labelled synthetic sample data. Do not present it as real market history.")
    st.markdown("### Important limitations")
    st.write("Historical price patterns do not guarantee future performance. This application is for education and experimentation, not investment advice.")
