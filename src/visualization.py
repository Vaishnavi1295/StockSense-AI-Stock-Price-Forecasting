import plotly.graph_objects as go

def price_chart(data, mode="Close price"):
    if mode == "Candlestick":
        fig = go.Figure(data=[go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"], name="OHLC"
        )])
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", line=dict(color="#0f766e", width=2)))
    fig.update_layout(title="Historical stock price", xaxis_title="Date", yaxis_title="Price",
                      template="plotly_white", height=460, margin=dict(l=10,r=10,t=45,b=10),
                      xaxis_rangeslider_visible=False)
    return fig

def comparison_chart(dates, actual, predicted, baseline):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, name="Actual close", line=dict(color="#0f172a", width=2)))
    fig.add_trace(go.Scatter(x=dates, y=predicted, name="RNN/LSTM prediction", line=dict(color="#0d9488", width=2)))
    fig.add_trace(go.Scatter(x=dates, y=baseline, name="Naive baseline", line=dict(color="#f59e0b", width=1.5, dash="dot")))
    fig.update_layout(title="Held-out test data: actual vs predicted", xaxis_title="Date", yaxis_title="Price",
                      template="plotly_white", height=440, margin=dict(l=10,r=10,t=45,b=10))
    return fig

def loss_chart(history):
    fig = go.Figure()
    if "loss" in history:
        fig.add_trace(go.Scatter(y=history["loss"], mode="lines", name="Training loss"))
    if "val_loss" in history:
        fig.add_trace(go.Scatter(y=history["val_loss"], mode="lines", name="Validation loss"))
    fig.update_layout(title="Training and validation loss", xaxis_title="Epoch", yaxis_title="MSE loss",
                      template="plotly_white", height=330, margin=dict(l=10,r=10,t=45,b=10))
    return fig
