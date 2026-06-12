import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tesla Stock Prediction – LSTM & RNN",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem; text-align: center;
    }
    .main-header h1 { color: #e94560; margin: 0; font-size: 2.2rem; }
    .main-header p  { color: #a8b2d8; margin: 0.4rem 0 0; font-size: 1rem; }
    .metric-card {
        background: #16213e; border: 1px solid #0f3460;
        border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
    }
    .metric-card .label { color: #a8b2d8; font-size: 0.85rem; margin-bottom: 0.3rem; }
    .metric-card .value { color: #e94560; font-size: 1.6rem; font-weight: 700; }
    .metric-card .sub   { color: #64ffda; font-size: 0.8rem; margin-top: 0.2rem; }
    .section-header {
        border-left: 4px solid #e94560; padding-left: 0.8rem;
        margin: 1.5rem 0 1rem; font-size: 1.2rem; font-weight: 600;
    }
    div[data-testid="stSidebar"] { background: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🚗 Tesla Stock Price Prediction</h1>
    <p>Deep Learning Time-Series Analysis &nbsp;|&nbsp; SimpleRNN &amp; LSTM Models &nbsp;|&nbsp; Multi-Day Horizons</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # Data source — bundled CSV by default, optional override
    BUNDLED = "TSLA.csv"
    uploaded_file = st.file_uploader(
        "📂 Upload a different CSV (optional)",
        type=["csv"],
        help="Leave empty to use the bundled TSLA dataset (2010–2020).",
    )

    st.markdown("### 🧠 Model Settings")
    window_size = st.slider("Look-back Window (days)", 30, 120, 60, 10)
    train_split = st.slider("Train/Test Split (%)",    60, 90,  80,  5) / 100
    epochs      = st.slider("Max Epochs",               20, 200, 100, 10)
    batch_size  = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)

    st.markdown("### 🔧 Hyperparameters")
    rnn_units = st.selectbox("RNN/LSTM Units", [32, 64, 128], index=1)
    dropout   = st.slider("Dropout Rate",   0.0, 0.5, 0.2, 0.05)
    lr        = st.select_slider("Learning Rate",
                    options=[0.0001, 0.0005, 0.001, 0.005], value=0.001)

    st.markdown("### 📅 Forecast Horizon")
    horizons = st.multiselect("Days to Forecast", [1, 3, 5, 10, 15, 30], default=[1, 5, 10])

    st.markdown("---")
    run_btn = st.button("🚀 Train & Predict", use_container_width=True, type="primary")

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data(raw_bytes: bytes | None) -> pd.DataFrame:
    import io
    src = io.BytesIO(raw_bytes) if raw_bytes else BUNDLED
    df = pd.read_csv(src, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df.ffill(inplace=True); df.bfill(inplace=True)
    df["Daily_Return"] = df["Close"].pct_change()
    df["MA_20"]  = df["Close"].rolling(20).mean()
    df["MA_50"]  = df["Close"].rolling(50).mean()
    df["MA_200"] = df["Close"].rolling(200).mean()
    return df

def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_rnn(units, dropout, lr, window):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    m = Sequential([
        Input(shape=(window, 1)),
        SimpleRNN(units, return_sequences=True),
        Dropout(dropout),
        SimpleRNN(units // 2),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1),
    ], name="SimpleRNN_Model")
    m.compile(optimizer=Adam(lr), loss="mse", metrics=["mae"])
    return m

def build_lstm(units, dropout, lr, window):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    m = Sequential([
        Input(shape=(window, 1)),
        LSTM(units, return_sequences=True),
        Dropout(dropout),
        LSTM(units // 2),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1),
    ], name="LSTM_Model")
    m.compile(optimizer=Adam(lr), loss="mse", metrics=["mae"])
    return m

def train_model(model, X_tr, y_tr, ep, bs, patience=15):
    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor="val_loss", patience=patience,
                       restore_best_weights=True, verbose=0)
    return model.fit(X_tr, y_tr, epochs=ep, batch_size=bs,
                     validation_split=0.1, callbacks=[es], verbose=0)

def metrics(actual, pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(actual, pred)
    return dict(MSE=mse, RMSE=np.sqrt(mse),
                MAE=mean_absolute_error(actual, pred),
                R2=r2_score(actual, pred),
                MAPE=float(np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100))

def multi_step(model, last_seq, scaler, n, window):
    cur = list(last_seq.flatten()[-window:])
    preds = []
    for _ in range(n):
        inp = np.array(cur[-window:]).reshape(1, window, 1)
        p = float(model.predict(inp, verbose=0)[0, 0])
        preds.append(p)
        cur.append(p)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
raw = uploaded_file.getvalue() if uploaded_file else None
df  = load_data(raw)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_eda, tab_train, tab_eval, tab_forecast = st.tabs(
    ["📊 EDA", "🏋️ Training", "📈 Evaluation", "🔮 Forecast"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – EDA
# ─────────────────────────────────────────────────────────────────────────────
with tab_eda:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, sub in zip(
        [c1, c2, c3, c4],
        ["Trading Days", "Avg Close", "All-Time High", "All-Time Low"],
        [f"{len(df):,}", f"${df['Close'].mean():.2f}",
         f"${df['Close'].max():.2f}", f"${df['Close'].min():.2f}"],
        [f"{df.index.min().date()} → {df.index.max().date()}",
         f"Std ${df['Close'].std():.2f}",
         str(df['Close'].idxmax().date()),
         str(df['Close'].idxmin().date())],
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{val}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Price History & Moving Averages</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"],  name="Close",  line=dict(color="#90CAF9", width=1),   opacity=0.8))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_20"],  name="MA 20",  line=dict(color="#F44336", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_50"],  name="MA 50",  line=dict(color="#4CAF50", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_200"], name="MA 200", line=dict(color="#FF9800", width=2)))
    fig.update_layout(height=380, template="plotly_dark", xaxis_title="Date", yaxis_title="Price (USD)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="section-header">Trading Volume</div>', unsafe_allow_html=True)
        fv = go.Figure(go.Bar(x=df.index, y=df["Volume"], marker_color="#FF9800", opacity=0.6))
        fv.update_layout(height=300, template="plotly_dark", yaxis_title="Volume")
        st.plotly_chart(fv, use_container_width=True)
    with cb:
        st.markdown('<div class="section-header">Daily Return Distribution</div>', unsafe_allow_html=True)
        fr = px.histogram(df["Daily_Return"].dropna(), nbins=80,
                          color_discrete_sequence=["#9C27B0"], template="plotly_dark", height=300)
        fr.update_layout(xaxis_title="Daily Return", yaxis_title="Frequency", showlegend=False)
        st.plotly_chart(fr, use_container_width=True)

    st.markdown('<div class="section-header">Candlestick — Last 90 Trading Days</div>', unsafe_allow_html=True)
    last90 = df.tail(90)
    fc = go.Figure(go.Candlestick(
        x=last90.index, open=last90["Open"], high=last90["High"],
        low=last90["Low"],  close=last90["Close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
    fc.update_layout(height=380, template="plotly_dark",
                     xaxis_rangeslider_visible=False, xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fc, use_container_width=True)

    st.markdown('<div class="section-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)
    corr = df[["Open","High","Low","Close","Volume"]].corr().round(2)
    fhm  = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                     template="plotly_dark", height=380)
    st.plotly_chart(fhm, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – TRAINING
# ─────────────────────────────────────────────────────────────────────────────
with tab_train:
    if not run_btn:
        st.info("Configure settings in the sidebar, then press **🚀 Train & Predict**.")
        st.stop()

    import tensorflow as tf
    tf.random.set_seed(42); np.random.seed(42)
    from sklearn.preprocessing import MinMaxScaler

    tcol   = "Adj Close" if "Adj Close" in df.columns else "Close"
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[[tcol]].values)

    X, y = create_sequences(scaled, window_size)
    split = int(len(X) * train_split)
    X_tr, X_te = X[:split].reshape(-1, window_size, 1), X[split:].reshape(-1, window_size, 1)
    y_tr, y_te = y[:split], y[split:]

    st.success(f"✅ {X_tr.shape[0]} training / {X_te.shape[0]} test sequences · window = {window_size} days · target = `{tcol}`")

    # ── Train SimpleRNN ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Training SimpleRNN</div>', unsafe_allow_html=True)
    pb_rnn = st.progress(5, text="SimpleRNN: training…")
    rnn_model   = build_rnn(rnn_units, dropout, lr, window_size)
    history_rnn = train_model(rnn_model, X_tr, y_tr, epochs, batch_size)
    pb_rnn.progress(100, text="SimpleRNN ✅")

    # ── Train LSTM ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Training LSTM</div>', unsafe_allow_html=True)
    pb_lstm = st.progress(5, text="LSTM: training…")
    lstm_model   = build_lstm(rnn_units, dropout, lr, window_size)
    history_lstm = train_model(lstm_model, X_tr, y_tr, epochs, batch_size)
    pb_lstm.progress(100, text="LSTM ✅")

    # Persist to session state
    for k, v in dict(rnn_model=rnn_model, lstm_model=lstm_model,
                     history_rnn=history_rnn, history_lstm=history_lstm,
                     X_te=X_te, y_te=y_te, scaler=scaler,
                     scaled=scaled, tcol=tcol, trained=True).items():
        st.session_state[k] = v

    # ── Loss curves ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Training & Validation Loss</div>', unsafe_allow_html=True)
    fig_loss = make_subplots(rows=1, cols=2, subplot_titles=["SimpleRNN Loss", "LSTM Loss"])
    for ci, (hist, color, name) in enumerate(
        [(history_rnn, "#2196F3", "RNN"), (history_lstm, "#E91E63", "LSTM")], 1):
        fig_loss.add_trace(go.Scatter(y=hist.history["loss"],     name=f"{name} Train",
                                      line=dict(color=color, width=2)), row=1, col=ci)
        fig_loss.add_trace(go.Scatter(y=hist.history["val_loss"], name=f"{name} Val",
                                      line=dict(color=color, width=2, dash="dot"), opacity=0.7), row=1, col=ci)
    fig_loss.update_layout(height=350, template="plotly_dark",
                           yaxis_type="log", yaxis2_type="log",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_loss, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("SimpleRNN — Best Val Loss",
              f"{min(history_rnn.history['val_loss']):.6f}",
              f"Stopped at epoch {len(history_rnn.history['loss'])}")
    c2.metric("LSTM — Best Val Loss",
              f"{min(history_lstm.history['val_loss']):.6f}",
              f"Stopped at epoch {len(history_lstm.history['loss'])}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_eval:
    if not st.session_state.get("trained"):
        st.info("Train models first in the **🏋️ Training** tab.")
        st.stop()

    rnn_m  = st.session_state["rnn_model"]
    lstm_m = st.session_state["lstm_model"]
    X_te   = st.session_state["X_te"]
    y_te   = st.session_state["y_te"]
    sc     = st.session_state["scaler"]

    actual    = sc.inverse_transform(y_te.reshape(-1, 1))
    rnn_pred  = sc.inverse_transform(rnn_m.predict(X_te, verbose=0))
    lstm_pred = sc.inverse_transform(lstm_m.predict(X_te, verbose=0))
    rnn_met   = metrics(actual, rnn_pred)
    lstm_met  = metrics(actual, lstm_pred)

    # Metric cards
    st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, key, label in zip(cols,
        ["RMSE","MAE","R2","MAPE"],
        ["RMSE (↓)","MAE (↓)","R² (↑)","MAPE % (↓)"]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{lstm_met[key]:.4f}</div>
                <div class="sub">LSTM &nbsp;|&nbsp; RNN: {rnn_met[key]:.4f}</div>
            </div>""", unsafe_allow_html=True)

    # Actual vs Predicted
    st.markdown('<div class="section-header">Actual vs Predicted — Test Set</div>', unsafe_allow_html=True)
    xax = list(range(len(actual)))
    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(x=xax, y=actual.flatten(),    name="Actual",    line=dict(color="#ffffff", width=1.5)))
    fig_avp.add_trace(go.Scatter(x=xax, y=rnn_pred.flatten(),  name="SimpleRNN", line=dict(color="#2196F3", width=1.5, dash="dot")))
    fig_avp.add_trace(go.Scatter(x=xax, y=lstm_pred.flatten(), name="LSTM",      line=dict(color="#E91E63", width=1.5, dash="dash")))
    fig_avp.update_layout(height=400, template="plotly_dark",
                          xaxis_title="Test Set Steps", yaxis_title="Price (USD)",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_avp, use_container_width=True)

    # Residuals
    st.markdown('<div class="section-header">Residuals</div>', unsafe_allow_html=True)
    cr1, cr2 = st.columns(2)
    for col, pred, name, color in [(cr1, rnn_pred, "SimpleRNN", "#2196F3"),
                                    (cr2, lstm_pred, "LSTM",     "#E91E63")]:
        res = actual.flatten() - pred.flatten()
        fr  = go.Figure()
        fr.add_trace(go.Scatter(y=res, mode="lines", line=dict(color=color, width=1)))
        fr.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fr.update_layout(height=300, template="plotly_dark",
                         title=f"{name} Residuals", yaxis_title="Error (USD)")
        col.plotly_chart(fr, use_container_width=True)

    # Comparison table + bar
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
    comp = pd.DataFrame({
        "Model": ["SimpleRNN","LSTM"],
        "RMSE":  [rnn_met["RMSE"],  lstm_met["RMSE"]],
        "MAE":   [rnn_met["MAE"],   lstm_met["MAE"]],
        "R²":    [rnn_met["R2"],    lstm_met["R2"]],
        "MAPE%": [rnn_met["MAPE"],  lstm_met["MAPE"]],
    }).set_index("Model")
    st.dataframe(comp.style.format("{:.4f}")
                     .highlight_min(subset=["RMSE","MAE","MAPE%"], color="#1a3a1a")
                     .highlight_max(subset=["R²"],                  color="#1a3a1a"),
                 height=100)

    fig_bar = make_subplots(rows=1, cols=3, subplot_titles=["RMSE","MAE","MAPE%"])
    for i, key in enumerate(["RMSE","MAE","MAPE"], 1):
        fig_bar.add_trace(go.Bar(name="SimpleRNN", x=["SimpleRNN"], y=[rnn_met[key]],  marker_color="#2196F3"), row=1, col=i)
        fig_bar.add_trace(go.Bar(name="LSTM",      x=["LSTM"],      y=[lstm_met[key]], marker_color="#E91E63"), row=1, col=i)
    fig_bar.update_layout(height=320, template="plotly_dark", showlegend=False, barmode="group")
    st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with tab_forecast:
    if not st.session_state.get("trained"):
        st.info("Train models first in the **🏋️ Training** tab.")
        st.stop()

    rnn_m  = st.session_state["rnn_model"]
    lstm_m = st.session_state["lstm_model"]
    sc     = st.session_state["scaler"]
    scaled = st.session_state["scaled"]
    tcol   = st.session_state["tcol"]

    last_price = df[tcol].iloc[-1]
    last_seq   = scaled[-window_size:].flatten()

    st.markdown('<div class="section-header">Multi-Step Recursive Forecasting</div>', unsafe_allow_html=True)
    st.caption(f"Seeding from last **{window_size}** trading days. Last known `{tcol}`: **${last_price:.2f}** on {df.index[-1].date()}")

    if not horizons:
        st.warning("Select at least one horizon in the sidebar.")
        st.stop()

    max_h = max(horizons)
    with st.spinner("Generating forecasts…"):
        rnn_fc  = multi_step(rnn_m,  last_seq, sc, max_h, window_size)
        lstm_fc = multi_step(lstm_m, last_seq, sc, max_h, window_size)

    # Summary table
    rows = []
    for h in sorted(horizons):
        rv, lv = rnn_fc[h-1], lstm_fc[h-1]
        rows.append({"Horizon": f"{h}-Day",
                     "SimpleRNN $": f"${rv:.2f}", "RNN Δ%":  f"{(rv/last_price-1)*100:+.2f}%",
                     "LSTM $":      f"${lv:.2f}", "LSTM Δ%": f"{(lv/last_price-1)*100:+.2f}%"})
    st.dataframe(pd.DataFrame(rows).set_index("Horizon"), use_container_width=True)

    # Chart
    ctx_n      = min(90, len(df))
    ctx_prices = df[tcol].values[-ctx_n:]
    ctx_dates  = df.index[-ctx_n:]
    future_days = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=max_h)

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=ctx_dates, y=ctx_prices,
                                name="Historical", line=dict(color="#607D8B", width=1.5)))
    x_fc   = [df.index[-1]] + list(future_days)
    rnn_y  = np.concatenate([[last_price], rnn_fc])
    lstm_y = np.concatenate([[last_price], lstm_fc])
    fig_fc.add_trace(go.Scatter(x=x_fc, y=rnn_y,  name="SimpleRNN Forecast",
                                line=dict(color="#2196F3", width=2), mode="lines+markers", marker=dict(size=6)))
    fig_fc.add_trace(go.Scatter(x=x_fc, y=lstm_y, name="LSTM Forecast",
                                line=dict(color="#E91E63", width=2, dash="dash"), mode="lines+markers", marker=dict(size=6)))
    for h in sorted(horizons):
        fig_fc.add_vline(x=future_days[h-1], line_dash="dot", line_color="rgba(255,255,255,0.25)")
    fig_fc.add_vline(x=df.index[-1], line_dash="dash", line_color="gold",
                     annotation_text="Last Data Point", annotation_position="top right")
    fig_fc.update_layout(height=420, template="plotly_dark",
                         xaxis_title="Date", yaxis_title="Price (USD)",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_fc, use_container_width=True)

    # Cards
    st.markdown('<div class="section-header">Forecast Cards</div>', unsafe_allow_html=True)
    card_cols = st.columns(len(horizons))
    for col, h in zip(card_cols, sorted(horizons)):
        rv, lv = rnn_fc[h-1], lstm_fc[h-1]
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{h}-Day Forecast</div>
                <div class="value">${lv:.2f}</div>
                <div class="sub">LSTM &nbsp;|&nbsp; RNN: ${rv:.2f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** Educational purposes only. Not financial advice.")