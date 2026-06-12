# 🚗 Tesla Stock Prediction — Streamlit App

Interactive dashboard for LSTM & SimpleRNN stock price prediction.
**The TSLA dataset (2010–2020) is bundled — no upload needed.**

---

## 📦 Quick Start

```bash
cd tesla_prediction_app
pip install -r requirements.txt
streamlit run app.py
```

Opens automatically at **http://localhost:8501**

---

## 🗂 Files

```
tesla_prediction_app/
├── app.py            ← Streamlit app
├── TSLA.csv          ← Bundled Tesla dataset (2010–2020, 2,416 rows)
├── requirements.txt  ← Python dependencies
└── README.md
```

---

## ✨ Tabs

| Tab | Content |
|-----|---------|
| 📊 EDA | Price + MAs, volume, return distribution, candlestick, correlation heatmap |
| 🏋️ Training | Train SimpleRNN & LSTM with configurable hyperparameters |
| 📈 Evaluation | RMSE / MAE / R² / MAPE, actual vs predicted, residuals, comparison chart |
| 🔮 Forecast | 1 / 5 / 10-day recursive multi-step predictions with interactive chart |

---

## ⚙️ Sidebar Controls

| Setting | Default | Description |
|---------|---------|-------------|
| Look-back Window | 60 days | History fed into each prediction |
| Train/Test Split | 80% | Fraction used for training |
| Max Epochs | 100 | Upper limit (early stopping active) |
| Batch Size | 32 | Mini-batch size |
| Units | 64 | RNN/LSTM layer units |
| Dropout | 0.2 | Regularisation rate |
| Learning Rate | 0.001 | Adam optimiser LR |
| Forecast Horizons | 1, 5, 10 | Days ahead to predict |

You can also upload a different CSV (same column format) to override the bundled dataset.
