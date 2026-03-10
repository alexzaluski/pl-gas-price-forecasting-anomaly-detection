# Gas Price Forecasting & Anomaly Detection

Time series analysis of monthly heating costs — full pipeline from data cleaning through anomaly detection to SARIMA forecasting and Prophet comparison. Includes detection and explanation of the February 2022 energy price spike caused by the Russian invasion of Ukraine.

**Stack**: Python · pandas · statsmodels (SARIMA, STL) · Prophet · matplotlib · scipy

---

## Pipeline

### 1. Data Preparation

- Normalized heterogeneous date formats (`mm.yyyy`, `yyyy-mm-dd`, `/`-separated) into a consistent `YYYY-MM` datetime index
- Detected NaT/NaN entries; manually corrected 5 month-shifted rows (saved as `cost_yyyymm_fixed.tsv`)

### 2. Anomaly Detection

- **Classic decomposition**: outlier contaminated trend/seasonal estimates → anomaly appeared normal in residuals
- **STL decomposition (`robust=True`)**: down-weights extreme points during fitting, producing reliable residuals
- Z-score at **3σ** detected primary anomaly (2022-02, z = 5.88); lowering to **2σ** revealed a _ghost anomaly_ (2022-03, z = 2.71) — carry-over from supply disruptions
- Both anomalies replaced with their STL expected values (`trend + seasonal`)

| Date    | Actual  | STL Expected | Excess           | Z-score |
| ------- | ------- | ------------ | ---------------- | ------- |
| 2022-02 | 2000.00 | 1000.32      | +999.68 (+99.9%) | 5.88σ   |
| 2022-03 | 1331.00 | 856.02       | +474.98 (+55.5%) | 2.71σ   |

**Combined excess cost: $1,474.67 over 2 months.**  
**Cause**: Russia invaded Ukraine on 24 February 2022. Europe cut gas imports from Russia — Poland's major supplier — causing an unprecedented price spike that persisted into March.

### 3. Stationarity & Transformation

| Check                       | Result                                |
| --------------------------- | ------------------------------------- |
| Sequence plot               | Non-stationary (trend + seasonality)  |
| Mean/variance by time chunk | Mean ×15, variance ×172 across chunks |
| Histogram / Q-Q plot        | Non-normal, heavy right tail          |

**Transformations**: log → seasonal diff (D=1, lag=12) → regular diff (d=1) → stationary series oscillating around 0.

### 4. SARIMA Model Selection

Three candidates evaluated on MSE over a held-out 12-month test set:

| Model                       | MSE (test set) |
| --------------------------- | -------------- |
| **SARIMA(0,1,1)(0,1,1,12)** | **0.3228**     |
| SARIMA(1,1,0)(1,1,0,12)     | 0.3304         |
| SARIMA(1,1,1)(1,1,1,12)     | 0.3385         |

**Winner**: `SARIMA(0,1,1)(0,1,1,12)` — best on all metrics. MA terms capture short-term shocks; seasonal MA handles yearly patterns.

### 5. Model Diagnostics

Residuals of the final model: oscillate around 0, roughly normally distributed (histogram + Q-Q), no visible pattern → consistent with white noise ✓

### 6. Forecast Comparison: SARIMA vs Prophet

| Model                       | Mean (2024–2027) | Peak   | Min   |
| --------------------------- | ---------------- | ------ | ----- |
| SARIMA — with anomalies     | 1336.5           | 4078.8 | 108.7 |
| SARIMA — anomalies removed  | **1517.9**       | 4974.0 | 127.8 |
| Prophet — anomalies removed | 989.7            | 2487.9 | 155.8 |

- Removing anomalies shifted the SARIMA mean forecast upward — the model no longer averages the extraordinary spike into its baseline, revealing the higher underlying cost trend
- Prophet predicts significantly lower peak costs (~2488 vs ~4974), reflecting its tendency to smooth structural breaks rather than propagate them

---

## Key Findings

- STL with `robust=True` is essential in the presence of outliers — classic decomposition masked the anomaly
- The 2σ threshold catches ghost anomalies that 3σ misses
- The February 2022 spike (+99.9%, z = 5.88) is statistically unambiguous and causally linked to the energy crisis
- SARIMA and Prophet agree on seasonal pattern but diverge sharply on peak magnitude (~2× difference)

---

## Structure

```
pl-gas-price-forecasting-anomaly-detection/
├── notebook.ipynb              # full analysis (all steps)
└── data/
    ├── cost.csv                # raw input (original invoice data, mixed date formats)
    ├── cost_yyyymm.tsv         # intermediate: auto-parsed output; contains 5 date errors
    └── cost_yyyymm_fixed.tsv   # final input: manually corrected dates (used in all analysis)
```

### Data pipeline

```
cost.csv  →  [auto-parser: cell 1]  →  cost_yyyymm.tsv  →  [manual correction]  →  cost_yyyymm_fixed.tsv
                                              ↑                                               ↑
                                    plotted in cell 2 to                            used in all subsequent
                                    visually detect errors                          analysis cells
```

`cost_yyyymm.tsv` contains 5 rows with shifted dates (e.g. `2020-11` instead of `2021-11`). These were corrected manually to produce `cost_yyyymm_fixed.tsv`. Both files are needed to reproduce the full pipeline.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open `notebook.ipynb` and run all cells in order.

---

## Requirements

See `requirements.txt` for the full dependency list.
