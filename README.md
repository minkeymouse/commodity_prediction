## Mitsui Commodity Prediction Challenge – README

### Purpose
Predict daily, cross‑sectional future log returns for 424 targets derived from LME/JPX/US/FX markets. The leaderboard metric rewards stable, per‑day ranking accuracy across targets rather than exact magnitudes.

### What you predict
- **Targets**: `target_0 .. target_423`
- **Single‑instrument target** at date t and lag k days:
  \( r_i(t,k) = \log P_i(t+k) - \log P_i(t) \)
- **Pair target** for instruments i and j at the same lag k:
  \( y_{i,j}(t,k) = r_i(t,k) - r_j(t,k) \)
  
Pairs are still based on log‑returns; you subtract two log‑returns computed over the same horizon.

### Files and schema
- `train.csv`: Historical features and prices for many instruments across exchanges. Column `date_id` is a shared calendar; some markets won’t trade every date.
- `train_labels.csv`: Wide table keyed by `date_id` with columns `target_[0-423]` (true labels during train period).
- `target_pairs.csv`: The “mapping” from each `target_x` to its definition:
  - **lag**: horizon k (how far ahead the return is measured)
  - **pair**: either one instrument (single target) or two instruments `(A,B)` (pair target)
- `test.csv`: Served structure of the unseen test set. In the public phase this mirrors the last ~90 days of the train period; the forecasting phase uses truly new dates.
- `lagged_test_labels/test_labels_lag_[1-4].csv`: For each day served during evaluation, these files contain labels that would be available by that date (enables online/rolling updates without leakage).
- `kaggle_evaluation/`: Evaluation API client used to stream data batches and write `submission.parquet`.

### Horizon (lag) and alignment
- The `lag` k is the look‑ahead length. Your label at date t for lag k is the k‑day forward log return.
- For a pair, compute both instruments’ k‑day log returns, then subtract.
- Feature alignment rule: features used to predict the label at `label_date_id = t` must come from data available up to `t - k` (no peeking).

### Evaluation metric: daily Spearman rank correlation Sharpe
1) For each scored `date_id`, compute the Spearman rank correlation across the 424 targets between your predictions and the true labels for that date.
2) Aggregate over all scored dates as the mean of these daily correlations divided by their standard deviation (an IC‑Sharpe).

Why rank correlation?
- **Cross‑sectional decisions**: You effectively rank assets each day to go long/short. Order matters more than absolute scale.
- **Scale invariance**: Targets have very different volatilities and units. Spearman ignores scale and focuses on ordering.
- **Robustness**: Heavy‑tailed returns and regime shifts make Pearson fragile; rank correlation is less sensitive to outliers.

Tiny intuition example (within a single day)
- True returns across 5 targets: `[0.10, -0.20, 0.50, 0.00, -0.10]`
- Pred A: `[2.0, -8.0, 10.0, 0.1, -0.5]` → same ordering → Spearman ≈ 1.0 despite very different scales.
- Pred B: `[2.0, -8.0, 10.0, -0.5, 0.1]` → last two swapped → lower Spearman.
- Flat predictions (many ties) produce low/undefined Spearman → avoid ties by ensuring per‑day dispersion.

Reference implementation (offline metric)
```python
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def daily_spearman_sharpe(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> float:
    # Index: date_id; Columns: target_0..target_423
    daily = []
    for d in y_true_df.index.intersection(y_pred_df.index):
        t = y_true_df.loc[d].values
        p = y_pred_df.loc[d].values
        mask = np.isfinite(t) & np.isfinite(p)
        if mask.sum() >= 3:
            r, _ = spearmanr(t[mask], p[mask])
            daily.append(r)
    daily = np.array(daily, dtype=float)
    return float(np.nanmean(daily) / np.nanstd(daily, ddof=1))
```

### How the evaluation loop runs (day by day)
The gateway streams one `date_id` at a time and expects one prediction row per date:
- For each `date_id` it provides:
  - `test_batch` = all test rows for that date
  - `test_labels_lag_1..4` filtered to that date (labels that would be known by then)
- Your `predict(...)` must return a single‑row DataFrame with predictions for all targets for that date (no `date_id` column in the predictions; the gateway adds row IDs and writes `submission.parquet`).

Implications
- You may update/retrain sequentially within the run using only labels provided up to that date (online learning without leakage).
- You must finish the full day‑by‑day loop within the time cap.

### Phases, refreshes, and runtime
- **Training phase**: public leaderboard uses a copy of the last ~90 days of train data; scores aren’t meaningful.
- **Forecasting phase**: the platform periodically “refreshes” by rerunning notebooks on an expanded, more recent test set and updating the leaderboard.
  - Timing is organizer‑controlled and not guaranteed/announced precisely.
  - Each refresh is a stateless full execution; if you want to retrain, do it within that execution using the data allowed at that time (e.g., train + newly released lagged labels).
- **Runtime limits**: notebooks must complete within 8 hours (training phase) and 9 hours (forecasting phase) with internet disabled.

### Submission format
- The gateway assembles predictions and row IDs into `submission.parquet` with row IDs first, then your target columns.
- One row per `date_id`. Your predictions must include exactly the target columns expected that day.

### Baseline implementation (momentum from lagged labels)
- A minimal per-day baseline is provided in `src/baseline_dayloop.py`.
- It infers the exact target columns from `lagged_test_labels/test_labels_lag_[1-4].csv`.
- For each `date_id`, it uses the most recently released lagged labels as a simple momentum-style prediction and adds tiny jitter to avoid ties (helpful for Spearman).
- It writes a valid `submission.parquet` with one row per `date_id` and all targets as columns.

How to run
```bash
python /data/kaggle_projects/commodity_prediction/src/baseline_dayloop.py \
  --data_dir /data/kaggle_projects/commodity_prediction/data \
  --out /data/kaggle_projects/commodity_prediction/submission.parquet
```

Dependencies
```bash
pip install -r /data/kaggle_projects/commodity_prediction/requirements.txt
```

### Ridge autoregressive labels baseline (optional)
Train one Ridge autoregressive model per target using past labels as features, then infer per day using the lagged label rows provided by the evaluation pipeline.

Train
```bash
python /data/kaggle_projects/commodity_prediction/src/train_ridge_autoreg.py \
  --data_dir /data/kaggle_projects/commodity_prediction/data \
  --out /data/kaggle_projects/commodity_prediction/models_ridge.pkl \
  --lags 1 2 3 4 --alpha 1.0
```

Infer
```bash
python /data/kaggle_projects/commodity_prediction/src/infer_ridge_autoreg.py \
  --models /data/kaggle_projects/commodity_prediction/models_ridge.pkl \
  --data_dir /data/kaggle_projects/commodity_prediction/data \
  --out /data/kaggle_projects/commodity_prediction/submission.parquet
```

### Best practices
- **Feature engineering**
  - Per‑instrument: rolling log returns (1, 3, 5, 10, 20d), rolling z‑scores, volatility, momentum/mean‑reversion features, rolling ranks/percentiles.
  - Cross‑asset: spreads/ratios aligned with `target_pairs.csv` definitions for pair targets.
  - Calendar/trading‑day features to handle exchange holidays and missing sessions.
- **Alignment**
  - Respect lag: features at time t should predict labels at t+k.
  - Ensure no use of `label_date_id` or future data in feature construction.
- **Modeling**
  - Start with ridge/lasso per target or a tree model (LightGBM/CatBoost). Per‑target models are simple and parallelizable.
  - Consider a shared model with target embeddings if resource limits allow.
- **Validation**
  - Time‑series CV by `date_id` (expanding window). Compute daily Spearman and aggregate with the IC‑Sharpe formula.
  - Consider purged CV to avoid leakage around splits.
- **Predictions**
  - Per day, ensure enough dispersion to avoid ties; scale calibration isn’t required for Spearman, order is.

### Common pitfalls
- Misalignment: using features from ≥ `label_date_id` (look‑ahead leakage) or ignoring `lag`.
- Overfitting to the last ~90 days (public LB). Optimize stability across many dates.
- Computing correlation on concatenated vectors instead of per‑day cross‑sections.
- Submissions with wrong shape: predictions must be exactly one row per served date and include all targets.

### Minimal baseline outline
1) Load `train.csv`, `train_labels.csv`, `target_pairs.csv`.
2) For each `target_x`:
   - Parse instruments and `lag` from `target_pairs.csv`.
   - Build features from those instruments: rolling returns, z‑scores, and volatilities.
   - Align features/labels so features up to t predict label at t+k.
3) Fit a regularized linear model per target (ridge) with expanding‑window CV.
4) Evaluate with the daily Spearman IC‑Sharpe.
5) For `test.csv`, run the per‑date loop: update model using available lagged labels (optional), predict one row for the current date, proceed to next.

### Results and offline evaluation
- The momentum baseline is intended to validate end-to-end plumbing and produce non-flat predictions; expect a low but non-zero IC most of the time.
- Use the metric utility in `src/metric.py` to compute the daily Spearman IC‑Sharpe offline:
```python
from src.metric import daily_spearman_sharpe
score = daily_spearman_sharpe(y_true_df, y_pred_df)
```
- Replace `y_pred_df` with your model outputs to iterate quickly before notebook submissions.

### FAQ
- **Do pair targets use log transforms?** Yes. Compute each instrument’s log‑return over lag k and subtract.
- **Is this one‑step forecasting?** Only when `lag=1`. Many targets have multi‑day horizons by design.
- **Does scale matter?** Not for the metric. Spearman only cares about per‑day ordering across targets; avoid ties.
- **Can I retrain during forecasting?** Yes, within each refresh execution, using only data available up to each served date.


### To‑Do
- [x] Write comprehensive README with task, data, targets, metric, evaluation loop.
- [x] Add IC‑Sharpe metric utility (`src/metric.py`).
- [x] Implement per‑day momentum baseline using lagged labels (`src/baseline_dayloop.py`).
- [ ] Implement aligned feature builder keyed by `target_pairs.csv`.
- [ ] Train per‑target ridge baseline with expanding-window CV.
- [ ] Add unified train/infer pipeline mirroring the gateway.
- [ ] Optimize runtime and memory for 9h forecasting phase limit.


