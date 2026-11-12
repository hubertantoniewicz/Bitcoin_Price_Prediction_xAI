# BTC Price Prediction with Chainlets + Explainable AI

Predicting daily Bitcoin prices for December 2017 (one-day-ahead horizon) from
Bitcoin transaction-graph "chainlet" features (occurrence and amount) and lagged
historical prices, using regression / feedforward neural networks. Model
behavior is explained with SHAP.

UCF - Master's in FinTech - CAP 5619 homework.

## Approach

- Dates parsed with datetime functions only; chainlet and price files merged on date.
- Features for day t use data up to day t-1: price lags, lag summaries, and OCC + AMO chainlet changes.
- Target is the daily price change; prediction = previous price + predicted change.
- Ridge, Lasso and feedforward NN compared on a validation window (Oct-Nov 2017); Lasso selected.
- SHAP (`LinearExplainer`) explains the December predictions globally and for three individual days.

## Results

| | December 2017 RMSE (USD) |
|---|---|
| Lasso (selected) | 1,302.16 |
| Naive (previous day's price) | 1,095.08 |

The model was fixed on validation data before the test month was evaluated.

## Files

- `BTC_Prediction_xAI_hu827797.ipynb`: full pipeline with outputs.
- `predictions_dec2017.csv`: `date, predicted_price` for the 31 days of December 2017.
- `requirements.txt`: Python dependencies.

## Run

Data files are course-provided and not included. Place `AmoChainletsInTime.txt`,
`OccChainletsInTime.txt` and `pricedBitcoin2009-2018.csv` in `data/`, then:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace BTC_Prediction_xAI_hu827797.ipynb
```
