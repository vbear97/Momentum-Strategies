# Momentum Strategies (Cryptocurrency Order Book Data)

This repo investigates whether there are momentum trading signals based on high frequency (10s) cryptocurrency data. 

## Description 

Investigates the profitability of momentum-based trading signals on high-frequency cryptocurrency data sourced via Crypto Lake. Covers: 

- Data cleaning + visualisation 
- Signal construction & backtesting 

## Key Results (WiP) - Summary
- Time series momentum strategies not profitable at high frequency as a retail trader due to a high fees/short term returns ratio 
- If we are predicting short term momentum at a high frequency based on order flow, we cannot rely on predicting **median** price movements, since short term returns are extremely fat-tailed and so building a quantile regression/decision tree model at the 50% percentile is simply not predictive. 