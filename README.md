# Stock Prediction Pipeline

A comprehensive Python pipeline for **stock price prediction** using historical data, feature engineering, and multiple machine learning models including **Linear Regression, Random Forest, and LSTM Neural Networks**. This project fetches historical stock data, performs exploratory data analysis, engineers technical indicators, trains models, evaluates them, and predicts future stock prices.

---

## Features

- Download historical stock data using `yfinance`.
- Perform **Exploratory Data Analysis (EDA)** with visualizations:
  - Stock price history
  - Daily returns
  - Trading volume
  - Correlation heatmap
- Engineer technical features for better predictions:
  - Moving Averages (MA)
  - Momentum indicators
  - Volatility measures
  - Relative Strength Index (RSI)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Volume-based indicators
- Prepare and scale data for modeling.
- Train and evaluate multiple models:
  - Linear Regression
  - Random Forest
  - LSTM Neural Network
- Generate visual comparison of actual vs predicted stock prices.
- Make **future predictions** for a specified number of days.
- Save all plots and results for reporting.

---

## Installation

Make sure you have Python 3.8+ installed. Install required dependencies using:

```bash
pip install pandas numpy matplotlib seaborn yfinance scikit-learn tensorflow
