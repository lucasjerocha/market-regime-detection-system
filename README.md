# Market Regime Detection System

A quantitative finance project designed to identify market regimes (Bull, Bear, Sideways) using unsupervised machine learning applied to historical equity market data.

The system combines feature engineering, clustering techniques, and financial interpretation to classify structural market states based on returns, volatility, and momentum indicators.

---

## Overview

Financial markets transition between distinct regimes characterized by differences in return distributions and volatility structure. Detecting these regimes can improve risk management, portfolio allocation, and strategy adaptation.

This project:

- Downloads historical market data (SPY ETF proxy)
- Engineers statistical and technical features
- Applies K-Means clustering
- Maps clusters into economically meaningful regimes
- Visualizes regime transitions over time

The objective is to demonstrate how unsupervised learning can extract structural patterns from financial time series.

---

## Methodology

### Data Source

Historical price data is retrieved using the `yfinance` library.  
SPY (SPDR S&P 500 ETF) is used as a proxy for the U.S. equity market.

---

### Feature Engineering

The model uses the following features:

- Daily returns
- Rolling volatility
- 50-day moving average
- 200-day moving average
- Relative Strength Index (RSI)

Data preprocessing includes:

- Handling missing values
- Removing infinite values
- Aligning feature indices
- Ensuring numerical stability

---

### Regime Classification

K-Means clustering (k = 3) is applied to the feature matrix.

Since K-Means assigns arbitrary cluster labels, clusters are mapped to regimes based on economic interpretation:

- Cluster with highest average return → Bull
- Cluster with lowest average return → Bear
- Remaining cluster → Sideways

This mapping ensures consistency and interpretability.

---

### Visualization

The system overlays classified regimes on SPY price data, allowing visual inspection of structural market transitions.

---

## Project Structure
Market Regime Detection System/
│
├── src/
│ ├── main.py
│ ├── regime_model.py
│ └── debug_features.py
│
├── requirements.txt
└── README.md

---

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate   # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the system:

```bash
python main.py
```

The script will:

- Download SPY historical data  
- Compute engineered features  
- Classify market regimes  
- Generate a regime visualization  

---

## Concepts Demonstrated

- Time-series feature engineering  
- Rolling statistics  
- Momentum indicators (RSI)  
- Unsupervised machine learning  
- Cluster interpretation in finance  
- Data preprocessing for ML pipelines  
- Regime modeling in financial markets  

---

## Potential Extensions

- Hidden Markov Models (HMM)  
- Gaussian Mixture Models  
- Macro-economic feature integration  
- Out-of-sample validation  
- Regime-based strategy backtesting  
- Silhouette-based dynamic cluster selection  
- Multi-asset regime detection  

---

## Motivation

Understanding regime dynamics is essential for:

- Tactical asset allocation  
- Volatility targeting  
- Risk management  
- Strategy adaptation under structural market shifts  

This project provides a systematic framework for identifying market regimes using observable price dynamics.

---

## Technologies

- Python  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- yfinance  

---

## Author

Lucas Rocha  
Quantitative Finance