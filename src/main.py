import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from regime_model import build_features, classify_regimes, regime_labels
import matplotlib
matplotlib.use("QtAgg") 


def plot_regimes(df, regimes):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df['Close'], label='SPY Price', color='black')

    # Only shade for the features index (regimes length)
    features_index = df.index[-len(regimes):]
    for regime in regime_labels:
        mask = (regimes == regime)
        ax.fill_between(features_index, df['Close'].min(), df['Close'].max(), where=mask, alpha=0.2,
                        color=regime_labels[regime]['color'], label=regime_labels[regime]['label'])

    # Add legend
    handles = [plt.Line2D([0], [0], color=regime_labels[r]['color'], lw=4, label=regime_labels[r]['label']) for r in regime_labels]
    ax.legend(handles=handles, loc='upper left')
    ax.set_title('SPY Price with Market Regimes')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.tight_layout()
    plt.show()


def main():
    # Download data
    df = yf.download('SPY', start='2000-01-01', end=None)
    if df is None or df.empty:
        print("Failed to download data for SPY.")
        return
    df = df.dropna()

    # Feature Engineering 
    features = build_features(df)
    if features is None or features.empty:
        print("Failed to build features.")
        return
    #Regime Classification
    regimes = classify_regimes(features)
    if regimes is None or len(regimes) == 0:
        print("Failed to classify regimes.")
        return
    
    #Visualization
    plot_regimes(df, regimes)

if __name__ == "__main__":
    main()