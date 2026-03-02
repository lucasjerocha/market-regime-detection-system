import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


regime_labels = {
    0: {'label':'Bull Market', 'color': 'green'},
    1: {'label': 'Bear Market', 'color': 'red'},
    2: {'label': 'Sideways Market', 'color': 'blue'}
}


def build_features(df):
    features = pd.DataFrame(index=df.index)
    features['returns'] = df['Close'].pct_change()
    #Rolling volatility
    features['volatility'] = features['returns'].rolling(window=20).std()
    #Moving average
    features['ma_50'] = df['Close'].rolling(window=50).mean()
    features['ma_200'] = df['Close'].rolling(window=200).mean()
    #Momentum indicator (e.g., RSI)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / roll_down
    features['rsi']= 100 - (100 / (1+rs))
    features = features.dropna()
    return features

def classify_regimes(features, method='kmeans'):
    X = features[['returns', 'volatility', 'rsi']].values
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Map clusters to regime labels based on mean returns
    clusters_stats = pd.DataFrame({
        'returns': features['returns'].groupby(clusters).mean(),
        'volatility': features['volatility'].groupby(clusters).mean()
    })

    #Asign Bull, Bear, Sideways
    bull = clusters_stats['returns'].idxmax()
    bear = clusters_stats['returns'].idxmin()
    sideways = [i for i in clusters_stats.index if i not in [bull, bear]][0]
    mapping = {bull: 0, bear: 1, sideways: 2}
    regimes = np.array([mapping[c] for c in clusters])
    return regimes



