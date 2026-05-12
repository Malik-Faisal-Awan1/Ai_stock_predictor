import numpy as np

def make_features_and_labels(df, prediction_horizon=1):
    """Creates rolling features; labels=1 if close_price[t+horizon] > close_price[t] else 0 (Up/Down)."""
    features = []
    labels = []
    for i in range(len(df) - prediction_horizon):
        window = df.iloc[i:i+10]   # Use past 10 days as features
        f = []
        f.extend(window['Close'].values)
        features.append(f)
        labels.append(int(df['Close'].iloc[i+prediction_horizon] > df['Close'].iloc[i]))
    return np.array(features), np.array(labels), [f"Close_day_-{i}" for i in range(10)]
