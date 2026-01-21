import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_mock_data(ticker, period="2y"):
    """
    Generate synthetic stock data for demonstration when API fails.
    """
    print(f"Generating mock data for {ticker}...")
    dates = pd.date_range(end=pd.Timestamp.now(), periods=730) # approx 2 years
    # Random walk with drift
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    price_path = 100 * np.cumprod(1 + returns)
    
    data = pd.DataFrame(data={'Close': price_path}, index=dates)
    return data

def download_data(ticker, period="2y"):
    """
    Download historical stock data from Yahoo Finance.
    Falls back to mock data if download fails or returns empty.
    """
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data is None or data.empty:
            raise ValueError("Empty data returned")
        return data
    except Exception as e:
        print(f"Error fetching data from Yahoo Finance: {e}")
        print("Falling back to Mock Data.")
        return generate_mock_data(ticker, period)

def preprocess_data(data, seq_length=60):
    """
    Preprocess data for LSTM:
    - Use 'Close' price.
    - Normalize using MinMaxScaler.
    - Create sequences.
    """
    if data is None or data.empty:
        return None, None, None
        
    prices = data['Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    X = []
    y = []
    
    for i in range(seq_length, len(scaled_prices)):
        X.append(scaled_prices[i-seq_length:i, 0])
        y.append(scaled_prices[i, 0])
        
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler
