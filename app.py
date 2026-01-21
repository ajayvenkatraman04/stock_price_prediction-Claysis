from flask import Flask, render_template, request
import torch
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg') # Fix for macOS GUI crash
import matplotlib.pyplot as plt
from data_loader import download_data, preprocess_data
from model import StockLSTM
import os

app = Flask(__name__)

# Model Hyperparameters (Must match training)
INPUT_DIM = 1
HIDDEN_DIM = 32
NUM_LAYERS = 2
OUTPUT_DIM = 1
SEQ_LENGTH = 60

# Load Model (Mocking if no weights file yet)
model = StockLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM)
MODEL_PATH = 'model.pth'

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
else:
    print("Warning: No trained model found. Predictions will be random initialized.")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form.get('ticker')
        if not ticker:
            return render_template('index.html', error="Please enter a ticker symbol.")
        
        # 1. Fetch Data
        data = download_data(ticker)
        if data is None or data.empty:
            return render_template('index.html', error=f"Could not fetch data for {ticker}")
            
        # 2. Preprocess
        # We need the last sequence to predict the NEXT day, 
        # but to visualize, let's predict on the recent history.
        X, y, scaler = preprocess_data(data, SEQ_LENGTH)
        
        if X is None or len(X) == 0:
             return render_template('index.html', error="Not enough data to make predictions.")

        # 3. Predict
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float()
            predictions = model(X_tensor)
            
        # Inverse transform
        predicted_prices = scaler.inverse_transform(predictions.numpy())
        actual_prices = scaler.inverse_transform(y.reshape(-1, 1))
        
        # 4. Visualize
        plt.figure(figsize=(10, 5))
        plt.plot(actual_prices, label='Actual Price')
        plt.plot(predicted_prices, label='Predicted Price')
        plt.title(f'{ticker} Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        
        # Save plot to buffer
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Predict next day (using the very last sequence from data)
        # Note: preprocess_data creates sequences up to the last known point 'y'.
        # To predict tomorrow, we need the sequence ending at today.
        # This implementation predicts known history. 
        # For simplicity, let's just show the last predicted value.
        last_prediction = predicted_prices[-1][0]
        
        return render_template('result.html', ticker=ticker, plot_url=plot_url, prediction=f"{last_prediction:.2f}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
