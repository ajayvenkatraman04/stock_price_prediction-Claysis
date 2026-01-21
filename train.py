import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import download_data, preprocess_data
from model import StockLSTM
import numpy as np

# Hyperparameters
INPUT_DIM = 1
HIDDEN_DIM = 32
NUM_LAYERS = 2
OUTPUT_DIM = 1
SEQ_LENGTH = 60
LEARNING_RATE = 0.01
NUM_EPOCHS = 20

def train_model(ticker="AAPL"):
    print(f"Fetching data for {ticker}...")
    data = download_data(ticker)
    X, y, scaler = preprocess_data(data, SEQ_LENGTH)
    
    # Split train/test (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Convert to tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    
    model = StockLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        outputs = model(X_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')
            
    print("Training finished.")
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to model.pth")

if __name__ == "__main__":
    train_model()
