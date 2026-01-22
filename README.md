# Stock Price Prediction Web App

This project was developed as part of the **Technical Task Round** for the interview process at **Claysys Technologies**.

It is a full-stack web application that uses Deep Learning (LSTM) to predict stock price movements based on historical data.

## 🚀 Project Workflow

The application follows a streamlined data pipeline:

1.  **Data Acquisition**: 
    -   The app fetches the last 2 years of historical stock data using the `yfinance` API.
    -   *Robustness*: If the API fails (due to network or rate limits), the system automatically generates synthetic mock data to ensure the application remains functional for testing.
2.  **Preprocessing**: 
    -   Data is normalized (scaled between 0 and 1) to ensure efficient neural network training.
    -   It is transformed into time-series sequences (using a 60-day sliding window).
3.  **AI Model (LSTM)**: 
    -   A **Long Short-Term Memory (LSTM)** neural network, built with **PyTorch**, analyzes the sequential patterns.
    -   The model learns to predict the next day's closing price based on the previous 60 days.
4.  **Web Interface**:
    -   Built with **Flask**, the web app provides a user-friendly interface.
    -   It visualizes the actual historical prices vs. the model's predictions using **Matplotlib**.

## 🛠️ Installation & Setup

### Prerequisites
-   Python 3.9+
-   Git

### Steps to Run

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/ajayvenkatraman04/stock_price_prediction-Claysis.git
    cd stock_price_prediction-Claysis
    ```

2.  **Create a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    ```bash
    python app.py
    ```

5.  **Access the App**
    Open your browser and navigate to: `http://127.0.0.1:5000`

## 📂 Project Structure

-   `app.py`: The main Flask application server.
-   `model.py`: PyTorch LSTM model architecture.
-   `data_loader.py`: Handles data fetching, mock data generation, and preprocessing.
-   `train.py`: Script to retrain the model (optional execution).
-   `templates/`: HTML frontend files.
-   `static/`: CSS styling files.

---
*Developed by Ajay Venkatraman.*
