from tensorflow.keras.layers import LSTM, Dense, Dropout
from tradingview_ta import TA_Handler, Interval
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
import pandas as pd
import numpy as np
import ccxt
import time

def fetch_historical_data(symbol, timeframe, limit):
    exchange_instance = ccxt.binance()
    ohlcv = exchange_instance.fetch_ohlcv(symbol, timeframe, limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def add_indicators(df):
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['ema_20'] = ta.ema(df['close'], length=20)
    macd = ta.macd(df['close'])
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    return df

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def prepare_training_data(df, window_size):
    df = add_indicators(df).dropna()

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = df[['close', 'volume', 'rsi', 'ema_20', 'macd']]
    y = df['close'].shift(-1).dropna()
    X = X.iloc[:-1]
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    X_seq, y_seq = create_sequences(X_scaled, window_size)

    return X_seq, y_seq, scaler_X, scaler_y

def train_lstm_model(X_train, y_train, X_val, y_val, window_size, feature_size):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_size, feature_size)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    history = model.fit(X_train, y_train, 
                        epochs=50, 
                        batch_size=32, 
                        validation_data=(X_val, y_val), 
                        verbose=1)
    return model

def fetch_real_time_data(symbol, interval=Interval.INTERVAL_1_HOUR):
    analysis = TA_Handler(
        symbol=symbol,
        exchange=exchange,
        screener="crypto",
        interval=interval
    )
    indicators = analysis.get_analysis().indicators
    return indicators

def calculate_portfolio_value(prices):
    total_value = 0
    for asset, quantity in portfolio.items():
        if asset == "USDT":
            total_value += quantity
        else:
            total_value += quantity * prices.get(asset, 0)
    return total_value

def rebalance_portfolio(prices, target_allocation, portfolio):
    total_value = calculate_portfolio_value(prices)
    adjustments = {}

    for asset, target_percent in target_allocation.items():
        target_value = total_value * (target_percent / 100)
        current_value = portfolio[asset] * prices.get(asset, 1)
        adjustments[asset] = (target_value - current_value) / prices.get(asset, 1)

    return adjustments

def predict_prices(models, scalers, symbols, window_size=10):
    predictions = {}
    for symbol in symbols:
        indicators = fetch_real_time_data(symbol)
        df = pd.DataFrame([indicators])
        df_scaled = scalers[symbol]['X'].transform(df)
        df_seq = df_scaled.reshape((1, window_size, df_scaled.shape[1]))
        
        prediction_scaled = models[symbol].predict(df_seq)
        prediction = scalers[symbol]['y'].inverse_transform(prediction_scaled)
        predictions[symbol] = prediction[0][0]

    return predictions

def main():
    portfolio = {
        "USDT": 10000,  
        "BTC": 0,
        "ETH": 0,
        "SOL": 0 
    }

    target_allocation = {
        "USDT": 50,
        "BTC": 25, 
        "ETH": 15, 
        "SOL": 10  
    }

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    window_size = 10
    models = {}
    scalers = {}

    for symbol in symbols:
        print(f"Model training for {symbol}...")
        
        df = fetch_historical_data(symbol, '1d', 365)
        
        X_seq, y_seq, scaler_X, scaler_y = prepare_training_data(df, window_size)
        feature_size = X_seq.shape[2]
        
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        model = train_lstm_model(X_train, y_train, X_val, y_val, window_size, feature_size)
        
        models[symbol] = model
        scalers[symbol] = {'X': scaler_X, 'y': scaler_y}

    while True:
        prices = {}
        for symbol in symbols:
            data = fetch_real_time_data(symbol)
            prices[symbol.replace("USDT", "")] = data["close"]

        total_value = calculate_portfolio_value(prices)
        print(f"Portfolio total value: {total_value:.2f} USDT")

        predictions = predict_prices(models, scalers, symbols)
        print(f"Predicted future prices: {predictions}")

        adjustments = rebalance_portfolio(prices, target_allocation, portfolio)
        print(f"Rebalancing adjustments: {adjustments}")

        for asset, adjustment in adjustments.items():
            portfolio[asset] += adjustment
        print(f"New portfolio: {portfolio}")

        time.sleep(600)

if __name__ == "__main__":
    main()