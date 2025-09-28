from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import json
from prophet import Prophet

app = Flask(__name__)
CORS(app)

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def get_stock_data(self, symbol, period='5y'):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            return hist
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def create_features(self, df):
        """Create technical indicators as features"""
        df = df.copy()
        
        # Price features
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        
        # Price rate of change
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def prepare_data(self, df, lookback=30):
        """Prepare data for training"""
        features = ['Close', 'SMA_10', 'SMA_30', 'EMA_12', 'EMA_26', 
                   'Volatility', 'RSI', 'MACD', 'ROC']
        
        X = []
        y = []
        
        for i in range(lookback, len(df)):
            X.append(df[features].iloc[i-lookback:i].values.flatten())
            y.append(df['Close'].iloc[i])
        
        return np.array(X), np.array(y)
    
    def predict_stock(self, symbol, days=30):
        """Predict stock prices using Random Forest"""
        try:
            # Get historical data
            df = self.get_stock_data(symbol)
            if df is None or df.empty:
                return None
            
            # Create features
            df = self.create_features(df)
            
            if len(df) < 60:  # Need enough data
                return None
            
            predictions = self.advanced_prediction(df, days)
            
            if predictions is None:
                return None 
            
            # Prepare historical data for response
            historical_data = []
            for date, row in df.tail(100).iterrows():  # Last 100 days
                historical_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'price': round(row['Close'], 2),
                    'volume': int(row['Volume'])
                })
            
            # Prepare prediction data
            prediction_dates = []
            last_date = df.index[-1]
            for i in range(1, days + 1):
                next_date = last_date + timedelta(days=i)
                # Skip weekends
                while next_date.weekday() >= 5:
                    next_date += timedelta(days=1)
                prediction_dates.append(next_date.strftime('%Y-%m-%d'))
            
            prediction_data = []
            for date, price in zip(prediction_dates, predictions):
                prediction_data.append({
                    'date': date,
                    'predicted_price': round(price, 2)
                })
            
            return {
                'symbol': symbol,
                'historical_data': historical_data,
                'predictions': prediction_data,
                'current_price': round(df['Close'].iloc[-1], 2),
                'accuracy_score': self.calculate_accuracy(df)
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
        
    def advanced_prediction(self, df, days=30):
        """Improved prediction using multiple approaches"""
        try:
            # Approach 1: ARIMA-like rolling prediction
            predictions_arima = self.arima_style_prediction(df, days)
            
            # Approach 2: Trend-based prediction
            predictions_trend = self.trend_based_prediction(df, days)
            
            # Approach 3: Seasonal naive forecast
            predictions_naive = self.seasonal_naive_prediction(df, days)
            
            # Combine predictions with weighting
            final_predictions = []
            for i in range(days):
                # Weight recent approaches more heavily
                combined = (
                    predictions_arima[i] * 0.4 +
                    predictions_trend[i] * 0.4 + 
                    predictions_naive[i] * 0.2
                )
                final_predictions.append(combined)
            
            return final_predictions
            
        except Exception as e:
            print(f"Advanced prediction error: {e}")
            return self.fallback_prediction(df, days)

    def arima_style_prediction(self, df, days):
        """ARIMA-inspired prediction using recent trends"""
        prices = df['Close'].values
        
        # Calculate recent momentum
        recent_prices = prices[-30:]  # Last 30 days
        momentum = np.mean(recent_prices[-5:]) - np.mean(recent_prices[:5])
        
        # Use exponential smoothing
        alpha = 0.3  # Smoothing factor
        last_price = prices[-1]
        predictions = []
        
        for i in range(days):
            # Add momentum and random walk component
            next_price = last_price + momentum * 0.1 + np.random.normal(0, last_price * 0.01)
            predictions.append(next_price)
            last_price = next_price
        
        return predictions

    def trend_based_prediction(self, df, days):
        """Prediction based on technical analysis trends"""
        prices = df['Close'].values
        
        # Calculate short and long term trends
        short_trend = self.calculate_trend(prices[-10:])  # 10-day trend
        medium_trend = self.calculate_trend(prices[-30:])  # 30-day trend
        
        # Use weighted trend
        trend = (short_trend * 0.7 + medium_trend * 0.3)
        last_price = prices[-1]
        
        predictions = []
        for i in range(days):
            # Apply trend with decay (trend weakens over time)
            trend_strength = max(0.5, 1.0 - (i * 0.05))  # Trend decays over time
            next_price = last_price * (1 + trend * trend_strength)
            
            # Add some noise based on recent volatility
            recent_volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
            noise = np.random.normal(0, next_price * recent_volatility * 0.5)
            next_price += noise
            
            predictions.append(next_price)
            last_price = next_price
        
        return predictions

    def seasonal_naive_prediction(self, df, days):
        """Naive prediction considering weekly patterns"""
        prices = df['Close'].values
        last_price = prices[-1]
        
        # Calculate recent average daily return
        returns = np.diff(prices[-20:]) / prices[-21:-1]  # Last 20 returns
        avg_daily_return = np.mean(returns)
        std_daily_return = np.std(returns)
        
        predictions = []
        for i in range(days):
            # Use average return with some randomness
            daily_return = np.random.normal(avg_daily_return, std_daily_return * 0.5)
            next_price = last_price * (1 + daily_return)
            predictions.append(next_price)
            last_price = next_price
        
        return predictions

    def calculate_trend(self, prices):
        """Calculate price trend using linear regression"""
        if len(prices) < 2:
            return 0
        
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        
        # Normalize trend by average price
        avg_price = np.mean(prices)
        trend = slope / avg_price
        
        return trend

    def calculate_accuracy(self, df):
        """Calculate model accuracy on historical data"""
        try:
            # Use last 20% of data for validation
            split_idx = int(len(df) * 0.8)
            train_data = df.iloc[:split_idx]
            test_data = df.iloc[split_idx:]
            
            if len(test_data) < 10:
                return 50.0  # Default accuracy if not enough test data
            
            # Simple accuracy measure based on direction prediction
            correct_directions = 0
            for i in range(1, len(test_data)):
                actual_dir = 1 if test_data['Close'].iloc[i] > test_data['Close'].iloc[i-1] else 0
                
                # Simple prediction: assume continuation of recent trend
                recent_trend = self.calculate_trend(test_data['Close'].iloc[max(0, i-5):i].values)
                predicted_dir = 1 if recent_trend > 0 else 0
                
                if actual_dir == predicted_dir:
                    correct_directions += 1
            
            accuracy = (correct_directions / (len(test_data) - 1)) * 100
            return round(accuracy, 2)
            
        except:
            return 50.0

    def fallback_prediction(self, df, days):
        """Simple fallback prediction method"""
        last_price = df['Close'].iloc[-1]
        recent_volatility = df['Close'].pct_change().std()
        
        predictions = []
        for i in range(days):
            # Simple random walk with drift
            change = np.random.normal(0, recent_volatility)
            next_price = last_price * (1 + change)
            predictions.append(next_price)
        
        return predictions

predictor = StockPredictor()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'AAPL').upper()
        days = data.get('days', 30)
        
        result = predictor.predict_stock(symbol, days)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Failed to fetch data or make predictions'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_stocks():
    query = request.args.get('q', '').upper()
    # Simple search - in production, use a proper stock database
    popular_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NFLX', 'NVDA']
    
    if query:
        results = [stock for stock in popular_stocks if query in stock]
    else:
        results = popular_stocks[:5]
    
    return jsonify({'stocks': results})

if __name__ == '__main__':
    app.run(debug=True, port=5000)