import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

class StockPredictionPipeline:
    def __init__(self, ticker, start_date=None, end_date=None, period='5y'):
        """
        Initialize the stock prediction pipeline
        
        Parameters:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        period (str): Period to download data for if start/end dates not provided
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.data = None
        self.models = {}
        self.scalers = {}
        self.prediction_days = 60  # Number of days to look back for predictions
        
    def download_data(self):
        """Download historical stock data"""
        print(f"Downloading data for {self.ticker}...")
        
        if self.start_date and self.end_date:
            self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        else:
            self.data = yf.download(self.ticker, period=self.period)
            
        # Basic data cleaning
        self.data = self.data.dropna()
        print(f"Downloaded {len(self.data)} days of data from {self.data.index.min().date()} to {self.data.index.max().date()}")
        return self.data
    
    def perform_eda(self):
        """Perform Exploratory Data Analysis"""
        if self.data is None:
            self.download_data()
            
        print("\n--- Exploratory Data Analysis ---")
        
        # Create a figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 16))
        
        # Plot 1: Stock price history
        axes[0].plot(self.data['Close'], label='Close Price')
        axes[0].plot(self.data['Open'], label='Open Price', alpha=0.6)
        axes[0].set_title(f'{self.ticker} Stock Price History')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: Daily returns
        daily_returns = self.data['Close'].pct_change() * 100
        axes[1].plot(daily_returns, label='Daily Returns', color='green')
        axes[1].set_title(f'{self.ticker} Daily Returns')
        axes[1].set_ylabel('Returns (%)')
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot 3: Trading volume
        axes[2].bar(self.data.index, self.data['Volume'], color='purple', alpha=0.5)
        axes[2].set_title(f'{self.ticker} Trading Volume')
        axes[2].set_ylabel('Volume')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.ticker}_eda.png")
        print(f"EDA plots saved to {self.ticker}_eda.png")
        
        # Basic statistical analysis
        print("\nBasic Statistics:")
        print(self.data.describe())
        
        print("\nCorrelation Matrix:")
        corr_matrix = self.data.corr()
        print(corr_matrix)
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f'{self.ticker} Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f"{self.ticker}_correlation.png")
        print(f"Correlation matrix saved to {self.ticker}_correlation.png")
        
        return self.data
    
    def engineer_features(self):
        """Engineer features for prediction model"""
        if self.data is None:
            self.download_data()
            
        print("\n--- Feature Engineering ---")
        
        # Create a new DataFrame for features
        feature_data = self.data.copy()
        
        # 1. Calculate moving averages
        for window in [5, 20, 50, 200]:
            feature_data[f'MA_{window}'] = feature_data['Close'].rolling(window=window).mean()
        
        # 2. Calculate price momentum (percentage change)
        for period in [1, 5, 10, 20]:
            feature_data[f'Momentum_{period}'] = feature_data['Close'].pct_change(periods=period) * 100
        
        # 3. Calculate volatility (standard deviation of returns)
        for window in [5, 20]:
            feature_data[f'Volatility_{window}'] = feature_data['Close'].pct_change().rolling(window=window).std() * 100
        
        # 4. Calculate RSI (Relative Strength Index)
        def calculate_rsi(data, window=14):
            delta = data.diff()
            up, down = delta.copy(), delta.copy()
            up[up < 0] = 0
            down[down > 0] = 0
            down = down.abs()
            
            avg_gain = up.rolling(window).mean()
            avg_loss = down.rolling(window).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        feature_data['RSI_14'] = calculate_rsi(feature_data['Close'])
        
        # 5. MACD (Moving Average Convergence Divergence)
        feature_data['EMA_12'] = feature_data['Close'].ewm(span=12, adjust=False).mean()
        feature_data['EMA_26'] = feature_data['Close'].ewm(span=26, adjust=False).mean()
        feature_data['MACD'] = feature_data['EMA_12'] - feature_data['EMA_26']
        feature_data['MACD_Signal'] = feature_data['MACD'].ewm(span=9, adjust=False).mean()
        
        # 6. Bollinger Bands
        feature_data['BB_Middle'] = feature_data['Close'].rolling(window=20).mean()
        feature_data['BB_Std'] = feature_data['Close'].rolling(window=20).std()
        feature_data['BB_Upper'] = feature_data['BB_Middle'] + (feature_data['BB_Std'] * 2)
        feature_data['BB_Lower'] = feature_data['BB_Middle'] - (feature_data['BB_Std'] * 2)
        feature_data['BB_Width'] = (feature_data['BB_Upper'] - feature_data['BB_Lower']) / feature_data['BB_Middle']
        
        # 7. Volume features
        feature_data['Volume_1d_Change'] = feature_data['Volume'].pct_change() * 100
        feature_data['Volume_MA_5'] = feature_data['Volume'].rolling(window=5).mean()
        feature_data['Volume_Ratio'] = feature_data['Volume'] / feature_data['Volume_MA_5']
        
        # Drop NaN values after feature creation
        feature_data = feature_data.dropna()
        
        print(f"Created {len(feature_data.columns) - len(self.data.columns)} new features")
        print("New features:", [col for col in feature_data.columns if col not in self.data.columns])
        
        # Plot some of the engineered features
        plt.figure(figsize=(14, 7))
        plt.plot(feature_data.index, feature_data['Close'], label='Close Price')
        plt.plot(feature_data.index, feature_data['MA_50'], label='50-day MA')
        plt.plot(feature_data.index, feature_data['MA_200'], label='200-day MA')
        plt.plot(feature_data.index, feature_data['BB_Upper'], label='Bollinger Upper', alpha=0.3)
        plt.plot(feature_data.index, feature_data['BB_Lower'], label='Bollinger Lower', alpha=0.3)
        plt.fill_between(feature_data.index, feature_data['BB_Upper'], feature_data['BB_Lower'], alpha=0.1)
        plt.title(f'{self.ticker} with Technical Indicators')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.ticker}_features.png")
        print(f"Feature visualization saved to {self.ticker}_features.png")
        
        self.feature_data = feature_data
        return feature_data
    
    def prepare_data_for_model(self, prediction_window=1, test_size=0.2):
        """
        Prepare data for modeling
        
        Parameters:
        prediction_window (int): How many days ahead to predict
        test_size (float): Proportion of data to use for testing
        """
        if not hasattr(self, 'feature_data'):
            self.engineer_features()
            
        print("\n--- Data Preparation for Modeling ---")
        
        # Define target
        target_column = 'Close'
        
        # Create future price column (what we want to predict)
        self.feature_data[f'Future_Close_{prediction_window}'] = self.feature_data[target_column].shift(-prediction_window)
        
        # Drop rows with NaN in target
        model_data = self.feature_data.dropna().copy()
        
        # Define features and target
        X = model_data.drop([f'Future_Close_{prediction_window}'], axis=1)
        y = model_data[f'Future_Close_{prediction_window}']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        # Scale the data
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        # Reshape y for scaling
        y_train_reshaped = y_train.values.reshape(-1, 1)
        y_test_reshaped = y_test.values.reshape(-1, 1)
        
        y_train_scaled = scaler_y.fit_transform(y_train_reshaped)
        y_test_scaled = scaler_y.transform(y_test_reshaped)
        
        # Store scalers for later use
        self.scalers['X'] = scaler_X
        self.scalers['y'] = scaler_y
        
        # Prepare data for models
        model_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train_scaled': y_train_scaled.flatten(),
            'y_test_scaled': y_test_scaled.flatten(),
            'feature_names': X.columns.tolist()
        }
        
        # Prepare sequences for LSTM
        X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train_scaled.flatten())
        X_test_seq, y_test_seq = self._create_sequences(X_test_scaled, y_test_scaled.flatten())
        
        model_data['X_train_seq'] = X_train_seq
        model_data['y_train_seq'] = y_train_seq
        model_data['X_test_seq'] = X_test_seq
        model_data['y_test_seq'] = y_test_seq
        
        self.model_data = model_data
        return model_data
    
    def _create_sequences(self, X, y):
        """Create sequences for time series models like LSTM"""
        X_seq, y_seq = [], []
        
        for i in range(self.prediction_days, len(X)):
            X_seq.append(X[i-self.prediction_days:i])
            y_seq.append(y[i])
            
        return np.array(X_seq), np.array(y_seq)
    
    def build_models(self):
        """Build and train multiple models"""
        if not hasattr(self, 'model_data'):
            self.prepare_data_for_model()
            
        print("\n--- Building Models ---")
        
        # 1. Linear Regression
        print("Training Linear Regression model...")
        lr_model = LinearRegression()
        lr_model.fit(self.model_data['X_train_scaled'], self.model_data['y_train_scaled'])
        self.models['Linear_Regression'] = lr_model
        
        # 2. Random Forest
        print("Training Random Forest model...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(self.model_data['X_train_scaled'], self.model_data['y_train_scaled'])
        self.models['Random_Forest'] = rf_model
        
        # 3. LSTM Neural Network
        print("Training LSTM model...")
        lstm_model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.prediction_days, self.model_data['X_train_scaled'].shape[1])),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(
            self.model_data['X_train_seq'], 
            self.model_data['y_train_seq'], 
            epochs=20, 
            batch_size=32,
            verbose=1
        )
        self.models['LSTM'] = lstm_model
        
        return self.models
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        if not self.models:
            self.build_models()
            
        print("\n--- Model Evaluation ---")
        
        results = {}
        
        # Evaluate Linear Regression and Random Forest
        for model_name in ['Linear_Regression', 'Random_Forest']:
            model = self.models[model_name]
            
            # Make predictions
            y_pred_scaled = model.predict(self.model_data['X_test_scaled'])
            
            # Inverse transform to get actual price values
            y_pred = self.scalers['y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_test = self.scalers['y'].inverse_transform(self.model_data['y_test_scaled'].reshape(-1, 1)).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[model_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'predictions': y_pred
            }
            
            print(f"\n{model_name} Results:")
            print(f"RMSE: ${rmse:.2f}")
            print(f"MAE: ${mae:.2f}")
            print(f"R² Score: {r2:.4f}")
            
        # Evaluate LSTM
        y_pred_scaled = self.models['LSTM'].predict(self.model_data['X_test_seq'])
        y_pred = self.scalers['y'].inverse_transform(y_pred_scaled).flatten()
        y_test = self.scalers['y'].inverse_transform(self.model_data['y_test_seq'].reshape(-1, 1)).flatten()
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results['LSTM'] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'predictions': y_pred
        }
        
        print("\nLSTM Results:")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Plot predictions
        plt.figure(figsize=(16, 8))
        
        # Get the dates for the test data
        test_dates = self.feature_data.index[-len(y_test):]
        
        plt.plot(test_dates, y_test, label='Actual Price', color='black')
        
        colors = {'Linear_Regression': 'blue', 'Random_Forest': 'green', 'LSTM': 'red'}
        
        for model_name, result in results.items():
            if model_name != 'LSTM':  # LSTM has fewer predictions due to sequence creation
                model_y_pred = result['predictions'][:len(y_test)]  # Ensure same length
                plt.plot(test_dates, model_y_pred, label=f'{model_name} Prediction', color=colors[model_name], alpha=0.7)
        
        # Plot LSTM predictions (fewer points due to sequences)
        plt.plot(test_dates[-len(results['LSTM']['predictions']):], results['LSTM']['predictions'], 
                 label='LSTM Prediction', color=colors['LSTM'], alpha=0.7)
        
        plt.title(f'{self.ticker} Stock Price Prediction - Model Comparison')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.ticker}_predictions.png")
        print(f"Prediction visualization saved to {self.ticker}_predictions.png")
        
        # Find best model
        best_model = min(results.items(), key=lambda x: x[1]['RMSE'])[0]
        print(f"\nBest model based on RMSE: {best_model}")
        
        self.results = results
        return results
    
    def make_future_predictions(self, days=30):
        """Make predictions for future days"""
        if not hasattr(self, 'results'):
            self.evaluate_models()
            
        print(f"\n--- Making Future Predictions for {days} Days ---")
        
        # Get the latest data point
        latest_data = self.feature_data.iloc[-1:].copy()
        
        # Find best model
        best_model_name = min(self.results.items(), key=lambda x: x[1]['RMSE'])[0]
        print(f"Using {best_model_name} for future predictions")
        
        future_dates = pd.date_range(start=self.feature_data.index[-1] + pd.Timedelta(days=1), periods=days)
        future_predictions = []
        
        current_data = latest_data.copy()
        
        for _ in range(days):
            # Scale the data
            X = self.scalers['X'].transform(current_data)
            
            # Make prediction
            if best_model_name == 'LSTM':
                # Create sequence
                X_seq = X.reshape(1, 1, X.shape[1])
                y_pred_scaled = self.models[best_model_name].predict(X_seq, verbose=0)
            else:
                y_pred_scaled = self.models[best_model_name].predict(X.reshape(1, -1))
            
            # Inverse transform
            y_pred = self.scalers['y'].inverse_transform(y_pred_scaled.reshape(-1, 1))[0][0]
            future_predictions.append(y_pred)
            
            # Update current data for next prediction (this is a simplification)
            # In a real-world scenario, you would need more complex logic to update all features
            current_data['Close'] = y_pred
            # You would need to update other columns as well based on the new values
        
        # Create a DataFrame with the predictions
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': future_predictions
        })
        future_df.set_index('Date', inplace=True)
        
        # Plot the predictions
        plt.figure(figsize=(16, 8))
        
        # Plot historical data
        plt.plot(self.feature_data.index[-90:], self.feature_data['Close'][-90:], label='Historical Close Price', color='blue')
        
        # Plot future predictions
        plt.plot(future_df.index, future_df['Predicted_Close'], label='Predicted Close Price', color='red', linestyle='--')
        
        plt.title(f'{self.ticker} Stock Price Prediction - Next {days} Days')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.ticker}_future_predictions.png")
        print(f"Future prediction visualization saved to {self.ticker}_future_predictions.png")
        
        print("\nFuture predictions:")
        print(future_df)
        
        return future_df
    
    def run_full_pipeline(self):
        """Run the entire pipeline from data download to future predictions"""
        self.download_data()
        self.perform_eda()
        self.engineer_features()
        self.prepare_data_for_model()
        self.build_models()
        self.evaluate_models()
        future_predictions = self.make_future_predictions()
        
        return future_predictions

# Example usage
if __name__ == "__main__":
    # Create pipeline instance
    pipeline = StockPredictionPipeline(ticker='AAPL', period='5y')
    
    # Run the full pipeline
    predictions = pipeline.run_full_pipeline()