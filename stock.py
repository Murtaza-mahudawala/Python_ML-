import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Step 1: Download stock data
stock_data = yf.download('AAPL', start='2015-01-01', end='2023-01-01')

# Step 2: Prepare the data (Create a copy to avoid SettingWithCopyWarning)
data = stock_data[['Adj Close']].copy()
data['Target'] = data['Adj Close'].shift(-1)
data.dropna(inplace=True)

# Step 3: Split the data
X = data[['Adj Close']]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Build the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions and evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Step 6: Plot actual vs predicted prices
plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
plt.plot(y_test.index, predictions, label='Predicted Price', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs Predicted Stock Prices')
plt.legend()
plt.show()

# Step 7: Save the model
joblib.dump(model, 'stock_price_predictor.pkl')
print("Model saved as 'stock_price_predictor.pkl'")
