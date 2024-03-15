import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Generate sample historical house price data
# Replace this with your own historical data
historical_prices = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]

# Fit ARIMA model to historical data
model = ARIMA(historical_prices, order=(1, 1, 1))  # Example order, you may need to tune this
model_fit = model.fit()

# Forecast house prices for the next 10 years (120 months)
forecast_steps = 120
forecast = model_fit.forecast(steps=forecast_steps)

# Plot historical data and forecast
plt.plot(np.arange(len(historical_prices)), historical_prices, label='Historical Prices')
plt.plot(np.arange(len(historical_prices), len(historical_prices) + forecast_steps), forecast, label='Forecast')
plt.xlabel('Months')
plt.ylabel('House Prices')
plt.title('House Price Forecast for the Next 10 Years')
plt.legend()
plt.show()

# Display forecasted prices for the next 10 years
print("Forecasted house prices for the next 10 years:")
for i, price in enumerate(forecast):
    print(f"Year {i // 12 + 1}, Month {i % 12 + 1}: ${price:.2f}")

