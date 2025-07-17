import pandas as pd

file_path = "C:/Users/HP/Desktop/CSV FILES/airline-passengers.csv"
df = pd.read_csv(file_path, parse_dates=['Month'], index_col=['Month'])

df.index.freq = 'MS'

from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df['Passengers'], order = (2,1,2))
model_fit = model.fit()

forecast = model_fit.forecast(steps = 1).iloc[0]
print(f"Forecasted Next Value Using ARINA : {forecast: 2f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(df['Passengers'], label='Original Data')
plt.plot(model_fit.fittedvalues, label = 'Fitted Values', linestyle = '--')
plt.title('ARIMA Forecasting on Airline Passengers')
plt.xlabel('Date')
plt.ylabel('Numbers of Passengers')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()