import pandas as pd

data_path = "C:/Users/HP/Desktop/CSV FILES/daily-min-temperatures.csv"
df = pd.read_csv(data_path, parse_dates=['Date'], index_col=['Date'])
series = df['Temp']

window_size = 7
moving_avg = series.rolling(window=window_size).mean()
forcast = moving_avg.iloc[-1]

print(f"Forcasting Next Value using {window_size} -day moving Average : {forcast: 3f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(series, label='Daily Min Temperature')
plt.plot(moving_avg, label= f"{window_size}-Daily Moving Average", linewidth =2)
plt.xlabel('Date')
plt.ylabel('Temperature( C )')
plt.title("Moving Average Forecasting on Daily Min Temperature (Melbourne)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()