import yfinance as yf
import matplotlib.pyplot as plt

# Specify the stock symbol (TATASTEEL) and date range
stock_symbol = "TATASTEEL.BO"  # Use .BO for BSE stocks, .NS for NSE stocks
start_date = "2022-01-01"
end_date = "2023-08-01"

# Fetch historical stock data using yfinance
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Plot the historical stock data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], marker='o', linestyle='-', color='b', label='TATASTEEL Closing Price')
plt.title(f'Historical Stock Price for {stock_symbol}')
plt.xlabel('Date')
plt.ylabel('Closing Price (INR)')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
