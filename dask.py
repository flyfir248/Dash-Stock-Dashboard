# Import necessary libraries
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

# Load your data
tata = pd.read_csv("historical_stock_data/Tata_Motors_Ltd._historical_data.csv")

# Calculate monthly mean closing price
tata['Date'] = pd.to_datetime(tata['Date'])
tata.set_index('Date', inplace=True)
monthly_mean_close = tata['Close'].resample('M').mean()

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("TATA Motors Stock Analysis Dashboard"),

    # Visualization 1: Time Series of 'Close' Prices
    dcc.Graph(id='close-price-time-series',
              figure=px.line(tata, x='Date', y='Close', title='TATA Motors Close Price Over Time')),

    # Visualization 2: Rolling Mean and Std Dev of 'Close' Prices
    dcc.Graph(id='rolling-stats',
              figure=px.line(tata, x='Date', y=['Close', 'RollingMean', 'RollingStd'],
                             title='TATA Motors Close Price with Rolling Statistics')),

    # Visualization 3: Monthly Mean Closing Price
    dcc.Graph(id='monthly-mean',
              figure=px.line(monthly_mean_close, x=monthly_mean_close.index, y=monthly_mean_close.values,
                             title='Monthly Mean Closing Price of Tata Motors Ltd.')),

    # Visualization 4: Moving Averages
    dcc.Graph(id='moving-averages',
              figure=px.line(tata, x=tata.index, y=['Close', '30_Day_MA', '60_Day_MA'],
                             title='Tata Motors Closing Price and Moving Averages')),

    # Visualization 5: Daily Returns
    dcc.Graph(id='daily-returns',
              figure=px.line(tata, x=tata.index, y='Daily_Return', title='Tata Motors Daily Returns')),

    # Visualization 6: Volatility Analysis
    dcc.Graph(id='volatility',
              figure=px.line(tata, x=tata.index, y='Volatility', title='Tata Motors Volatility Analysis')),

    # Visualization 7: Predicted Opening and Closing Prices
    dcc.Graph(id='predicted-prices',
              figure={
                  'data': [
                      {'x': next_10_days.flatten(), 'y': predicted_open.flatten(), 'type': 'line',
                       'name': 'Predicted Open'},
                      {'x': next_10_days.flatten(), 'y': predicted_close.flatten(), 'type': 'line',
                       'name': 'Predicted Close'},
                  ],
                  'layout': {
                      'title': 'Predicted Opening and Closing Prices for the Next 10 Days'
                  }
              }
              ),

    # Text output for Buy/Sell suggestions
    html.Div(id='buy-sell-suggestions')
])

# Define callback for Buy/Sell suggestions
@app.callback(
    Output('buy-sell-suggestions', 'children'),
    [Input('predicted-prices', 'relayoutData')]
)
def update_buy_sell_suggestions(relayout_data):
    # Calculate price differences
    price_diff_open = predicted_open[-1] - tata['Open'].iloc[-1]
    price_diff_close = predicted_close[-1] - tata['Close'].iloc[-1]

    # Suggest whether buying is sensible
    suggestion_open = "Buy" if price_diff_open > 0 else "Sell"
    suggestion_close = "Buy" if price_diff_close > 0 else "Sell"

    return f"Predicted opening price change for the next 10 days: {price_diff_open:.2f} INR ({suggestion_open})\nPredicted closing price change for the next 10 days: {price_diff_close:.2f} INR ({suggestion_close})"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)