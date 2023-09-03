import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

try:
    # Load historical stock data
    tata = pd.read_csv("historical_stock_data/Tata_Motors_Ltd._historical_data.csv")
    tata['Date'] = pd.to_datetime(tata['Date'])
    tata.set_index('Date', inplace=True)

    # Calculate daily percentage change
    tata['DailyReturn'] = tata['Close'].pct_change()

    # Calculate volatility as the rolling standard deviation of daily returns
    tata['Volatility'] = tata['DailyReturn'].rolling(window=50, min_periods=1).std()

    # Calculate Rolling Mean and Rolling Std Deviation
    tata['RollingMean'] = tata['Close'].rolling(window=50, min_periods=1).mean()
    tata['RollingStd'] = tata['Close'].rolling(window=50, min_periods=1).std()

except Exception as e:
    print(f"Error loading data: {str(e)}")

# Initialize Dash app
app = dash.Dash(__name__)

# Define the layout for the main page
app.layout = html.Div([
    html.H1("Tata Motors Stock Analysis Dashboard"),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Data Summary & EDA', value='tab-1'),
        dcc.Tab(label='Close Price Analysis', value='tab-2'),
        dcc.Tab(label='Moving Averages & Returns', value='tab-3'),
        dcc.Tab(label='Volatility Analysis', value='tab-4'),
        dcc.Tab(label='Price Prediction (Next 10 Days)', value='tab-5'),
        dcc.Tab(label='Regression Models & Prediction', value='tab-6')
    ]),
    html.Div(id='tabs-content')
])

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        # Data Summary & EDA Page
        try:
            missing_values_heatmap = dcc.Graph(
                id='missing-values-heatmap',
                figure={
                    'data': [
                        go.Heatmap(
                            z=tata.isnull().values.T,  # Transpose the data to match x and y
                            x=tata.columns,
                            y=tata.index,
                            colorscale='Viridis',
                            showscale=False
                        )
                    ],
                    'layout': {
                        'title': 'Missing Values Heatmap'
                    }
                }
            )

            close_price_time_series = dcc.Graph(
                id='close-price-time-series',
                figure=px.line(tata, x=tata.index, y='Close', title='TATA Motors Close Price Over Time')
            )

            # Plot Rolling Mean
            rolling_mean_fig = px.line(tata, x=tata.index, y='RollingMean', title='Rolling Mean of Close Price')
            rolling_mean_fig.update_xaxes(title_text='Date')
            rolling_mean_fig.update_yaxes(title_text='Rolling Mean')

            # Plot Rolling Std Deviation
            rolling_std_fig = px.line(tata, x=tata.index, y='RollingStd', title='Rolling Std Deviation of Close Price')
            rolling_std_fig.update_xaxes(title_text='Date')
            rolling_std_fig.update_yaxes(title_text='Rolling Std Deviation')

            return html.Div([
                html.H2("Close Price Analysis"),
                html.Div([
                    rolling_mean_fig,
                    html.H3("Rolling Mean"),
                ]),
                html.Div([
                    rolling_std_fig,
                    html.H3("Rolling Std Deviation"),
                ]),
            ])
        except Exception as e:
            print(f"Error in tab-1: {str(e)}")

    elif tab == 'tab-2':
        # Close Price Analysis Page
        try:
            close_price_time_series = dcc.Graph(
                id='close-price-time-series',
                figure=px.line(tata, x=tata.index, y='Close', title='TATA Motors Close Price Over Time')
            )

            rolling_statistics = dcc.Graph(
                id='rolling-statistics',
                figure=px.line(tata, x=tata.index, y=['RollingMean', 'RollingStd'],
                                title='TATA Motors Close Price with 50-Day Rolling Statistics')
            )

            monthly_mean_close = dcc.Graph(
                id='monthly-mean-close',
                figure={
                    'data': [
                        go.Scatter(
                            x=tata.resample('M').mean().index,
                            y=tata.resample('M').mean()['Close'],
                            mode='lines+markers',
                            name='Monthly Mean Closing Price'
                        )
                    ],
                    'layout': {
                        'title': 'Monthly Mean Closing Price of Tata Motors Ltd.',
                        'xaxis': {'title': 'Date'},
                        'yaxis': {'title': 'Price (INR)'}
                    }
                }
            )

            return html.Div([
                html.H2("Close Price Analysis"),
                html.Div([
                    close_price_time_series,
                    html.H3("Close Price Time Series"),
                ]),
                html.Div([
                    rolling_statistics,
                    html.H3("Rolling Mean and Standard Deviation"),
                ]),
                html.Div([
                    monthly_mean_close,
                    html.H3("Monthly Mean Closing Price"),
                ]),
            ])
        except Exception as e:
            print(f"Error in tab-2: {str(e)}")

    elif tab == 'tab-3':
        # Moving Averages & Returns Page
        try:
            moving_averages = dcc.Graph(
                id='moving-averages',
                figure=px.line(tata, x=tata.index, y=['30_Day_MA', '60_Day_MA'],
                                title='Tata Motors Ltd. Closing Price and Moving Averages')
            )

            returns_calculation = dcc.Graph(
                id='returns-calculation',
                figure=px.line(tata, x=tata.index, y='DailyReturn',
                                title='Tata Motors Ltd. Daily Returns')
            )

            return html.Div([
                html.H2("Moving Averages & Returns"),
                html.Div([
                    moving_averages,
                    html.H3("Moving Averages"),
                ]),
                html.Div([
                    returns_calculation,
                    html.H3("Returns Calculation"),
                ]),
            ])
        except Exception as e:
            print(f"Error in tab-3: {str(e)}")

    elif tab == 'tab-4':
        # Volatility Analysis Page
        try:
            volatility_analysis = dcc.Graph(
                id='volatility-analysis',
                figure=px.line(tata, x=tata.index, y='Volatility',
                                title='Tata Motors Ltd. Volatility Analysis')
            )

            return html.Div([
                html.H2("Volatility Analysis"),
                html.Div([
                    volatility_analysis,
                    html.H3("Volatility Analysis"),
                ]),
            ])
        except Exception as e:
            print(f"Error in tab-4: {str(e)}")

    elif tab == 'tab-5':
        # Price Prediction (Next 10 Days) Page
        try:
            X = np.arange(len(tata)).reshape(-1, 1)
            y_open = tata['Open']
            y_close = tata['Close']

            X_train, X_test, y_open_train, y_open_test, y_close_train, y_close_test = train_test_split(
                X, y_open, y_close, test_size=0.2, shuffle=False
            )

            lr_open = LinearRegression()
            lr_open.fit(X_train, y_open_train)

            lr_close = LinearRegression()
            lr_close.fit(X_train, y_close_train)

            next_10_days = np.arange(len(tata), len(tata) + 10).reshape(-1, 1)
            predicted_open = lr_open.predict(next_10_days)
            predicted_close = lr_close.predict(next_10_days)

            price_diff_open = predicted_open[-1] - tata['Open'].iloc[-1]
            price_diff_close = predicted_close[-1] - tata['Close'].iloc[-1]

            suggestion_open = "Buy" if price_diff_open > 0 else "Sell"
            suggestion_close = "Buy" if price_diff_close > 0 else "Sell"

            open_price_prediction = dcc.Graph(
                id='open-price-prediction',
                figure={
                    'data': [
                        go.Scatter(
                            x=X_test[-10:].flatten(),
                            y=y_open_test[-10:],
                            mode='lines+markers',
                            name='Actual Opening Price'
                        ),
                        go.Scatter(
                            x=next_10_days.flatten(),
                            y=predicted_open,
                            mode='lines+markers',
                            name='Predicted Opening Price'
                        )
                    ],
                    'layout': {
                        'title': 'Opening Price Prediction for Next 10 Days',
                        'xaxis': {'title': 'Days'},
                        'yaxis': {'title': 'Price (INR)'}
                    }
                }
            )

            close_price_prediction = dcc.Graph(
                id='close-price-prediction',
                figure={
                    'data': [
                        go.Scatter(
                            x=X_test[-10:].flatten(),
                            y=y_close_test[-10:],
                            mode='lines+markers',
                            name='Actual Closing Price'
                        ),
                        go.Scatter(
                            x=next_10_days.flatten(),
                            y=predicted_close,
                            mode='lines+markers',
                            name='Predicted Closing Price'
                        )
                    ],
                    'layout': {
                        'title': 'Closing Price Prediction for Next 10 Days',
                        'xaxis': {'title': 'Days'},
                        'yaxis': {'title': 'Price (INR)'}
                    }
                }
            )

            return html.Div([
                html.H2("Price Prediction (Next 10 Days)"),
                html.Div([
                    open_price_prediction,
                    html.H3("Opening Price Prediction"),
                    html.P(f"Predicted opening price change for the next 10 days: {price_diff_open:.2f} INR ({suggestion_open})")
                ]),
                html.Div([
                    close_price_prediction,
                    html.H3("Closing Price Prediction"),
                    html.P(f"Predicted closing price change for the next 10 days: {price_diff_close:.2f} INR ({suggestion_close})")
                ]),
            ])
        except Exception as e:
            print(f"Error in tab-5: {str(e)}")

    elif tab == 'tab-6':
        # Regression Models & Prediction Page
        try:
            X = np.arange(len(tata)).reshape(-1, 1)
            y_open = tata['Open']
            y_close = tata['Close']

            X_train, X_test, y_open_train, y_open_test, y_close_train, y_close_test = train_test_split(
                X, y_open, y_close, test_size=0.2, shuffle=False
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regression": DecisionTreeRegressor(),
                "Random Forest Regression": RandomForestRegressor(),
                "Support Vector Regression": SVR(),
                "Gradient Boosting Regression": GradientBoostingRegressor(),
            }

            predictions_open = {}
            predictions_close = {}

            for model_name, model in models.items():
                model.fit(X_train, y_open_train)
                predicted_open = model.predict(X_test)
                predictions_open[model_name] = predicted_open

                model.fit(X_train, y_close_train)
                predicted_close = model.predict(X_test)
                predictions_close[model_name] = predicted_close

            next_10_days = np.arange(len(tata), len(tata) + 10).reshape(-1, 1)

            open_price_predictions = {}
            close_price_predictions = {}

            for model_name, model in models.items():
                model.fit(X, y_open)
                predicted_open = model.predict(next_10_days)
                open_price_predictions[model_name] = predicted_open

                model.fit(X, y_close)
                predicted_close = model.predict(next_10_days)
                close_price_predictions[model_name] = predicted_close

            open_price_comparison = dcc.Graph(
                id='open-price-comparison',
                figure={
                    'data': [
                        go.Scatter(
                            x=X_test.flatten(),
                            y=y_open_test,
                            mode='lines+markers',
                            name='Actual Opening Price'
                        ),
                        go.Scatter(
                            x=X_test.flatten(),
                            y=predictions_open["Linear Regression"],
                            mode='lines',
                            line=dict(dash='dot'),
                            name='Linear Regression Prediction'
                        ),
                        go.Scatter(
                            x=X_test.flatten(),
                            y=predictions_open["Decision Tree Regression"],
                            mode='lines',
                            line=dict(dash='dot'),
                            name='Decision Tree Regression Prediction'
                        ),
                        go.Scatter(
                            x=X_test.flatten(),
                            y=predictions_open["Random Forest Regression"],
                            mode='lines',
                            line=dict(dash='dot'),
                            name='Random Forest Regression Prediction'
                        ),
                        go.Scatter(
                            x=X_test.flatten(),
                            y=predictions_open["Support Vector Regression"],
                            mode='lines',
                            line=dict(dash='dot'),
                            name='Support Vector Regression Prediction'
                        ),
                        go.Scatter(
                            x=X_test.flatten(),
                            y=predictions_open["Gradient Boosting Regression"],
                            mode='lines',
                            line=dict(dash='dot'),
                            name='Gradient Boosting Regression Prediction'
                        ),
                    ],
                    'layout': {
                        'title': 'Opening Price Comparison for Regression Models',
                        'xaxis': {'title': 'Days'},
                        'yaxis': {'title': 'Price (INR)'}
                    }
                }
            )

            close_price_comparison = dcc.Graph(
                id='close-price-comparison',
                figure={
                    'data': [
                        go.Scatter(
                            x=X_test.flatten(),
                            y=y_close_test,
                            mode='lines+markers',
                            name='Actual Closing Price'
                        ),
                        go.Scatter(
                            x=X_test.flatten(),
                            y=predictions_close["Linear Regression"],
                            mode='lines',
                            line=dict(dash='dot'),
                            name='Linear Regression Prediction'
                        ),
                        go.Scatter(
                            x=X_test.flatten(),
                            y=predictions_close["Decision Tree Regression"],
                            mode='lines',
                            line=dict(dash='dot'),
                            name='Decision Tree Regression Prediction'
                        ),
                        go.Scatter(
                            x=X_test.flatten(),
                            y=predictions_close["Random Forest Regression"],
                            mode='lines',
                            line=dict(dash='dot'),
                            name='Random Forest Regression Prediction'
                        ),
                        go.Scatter(
                            x=X_test.flatten(),
                            y=predictions_close["Support Vector Regression"],
                            mode='lines',
                            line=dict(dash='dot'),
                            name='Support Vector Regression Prediction'
                        ),
                        go.Scatter(
                            x=X_test.flatten(),
                            y=predictions_close["Gradient Boosting Regression"],
                            mode='lines',
                            line=dict(dash='dot'),
                            name='Gradient Boosting Regression Prediction'
                        ),
                    ],
                    'layout': {
                        'title': 'Closing Price Comparison for Regression Models',
                        'xaxis': {'title': 'Days'},
                        'yaxis': {'title': 'Price (INR)'}
                    }
                }
            )

            open_price_predictions_graph = dcc.Graph(
                id='open-price-predictions',
                figure={
                    'data': [
                        go.Scatter(
                            x=next_10_days.flatten(),
                            y=open_price_predictions["Linear Regression"],
                            mode='lines+markers',
                            name='Linear Regression Prediction'
                        ),
                        go.Scatter(
                            x=next_10_days.flatten(),
                            y=open_price_predictions["Decision Tree Regression"],
                            mode='lines+markers',
                            name='Decision Tree Regression Prediction'
                        ),
                        go.Scatter(
                            x=next_10_days.flatten(),
                            y=open_price_predictions["Random Forest Regression"],
                            mode='lines+markers',
                            name='Random Forest Regression Prediction'
                        ),
                        go.Scatter(
                            x=next_10_days.flatten(),
                            y=open_price_predictions["Support Vector Regression"],
                            mode='lines+markers',
                            name='Support Vector Regression Prediction'
                        ),
                        go.Scatter(
                            x=next_10_days.flatten(),
                            y=open_price_predictions["Gradient Boosting Regression"],
                            mode='lines+markers',
                            name='Gradient Boosting Regression Prediction'
                        ),
                    ],
                    'layout': {
                        'title': 'Opening Price Predictions for Next 10 Days (Regression Models)',
                        'xaxis': {'title': 'Days'},
                        'yaxis': {'title': 'Price (INR)'}
                    }
                }
            )

            close_price_predictions_graph = dcc.Graph(
                id='close-price-predictions',
                figure={
                    'data': [
                        go.Scatter(
                            x=next_10_days.flatten(),
                            y=close_price_predictions["Linear Regression"],
                            mode='lines+markers',
                            name='Linear Regression Prediction'
                        ),
                        go.Scatter(
                            x=next_10_days.flatten(),
                            y=close_price_predictions["Decision Tree Regression"],
                            mode='lines+markers',
                            name='Decision Tree Regression Prediction'
                        ),
                        go.Scatter(
                            x=next_10_days.flatten(),
                            y=close_price_predictions["Random Forest Regression"],
                            mode='lines+markers',
                            name='Random Forest Regression Prediction'
                        ),
                        go.Scatter(
                            x=next_10_days.flatten(),
                            y=close_price_predictions["Support Vector Regression"],
                            mode='lines+markers',
                            name='Support Vector Regression Prediction'
                        ),
                        go.Scatter(
                            x=next_10_days.flatten(),
                            y=close_price_predictions["Gradient Boosting Regression"],
                            mode='lines+markers',
                            name='Gradient Boosting Regression Prediction'
                        ),
                    ],
                    'layout': {
                        'title': 'Closing Price Predictions for Next 10 Days (Regression Models)',
                        'xaxis': {'title': 'Days'},
                        'yaxis': {'title': 'Price (INR)'}
                    }
                }
            )

            return html.Div([
                html.H2("Regression Models & Prediction"),
                html.Div([
                    open_price_comparison,
                    html.H3("Opening Price Comparison for Regression Models"),
                ]),
                html.Div([
                    close_price_comparison,
                    html.H3("Closing Price Comparison for Regression Models"),
                ]),
                html.Div([
                    open_price_predictions_graph,
                    html.H3("Opening Price Predictions for Next 10 Days (Regression Models)"),
                ]),
                html.Div([
                    close_price_predictions_graph,
                    html.H3("Closing Price Predictions for Next 10 Days (Regression Models)"),
                ]),
            ])
        except Exception as e:
            print(f"Error in tab-6: {str(e)}")

if __name__ == '__main__':
    app.run_server(debug=True)
