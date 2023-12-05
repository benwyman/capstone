import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz
from matplotlib import pyplot as plt


def calculate_moving_average(ticker):
    # Fetch data for the past 6 months
    stock_data = yf.download(ticker, period='6mo')

    # Calculate moving averages for specified k values
    k_values = [10, 13, 20, 30]
    for k in k_values:
        stock_data[f'Moving_Average_{k}'] = stock_data['Close'].rolling(window=k).mean()
    return stock_data


def compare_moving_average(ticker):
    stock_data = calculate_moving_average(ticker)
    current_price = stock_data['Close'].iloc[-1]
    avg_ma = stock_data[[f'Moving_Average_{k}' for k in [10, 13, 20, 30]]].iloc[-1].mean()

    print(f"Comparing moving average to current price for {ticker}")
    print(f"Average moving average: {avg_ma:.2f}, Current Price: {current_price:.2f}")

    risk_level = +1 if avg_ma > current_price else -1
    if risk_level == 1:
        print(f"Current price is below the average and will likely converge upwards.")
    else:
        print(f"Current price is above the average and will likely converge downwards.")
    return risk_level


def calculate_bollinger_bands(ticker):
    # Calculate the moving averages
    stock_data = calculate_moving_average(ticker)

    # Calculate Bollinger Bands for each moving average
    k_values = [10, 13, 30]
    for k in k_values:
        ma_col = f'Moving_Average_{k}'
        std_dev_col = f'Standard_Deviation_{k}'
        upper_band_col = f'Upper_Band_{k}'
        lower_band_col = f'Lower_Band_{k}'

        # Calculate standard deviation
        stock_data[std_dev_col] = stock_data['Close'].rolling(window=k).std()

        # Calculate upper and lower Bollinger Bands
        stock_data[upper_band_col] = stock_data[ma_col] + (stock_data[std_dev_col] * 2)
        stock_data[lower_band_col] = stock_data[ma_col] - (stock_data[std_dev_col] * 2)
    return stock_data


def calculate_bollinger_band_spread(ticker):
    # Calculate Bollinger Bands
    stock_data = calculate_bollinger_bands(ticker)

    # Identify upper and lower band columns
    upper_band_cols = [col for col in stock_data.columns if 'Upper_Band_' in col]
    lower_band_cols = [col for col in stock_data.columns if 'Lower_Band_' in col]

    # Calculate average upper and lower bands
    avg_upper_band = stock_data[upper_band_cols].mean().mean()
    avg_lower_band = stock_data[lower_band_cols].mean().mean()

    # Calculate percentage spread as a percentage of average closing price
    percent_spread = ((avg_upper_band - avg_lower_band) / stock_data['Close'].mean()) * 100
    print(f"Percentage spread of Bollinger Bands for {ticker} is {percent_spread:.2f}%")

    return percent_spread


def check_bollinger_bands_percent(ticker, threshold):
    percent_bollinger = calculate_bollinger_band_spread(ticker)
    print(f"Comparing Bollinger Bands spread to threshold for {ticker}")
    print(f"Bollinger Bands Spread: {percent_bollinger:.2f}%, Threshold: {threshold}%")
    if percent_bollinger > threshold:
        print(f"Bollinger spread is above current {threshold}% risk threshold, expect higher volatility.")
    else:
        print(f"Bollinger spread is below current {threshold}% risk threshold, expect lower volatility.")
    risk_level = -1 if percent_bollinger > threshold else 1
    return risk_level


def check_52_week_range(ticker, threshold):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    low_52w = hist['Low'].min()
    high_52w = hist['High'].max()
    current_price = hist['Close'].iloc[-1]
    avg_52w = (high_52w + low_52w) / 2
    percent_diff = ((current_price - avg_52w) / avg_52w) * 100

    print(f"Comparing 52-week price range average to current price for {ticker}")
    print(f"Current Price: {current_price}, 52-week Average: {avg_52w}, Percent Difference: {percent_diff:.2f}%")

    if percent_diff > threshold:
        print(f"Difference between 52-week average and current price is above the {threshold}% risk threshold.")
    else:
        print(f"Difference between 52-week average and current price is below the {threshold}% risk threshold.")
    risk_level = -1 if percent_diff > threshold else 1
    return risk_level


# def check_insider_purchases(ticker):
    # stock = yf.Ticker(ticker)
    # insider_trades = stock.insider_trades()
    # return +1 if insider_trades[-1] > 0 else -1


def get_top_institutional_holders(ticker):
    stock = yf.Ticker(ticker)
    institutional_holders = stock.institutional_holders
    if institutional_holders is not None and not institutional_holders.empty:
        return institutional_holders.iloc[0]
    else:
        return "No institutional holder data available."


def get_top_mutual_fund_holders(ticker):
    stock = yf.Ticker(ticker)
    mutual_fund_holders = stock.mutualfund_holders
    if mutual_fund_holders is not None and not mutual_fund_holders.empty:
        return mutual_fund_holders.iloc[0]
    else:
        return "No mutual fund holder data available."


def check_volume_growth(days, threshold, ticker):
    # Set timezone for New York
    ny_tz = pytz.timezone('America/New_York')

    # Get the current datetime in New York time
    current_datetime = datetime.now(ny_tz)

    # Fetch enough data to cover the days for analysis plus 2 extra days
    # (one for the control day, another in case today is incomplete)
    period = f"{days + 2}d"

    # Fetch historical data
    stock_data = yf.download(ticker, period=period)

    # If the current time is within trading hours or before the market opens,
    # drop the most recent day (today's data)
    if 9 <= current_datetime.hour < 20 or (current_datetime.hour < 9):
        stock_data = stock_data[:-1]

    # Ensure we only use the last 'days + 1' days of data, including the control day
    stock_data = stock_data.tail(days + 1)

    # Initialize a list to store daily growth percentages
    daily_growth_percentages = []

    # Initialize a flag to track the first iteration (control day)
    first_iteration = True

    # Initialize a variable to store previous day's volume
    prev_volume = None

    # Iterate over each day's data
    for date, row in stock_data.iterrows():
        current_volume = row['Volume']

        # Handle the control day separately without calculating growth
        if first_iteration:
            print(f"Control Day - Date: {date.strftime('%Y-%m-%d')}, Volume: {current_volume}")
            first_iteration = False
        else:
            # Calculate daily growth if previous day's volume is available and not zero
            if prev_volume and prev_volume != 0:
                daily_growth = ((current_volume - prev_volume) / prev_volume) * 100
                daily_growth_percentages.append(daily_growth)
                print(f"Date: {date.strftime('%Y-%m-%d')}, Volume: {current_volume}, Daily Growth: {daily_growth}%")

        # Update the previous volume for the next iteration
        prev_volume = current_volume

    # Calculate the average of these growth percentages
    average_growth = sum(daily_growth_percentages) / len(daily_growth_percentages) if daily_growth_percentages else 0
    print(f"Average growth in volume over {days} days is {average_growth}%")

    # Check if the average growth meets the target
    return abs(average_growth) >= threshold


def analyze_sentiment_and_volume_growth(df_summary, volume_growth_result, volume_threshold):
    print("Analyzing sentiment and volume growth...")
    risk_level = 0
    if volume_growth_result >= volume_threshold:
        print(f"The volume shift was over {volume_threshold}% so sentiment is more likely related to the share price. Risk level adjusted.")
        if df_summary['sentiment_counts']['Negative'] > df_summary['sentiment_counts']['Positive']:
            risk_level = -1
        else:
            risk_level = 1
    else:
        print(f"The volume shift was under {volume_threshold}% so sentiment is less likely related to the share price. Risk level unadjusted.")
    return risk_level


def calculate_macd(ticker, days):
    print(f"Calculating MACD for {ticker} over {days} days...")

    # fetch historical data for the specified period
    print(f"Downloading stock data for the past {days} days...")
    stock_data = yf.download(ticker, period=f'{days}d')
    prices = stock_data['Close']

    # calculate EMA for 12 days and 26 days
    ema_12 = prices.ewm(span=12, adjust=False).mean()
    ema_26 = prices.ewm(span=26, adjust=False).mean()

    # calculate MACD line
    macd_line = ema_12 - ema_26
    print("MACD line calculated.")

    # calculate signal line (EMA for 9 days of MACD line)
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    print("Signal line calculated.")

    return macd_line, signal_line


def calculate_macd_signal_crossings(ticker, days):
    print(f"Calculating MACD Signal Crossings and Risk Level for {ticker} over {days} days...")

    # Calculate MACD and signal line for the specified period
    macd_line, signal_line = calculate_macd(ticker, days)

    # Initialize risk level score
    risk_level = 0

    # Iterate through the data to adjust risk level
    for i in range(len(macd_line)):
        if macd_line.iloc[i] > signal_line.iloc[i]:
            risk_level += 1  # Add 1 for a buy signal (MACD above signal line)
            print(f"Day {i}: MACD above signal line, risk level increased to {risk_level}.")
        elif macd_line.iloc[i] < signal_line.iloc[i]:
            risk_level -= 1  # Subtract 1 for a sell signal (MACD below signal line)
            print(f"Day {i}: MACD below signal line, risk level decreased to {risk_level}.")

    # Handling the scenario where risk level is 0
    if risk_level == 0:
        current_position = 1 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1
        print(f"Final risk level is 0. Current day's MACD position used for final decision: {current_position}")
        return current_position

    print(f"Finished calculating MACD signal crossings. Final risk level: {risk_level}")
    return risk_level


def calculate_adx(ticker, days):
    # Retrieve historical data for the ticker
    data = yf.download(ticker, period=f'{days}d')

    # Initialize columns for calculations
    data['+DM'] = 0.0
    data['-DM'] = 0.0
    data['TR'] = 0.0
    data['Smoothed +DM'] = 0.0
    data['Smoothed -DM'] = 0.0
    data['Smoothed TR'] = 0.0
    data['+DI'] = 0.0
    data['-DI'] = 0.0
    data['DX'] = 0.0
    data['ADX'] = 0.0

    # Calculate +DM, -DM, and TR
    for i in range(1, len(data)):
        data.loc[data.index[i], '+DM'] = max(data['High'].iloc[i] - data['High'].iloc[i-1], 0)
        data.loc[data.index[i], '-DM'] = max(data['Low'].iloc[i-1] - data['Low'].iloc[i], 0)
        high_low = data['High'].iloc[i] - data['Low'].iloc[i]
        high_close = abs(data['High'].iloc[i] - data['Close'].iloc[i-1])
        low_close = abs(data['Low'].iloc[i] - data['Close'].iloc[i-1])
        data.loc[data.index[i], 'TR'] = max(high_low, high_close, low_close)

    # Calculate smoothed values
    for i in range(14, len(data)):
        data.loc[data.index[i], 'Smoothed +DM'] = (data['+DM'].iloc[i-13:i+1].sum() - data['+DM'].iloc[i-13:i+1].sum() / 14) + data['+DM'].iloc[i]
        data.loc[data.index[i], 'Smoothed -DM'] = (data['-DM'].iloc[i-13:i+1].sum() - data['-DM'].iloc[i-13:i+1].sum() / 14) + data['-DM'].iloc[i]
        data.loc[data.index[i], 'Smoothed TR'] = (data['TR'].iloc[i-13:i+1].sum() - data['TR'].iloc[i-13:i+1].sum() / 14) + data['TR'].iloc[i]

    # Calculate +DI, -DI, DX
    for i in range(14, len(data)):
        data.loc[data.index[i], '+DI'] = (data['Smoothed +DM'].iloc[i] / data['Smoothed TR'].iloc[i]) * 100
        data.loc[data.index[i], '-DI'] = (data['Smoothed -DM'].iloc[i] / data['Smoothed TR'].iloc[i]) * 100
        data.loc[data.index[i], 'DX'] = abs(data['+DI'].iloc[i] - data['-DI'].iloc[i]) / (data['+DI'].iloc[i] + data['-DI'].iloc[i]) * 100

    # Calculate ADX
    for i in range(28, len(data)):
        data.loc[data.index[i], 'ADX'] = data['DX'].iloc[i - 13:i + 1].mean()

    return data[['Smoothed +DM', 'Smoothed -DM', 'ADX']]


def graph_bollinger_bands(df_stock, ticker):
    plt.figure(figsize=(15, 7))
    plt.plot(df_stock['Close'], label='Closing Price', color='blue')

    k_values = [10, 13, 30]
    colors = ['green', 'orange', 'cyan']
    for i, k in enumerate(k_values):
        ma_col = f'Moving_Average_{k}'
        upper_band_col = f'Upper_Band_{k}'
        lower_band_col = f'Lower_Band_{k}'

        plt.plot(df_stock[ma_col], label=f'Moving Average ({k} periods)', color=colors[i], linestyle='--')
        plt.plot(df_stock[upper_band_col], label=f'Upper Bollinger Band ({k} periods)', color=colors[i], alpha=0.8)
        plt.plot(df_stock[lower_band_col], label=f'Lower Bollinger Band ({k} periods)', color=colors[i], alpha=0.8)

    plt.title(f'Bollinger Bands for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


def hello(string):
    print("hello", + string)