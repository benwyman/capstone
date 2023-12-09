import pandas as pd
import yfinance as yf
import web_scraper
import sentiment_tool
import datetime
import pytz
from matplotlib import pyplot as plt


def calculate_moving_average(ticker):
    """
    Downloads stock data for the past 6 months and calculates moving averages
    for different periods (10, 13, 20, 30 days).
    :param ticker: Stock ticker symbol.
    :return: DataFrame with moving averages added.
    """
    # fetch data for the past 6 months
    stock_data = yf.download(ticker, period='6mo')

    # calculate moving averages for specified k values
    k_values = [10, 13, 20, 30]
    for k in k_values:
        stock_data[f'Moving_Average_{k}'] = stock_data['Close'].rolling(window=k).mean()
    return stock_data


def compare_moving_average(ticker):
    """
    Compares the current stock price with its average moving averages calculated
    over various periods to assess the stock's performance.
    :param ticker: Stock ticker symbol.
    :return: Risk level indicator based on the comparison.
    """
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
    print("Moving Average Comparison:", risk_level)
    return risk_level


def calculate_bollinger_bands(ticker):
    """
    Calculates Bollinger Bands for the stock over various periods.
    This includes the upper and lower Bollinger Bands based on standard deviation.
    :param ticker: Stock ticker symbol.
    :return: DataFrame with Bollinger Bands added.
    """
    stock_data = calculate_moving_average(ticker)

    k_values = [10, 13, 30]
    for k in k_values:
        ma_col = f'Moving_Average_{k}'
        std_dev_col = f'Standard_Deviation_{k}'
        upper_band_col = f'Upper_Band_{k}'
        lower_band_col = f'Lower_Band_{k}'

        stock_data[std_dev_col] = stock_data['Close'].rolling(window=k).std()

        # Calculate upper and lower Bollinger Bands
        stock_data[upper_band_col] = stock_data[ma_col] + (stock_data[std_dev_col] * 2)
        stock_data[lower_band_col] = stock_data[ma_col] - (stock_data[std_dev_col] * 2)
    return stock_data


def calculate_bollinger_band_spread(ticker):
    """
        Calculates the percentage spread of Bollinger Bands to understand the stock's volatility.
        :param ticker: Stock ticker symbol.
        :return: Percentage spread of Bollinger Bands.
        """
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
    print("Bollinger Band Spread Comparison:", percent_spread)
    return percent_spread


def check_bollinger_bands_percent(ticker, threshold):
    """
    Compares the Bollinger Bands spread percentage against a threshold to evaluate risk.
    :param ticker: Stock ticker symbol.
    :param threshold: Threshold percentage to compare against.
    :return: Risk level based on Bollinger Bands spread.
    """
    percent_bollinger = calculate_bollinger_band_spread(ticker)
    print(f"Comparing Bollinger Bands spread to threshold for {ticker}")
    print(f"Bollinger Bands Spread: {percent_bollinger:.2f}%, Threshold: {threshold}%")
    if percent_bollinger > threshold:
        print(f"Bollinger spread is above current {threshold}% risk threshold, expect higher volatility.")
    else:
        print(f"Bollinger spread is below current {threshold}% risk threshold, expect lower volatility.")
    risk_level = -1 if percent_bollinger > threshold else 1
    print(f"Bollinger spread risk level: {risk_level}.")
    return risk_level


def check_52_week_range(ticker, threshold):
    """
    Analyzes the stock's current price in comparison with its 52-week range (high and low)
    to assess if it's within a certain risk threshold.
    :param ticker: Stock ticker symbol.
    :param threshold: Threshold percentage for comparison.
    :return: Risk level based on 52-week price range analysis.
    """
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
    print("52 Week Price Range Comparison:", risk_level)
    return risk_level


def get_top_institutional_holders(ticker):
    """
    Retrieves the top institutional holders of the given stock.
    :param ticker: Stock ticker symbol.
    :return: Information about the top institutional holder or a message if data is unavailable.
    """
    stock = yf.Ticker(ticker)
    institutional_holders = stock.institutional_holders
    if institutional_holders is not None and not institutional_holders.empty:
        return institutional_holders.iloc[0]
    else:
        return "No institutional holder data available."


def get_top_mutual_fund_holders(ticker):
    """
    Retrieves the top mutual fund holders of the given stock.
    :param ticker: Stock ticker symbol.
    :return: Information about the top mutual fund holder or a message if data is unavailable.
    """
    stock = yf.Ticker(ticker)
    mutual_fund_holders = stock.mutualfund_holders
    if mutual_fund_holders is not None and not mutual_fund_holders.empty:
        return mutual_fund_holders.iloc[0]
    else:
        return "No mutual fund holder data available."


def check_volume_growth(ticker, days, threshold):
    """
    Analyzes the stock's trading volume growth over a specified number of days
    and checks if it meets or exceeds a specified threshold.
    :param ticker: Stock ticker symbol.
    :param days: Number of days to analyze.
    :param threshold: Threshold percentage for volume growth.
    :return: Boolean indicating if volume growth meets the threshold.
    """
    ny_tz = pytz.timezone('America/New_York')

    current_datetime = datetime.datetime.now(ny_tz)

    # fetch enough data to cover the days for analysis plus 2 extra days
    # (one for the control day, another in case today is incomplete)
    period = f"{days + 2}d"

    stock_data = yf.download(ticker, period=period)

    # if the current time is within trading hours or before the market opens,
    # drop the most recent day (today's data)
    if 9 <= current_datetime.hour < 20 or (current_datetime.hour < 9):
        stock_data = stock_data[:-1]

    # ensure we only use the last 'days + 1' days of data, including the control day
    stock_data = stock_data.tail(days + 1)

    daily_growth_percentages = []

    # initialize a flag to track the first iteration (control day)
    first_iteration = True

    # initialize a variable to store previous day's volume
    prev_volume = None

    # iterate over each day's data
    for date, row in stock_data.iterrows():
        current_volume = row['Volume']

        # handle the control day separately without calculating growth
        if first_iteration:
            print(f"Control Day - Date: {date.strftime('%Y-%m-%d')}, Volume: {current_volume}")
            first_iteration = False
        else:
            # calculate daily growth if previous day's volume is available and not zero
            if prev_volume and prev_volume != 0:
                daily_growth = ((current_volume - prev_volume) / prev_volume) * 100
                daily_growth_percentages.append(daily_growth)
                print(f"Date: {date.strftime('%Y-%m-%d')}, Volume: {current_volume}, Daily Growth: {daily_growth}%")

        # update the previous volume for the next iteration
        prev_volume = current_volume

    # calculate the average of these growth percentages
    average_growth = sum(daily_growth_percentages) / len(daily_growth_percentages) if daily_growth_percentages else 0
    print(f"Average growth in volume over {days} days is {average_growth}%")

    # check if the average growth meets the target
    return abs(average_growth) >= threshold


def analyze_sentiment_and_volume_growth(df_summary, volume_growth_result, volume_threshold):
    """
    Analyzes the relationship between stock sentiment and volume growth,
    adjusting the risk level based on volume changes.
    :param df_summary: DataFrame summarizing sentiment data.
    :param volume_growth_result: Result from volume growth analysis.
    :param volume_threshold: Threshold for considering volume growth significant.
    :return: Adjusted risk level based on sentiment and volume growth analysis.
    """
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


def total_sentiment_analysis(ticker, days, threshold):
    """
    Conducts a total sentiment analysis of the stock over a specified number of days.
    Includes fetching news data, analyzing sentiment, and summarizing results.
    :param ticker: Stock ticker symbol.
    :param days: Number of days for sentiment analysis.
    :param threshold: Risk threshold for analysis.
    :return: Final risk level based on sentiment analysis.
    """
    # calculate the start and end date based on how many days the user wants to look back
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    df_news = web_scraper.fetch_news(ticker, start_date, end_date)
    print("Fetched News Data:")
    print(df_news)

    analyzed_df = sentiment_tool.analyze_data(df_news)

    print("Analyzed Sentiment Data:")
    print(analyzed_df)

    print("Summarizing sentiment data...")
    df_summary = sentiment_tool.summarize_sentiment_data(analyzed_df)

    volume_growth_result = check_volume_growth(ticker, days, threshold)  # Example days and percent_target
    print("Volume Growth Check:", volume_growth_result)

    sentiment_result = analyze_sentiment_and_volume_growth(df_summary, volume_growth_result, volume_threshold=20)
    risk_level = sentiment_result
    return risk_level


def calculate_macd(ticker):
    """
    Calculates the Moving Average Convergence Divergence (MACD) for the stock
    over a specified period.
    :param ticker: Stock ticker symbol.
    :param days: Number of days for MACD calculation.
    :return: Tuple of MACD line and signal line.
    """
    # fetch historical data for the specified period
    stock_data = yf.download(ticker, period=30)
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
    """
    Calculates MACD signal crossings and evaluates the risk level for the stock
    based on these crossings over a specified period.
    :param ticker: Stock ticker symbol.
    :param days: Number of days to analyze.
    :return: Overall risk level based on MACD signal crossings.
    """
    print(f"Calculating MACD Signal Crossings and Risk Level for {ticker} over {days}  days...")

    # calculate MACD and signal line for the specified period
    macd_line, signal_line = calculate_macd(ticker)

    risk_level = 0

    # iterate through the data to adjust risk level
    for i in range(len(macd_line)):
        if macd_line.iloc[i] > signal_line.iloc[i]:
            risk_level += 1  # add 1 for a sell signal (MACD above signal line)
            print(f"Day {i}: MACD above signal line, risk level increased to {risk_level}.")
        elif macd_line.iloc[i] < signal_line.iloc[i]:
            risk_level -= 1  # subtract 1 for a buy signal (MACD below signal line)
            print(f"Day {i}: MACD below signal line, risk level decreased to {risk_level}.")

    # handling the scenario where risk level is 0
    if risk_level == 0:
        current_position = 1 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1
        print(f"Final risk level is 0. Current day's MACD position used for final decision: {current_position}")
        return current_position

    print(f"MACD Signal Crossings Risk for {ticker} over {days}days: ", risk_level)
    return risk_level


def calculate_adx(ticker, days):
    """
    Calculates the Average Directional Index (ADX) for the stock over a specified period,
    which helps in determining the strength of a trend.
    :param ticker: Stock ticker symbol.
    :param days: Number of days for ADX calculation.
    :return: DataFrame with ADX and related indicators.
    """
    print(f"Calculating ADX for {ticker} over the last {days} days...")
    data = yf.download(ticker, period=f'{days}d')

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

    # calculate +DM, -DM, and TR
    for i in range(1, len(data)):
        data.loc[data.index[i], '+DM'] = max(data['High'].iloc[i] - data['High'].iloc[i-1], 0)
        data.loc[data.index[i], '-DM'] = max(data['Low'].iloc[i-1] - data['Low'].iloc[i], 0)
        high_low = data['High'].iloc[i] - data['Low'].iloc[i]
        high_close = abs(data['High'].iloc[i] - data['Close'].iloc[i-1])
        low_close = abs(data['Low'].iloc[i] - data['Close'].iloc[i-1])
        data.loc[data.index[i], 'TR'] = max(high_low, high_close, low_close)

    # calculate smoothed values
    for i in range(14, len(data)):
        data.loc[data.index[i], 'Smoothed +DM'] = (data['+DM'].iloc[i-13:i+1].sum() - data['+DM'].iloc[i-13:i+1].sum() / 14) + data['+DM'].iloc[i]
        data.loc[data.index[i], 'Smoothed -DM'] = (data['-DM'].iloc[i-13:i+1].sum() - data['-DM'].iloc[i-13:i+1].sum() / 14) + data['-DM'].iloc[i]
        data.loc[data.index[i], 'Smoothed TR'] = (data['TR'].iloc[i-13:i+1].sum() - data['TR'].iloc[i-13:i+1].sum() / 14) + data['TR'].iloc[i]

    # calculate +DI, -DI, DX
    for i in range(14, len(data)):
        data.loc[data.index[i], '+DI'] = (data['Smoothed +DM'].iloc[i] / data['Smoothed TR'].iloc[i]) * 100
        data.loc[data.index[i], '-DI'] = (data['Smoothed -DM'].iloc[i] / data['Smoothed TR'].iloc[i]) * 100
        data.loc[data.index[i], 'DX'] = abs(data['+DI'].iloc[i] - data['-DI'].iloc[i]) / (data['+DI'].iloc[i] + data['-DI'].iloc[i]) * 100

    # calculate ADX
    for i in range(28, len(data)):
        data.loc[data.index[i], 'ADX'] = data['DX'].iloc[i - 13:i + 1].mean()

    print("ADX Results:")
    print(data.tail())

    # interpret ADX results
    latest_adx = data['ADX'].iloc[-1]
    print(f"Latest ADX value for {ticker}: {latest_adx}")
    if latest_adx > 25:
        print("Strong trend detected.")
    elif latest_adx > 20:
        print("There is a trend.")
    else:
        print("Weak or no trend.")

    return data[['Smoothed +DM', 'Smoothed -DM', 'ADX']]


def graph_bollinger_bands(df_stock, ticker):
    """
    Generates a plot of Bollinger Bands for the stock.
    :param df_stock: DataFrame containing stock data with Bollinger Bands.
    :param ticker: Stock ticker symbol.
    """
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


def check_holders(query):
    """
    Checks and prints information about the top institutional and mutual fund holders
    of a given stock.
    :param query: Stock ticker symbol.
    """
    # get top institutional holders
    top_institutional_holders = get_top_institutional_holders(query)
    print("Top Institutional Holders:")
    print(top_institutional_holders)

    # get top mutual fund holders
    top_mutual_fund_holders = get_top_mutual_fund_holders(query)
    print("Top Mutual Fund Holders:")
    print(top_mutual_fund_holders)


def calculate_risk(query, days, threshold):
    """
    Calculates the overall risk of a stock based on various factors like sentiment analysis,
    Bollinger Bands, 52-week price range, MACD signal crossings, and ADX over a specified period.
    :param query: Stock ticker symbol.
    :param days: Number of days for analysis.
    :param threshold: Threshold for risk calculation.
    :return: Calculated risk level for the stock.
    """
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(int(days))
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    risk_level = 0
    df_news = web_scraper.fetch_news(query, start_date, end_date)

    print("Fetched News Data:")
    print(df_news)

    analyzed_df = sentiment_tool.analyze_data(df_news)

    print("Analyzed Sentiment Data:")
    print(analyzed_df)

    print("Summarizing sentiment data...")
    df_summary = sentiment_tool.summarize_sentiment_data(analyzed_df)

    df_stock = calculate_bollinger_bands(query)

    # graph the bollinger bands
    graph_bollinger_bands(df_stock, query)

    # check 52 Week Price Range
    range_check_result = check_52_week_range(query, threshold)
    risk_level += range_check_result

    # check Bollinger Bands Percent Spread
    bollinger_check_result = check_bollinger_bands_percent(query, threshold)
    risk_level += bollinger_check_result

    # compare moving average
    ma_comparison_result = compare_moving_average(query)
    risk_level += ma_comparison_result

    # check MACD Signal Crossings
    macd_signal_crossings_result = calculate_macd_signal_crossings(query, days)
    risk_level += macd_signal_crossings_result

    # ADX Check
    adx_days = 30  # Example: 60 days
    adx_result = calculate_adx(query, adx_days)

    # check volume growth
    volume_growth_result = check_volume_growth(query, days, threshold)  # Example days and percent_target
    print("Volume Growth Check:", volume_growth_result)

    sentiment_result = analyze_sentiment_and_volume_growth(df_summary, volume_growth_result, volume_threshold=20)
    risk_level += sentiment_result

    print("Final Risk Level: ", risk_level)
    return risk_level
