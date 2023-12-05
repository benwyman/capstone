from forecast_tool import *
from sentiment_tool import *
from web_scraper import *
import matplotlib.pyplot as plt
import pandas as pd


def main(query, start_date, end_date):
    risk_level = 0
    # Fetch news data
    df_news = fetch_news(query, start_date, end_date)

    # Print the fetched news DataFrame
    print("Fetched News Data:")
    print(df_news)

    # Analyze the data
    analyzed_df = analyze_data(df_news)

    # Print the analyzed sentiment data
    print("Analyzed Sentiment Data:")
    print(analyzed_df)

    # Generate and display graphs
    print("Generating graphs...")

    # Graph 1: Sentiment by Date
    fig1 = graph_sentiment_by_date(analyzed_df)
    plt.show()

    # Graph 2: Sentiment Frequency
    fig2 = graph_sentiment_frequency(analyzed_df)
    plt.show()

    # Graph 3: Polarity Distribution
    fig3 = graph_polarity_distribution(analyzed_df)
    plt.show()

    # Graph 4: Polarity by Date
    fig4 = graph_polarity_date(analyzed_df)
    plt.show()

    # Graph 5: Volume of Texts Over Time
    fig5 = graph_volume(analyzed_df)
    plt.show()

    print("Summarizing sentiment data...")
    df_summary = summarize_sentiment_data(analyzed_df)

    print("Generating market graph...")
    df_stock = calculate_bollinger_bands(query)

    # Call the new function from forecast_tool
    graph_bollinger_bands(df_stock, query)

    # Check 52 Week Price Range
    range_check_result = check_52_week_range(query, threshold=10)
    print("52 Week Range Check:", range_check_result)
    risk_level += range_check_result

    # Check Bollinger Bands Percent
    bollinger_check_result = check_bollinger_bands_percent(query, threshold=15)
    print("Bollinger Bands Percent Check:", bollinger_check_result)
    risk_level += bollinger_check_result

    # Compare Moving Average
    ma_comparison_result = compare_moving_average(query)
    print("Moving Average Comparison:", ma_comparison_result)
    risk_level += ma_comparison_result

    # Check MACD Signal Crossings
    macd_signal_crossings_result = calculate_macd_signal_crossings(query, 3)
    print(f"MACD Signal Crossings Risk for {query} over 30:", macd_signal_crossings_result)
    risk_level += macd_signal_crossings_result



    # Check Insider Purchases
    # insider_purchases_result = check_insider_purchases(query)
    # print("Insider Purchases Check:", insider_purchases_result)

    # Get top institutional holders
    # top_institutional_holders = get_top_institutional_holders(query)
    # print("Top Institutional Holders:")
    # print(top_institutional_holders)

    # Get top mutual fund holders
    # top_mutual_fund_holders = get_top_mutual_fund_holders(query)
    # print("Top Mutual Fund Holders:")
    # print(top_mutual_fund_holders)

    # ADX Check
    adx_days = 60  # Example: 60 days
    print(f"Calculating ADX for {query} over the last {adx_days} days...")
    adx_result = calculate_adx(query, adx_days)
    print("ADX Results:")
    print(adx_result.tail())  # Display the last few rows of the ADX result

    # Interpret ADX results
    latest_adx = adx_result['ADX'].iloc[-1]
    print(f"Latest ADX value for {query}: {latest_adx}")
    if latest_adx > 25:
        print("Strong trend detected.")
    elif latest_adx > 20:
        print("There is a trend.")
    else:
        print("Weak or no trend.")

    # Check Volume Growth
    volume_growth_result = check_volume_growth(3, 20, query)  # Example days and percent_target
    print("Volume Growth Check:", volume_growth_result)

    sentiment_result = analyze_sentiment_and_volume_growth(df_summary, volume_growth_result, volume_threshold=20)
    risk_level += sentiment_result

    print("Final Risk Level: ", risk_level)


if __name__ == '__main__':
    # Example values for the parameters
    query = "tsla"
    start_date = "2023-11-15"
    end_date = "2023-11-23"

    main(query, start_date, end_date)
