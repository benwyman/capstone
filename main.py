import datetime

import forecast_tool


def display_menu():
    print("1. 52-week price comparison for {ticker} with risk threshold {%}")
    print("2. Bollinger band spread for {ticker}")
    print("3. Compare Moving Average for {ticker}")
    print("4. MACD Signal Crossings for {ticker} over {days}")
    print("5. Bollinger Bands Percent for {ticker} with risk threshold {%}")
    print("6. ADX for {ticker} over {days}")
    print("7. Check sentiment of {ticker} over {days} with risk threshold {%}. (3 days recommended)")
    print("8. Check Holders for {ticker}")
    print("9. Calculate Risk for {ticker} between over {days} with risk threshold {%}")
    print("0. Print Menu")
    print("*. Exit")


def get_user_choice():
    user_input = input("Enter your choice: ")
    return user_input


def main():
    display_menu()
    while True:
        choice = get_user_choice()

        if choice == '1':
            ticker = input("Enter a stock ticker: ")
            threshold = input("Enter your risk threshold: ")
            forecast_tool.check_52_week_range(ticker, float(threshold))
        elif choice == '2':
            ticker = input("Enter a stock ticker: ")
            forecast_tool.calculate_bollinger_band_spread(ticker)
        elif choice == '3':
            ticker = input("Enter a stock ticker: ")
            forecast_tool.compare_moving_average(ticker)
        elif choice == '4':
            ticker = input("Enter a stock ticker: ")
            days = input("Enter the number of days to look back: ")
            forecast_tool.calculate_macd_signal_crossings(ticker, int(days))
        elif choice == '5':
            ticker = input("Enter a stock ticker: ")
            threshold = input("Enter your risk threshold: ")
            forecast_tool.check_bollinger_bands_percent(ticker, float(threshold))
        elif choice == '6':
            ticker = input("Enter a stock ticker: ")
            days = input("Enter the number of days to look back: ")
            forecast_tool.calculate_adx(ticker, int(days))
        elif choice == '7':
            ticker = input("Enter a stock ticker: ")
            days = input("Enter the number of days to look back: ")
            threshold = input("Enter your risk threshold: ")
            forecast_tool.total_sentiment_analysis(ticker, int(days), float(threshold))
        elif choice == '8':
            ticker = input("Enter a stock ticker: ")
            forecast_tool.check_holders(ticker)
        elif choice == '9':
            ticker = input("Enter a stock ticker: ")
            days = input("Enter the number of days to look back: ")
            threshold = input("Enter your risk threshold: ")
            forecast_tool.calculate_risk(ticker, int(days), float(threshold))
        elif choice == '0':
            display_menu()
        elif choice == '*':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a valid option.")


if __name__ == "__main__":
    main()
