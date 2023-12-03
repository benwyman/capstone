from forecast_tool import *
from sentiment_tool import *
from web_scraper import *
import matplotlib.pyplot as plt
import pandas as pd


def press_52weekbutton(ticker, risk):
    risk_level = check_52_week_range(ticker, checkThresh(risk_tolerance))
    return risk_level


def checkThresh(risk):
    if risk == "L":
        return 5
    elif risk == "M":
        return 10
    elif risk == "H":
        return 15


# Example values for the parameters
query = "tsla"
risk_tolerance = "L"
amount_of_days = 5

press_52weekbutton(query, risk_tolerance)