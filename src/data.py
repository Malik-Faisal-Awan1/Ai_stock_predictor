import yfinance as yf
import pandas as pd

def get_stock_data(ticker, years=2):
    df = yf.download(ticker, period=f"{years}y")
    if df.empty:
        return None
    return df
