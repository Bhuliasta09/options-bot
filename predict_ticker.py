import os
import yfinance as yf
import requests
import numpy as np
from datetime import datetime
import scipy.stats as si
from transformers import pipeline

# --- Configuration ---
TARGET_DAYS_OUT = 7 # 1 week out expiration target
MAX_CONTRACT_COST = 150.0 # Max budget per contract ($1.50 per share x 100)
RISK_FREE_RATE = 0.05

print("Loading AI Sentiment Model...")
sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert")

def calculate_delta(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return si.norm.cdf(d1, 0.0, 1.0)
    else:
        return si.norm.cdf(-d1, 0.0, 1.0)

def analyze_single_stock(ticker_symbol):
    ticker_symbol = ticker_symbol.upper().strip()
    print(f"\nAnalyzing {ticker_symbol}...")

    stock = yf.Ticker(ticker_symbol)
    try:
        current_price = stock.fast_info['lastPrice']
    except Exception as e:
        print(f"Error fetching price for {ticker_symbol}: {e}")
        return

    # 1. Fundamental Score
    fund_score = 0
    try:
        info = stock.info
        calc = (info.get("profitMargins", 0) * 100) - (info.get("debtToEquity", 0) / 10)
        fund_score = max(-25, min(25, calc))
    except:
        pass

    # 2. Sentiment Score
    sentiment_score = 0
    try:
        api_key = os.environ.get("NEWS_API_KEY")
        if api_key:
            url = f"https://newsapi.org/v2/everything?domains=bloomberg.com,wsj.com,cnbc.com&q={ticker_symbol}&sortBy=publishedAt&apiKey={api_key}"
            response = requests.get(url).json()
            if response.get("status") == "ok" and "articles" in response:
                headlines = [article["title"] for article in response["articles"][:15]]
                results = sentiment_analyzer(headlines)
                for res in results:
                    if res["label"] == "positive":
                        sentiment_score += 5
                    elif res["label"] == "negative":
                        sentiment_score -= 5
                sentiment_score = max(-25, min(25, sentiment_score))
    except:
        pass
        
    total_score = fund_score + sentiment_score
    print(f"Stock Score for {ticker_symbol}: {total_score} (Fundamentals: {fund_score}, Sentiment: {sentiment_score})")

    # Determine Directional Strategy based on score threshold
    if total_score >= 0:
        strategy = "call"
        bias = "CALL (Bullish)"
    else:
        strategy = "put"
        bias = "PUT (Bearish)"

    # 3. Scan Options Chain
    try:
        expirations = stock.options
        target_exp, best_diff = None, 9999

        for exp in expirations:
            days_to_exp = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
            if TARGET_DAYS_OUT < days_to_exp < best_diff:
                best_diff, target_exp = days_to_exp, exp

        if not target_exp:
            print("No suitable expiration date found within target window.")
            return

        chain = stock.option_chain(target_exp)
        options = chain.calls if strategy == "call" else chain.puts

        liquid_options = options[(options['volume'] > 10) & (options['openInterest'] > 50)].copy()
        if liquid_options.empty:
            print("No liquid options found meeting criteria.")
            return

        liquid_options['Delta'] = liquid_options.apply(
            lambda row: calculate_delta(current_price, row['strike'], best_diff / 365.0, RISK_FREE_RATE, row['impliedVolatility'], strategy), axis=1
        )

        if strategy == "call":
            ideal = liquid_options[(liquid_options['Delta'] >= 0.50) & (liquid_options['Delta'] <= 0.80)].copy()
        else:
            ideal = liquid_options[(liquid_options['Delta'] <= -0.50) & (liquid_options['Delta'] >= -0.80)].copy()

        if ideal.empty:
            ideal = liquid_options # Fallback to liquid chain if specific delta window is empty

        ideal['Spread'] = ideal['ask'] - ideal['bid']
        best_option = ideal.sort_values('Spread').iloc[0]

        ask_price = best_option['ask']
        total_cost = ask_price * 100

        # Calculate Target Sell Prices for Profit
        target_sell_50pct = ask_price * 1.50 # 50% gain target
        target_sell_100pct = ask_price * 2.00 # 100% gain target (double)

        print("\n" + "="*40)
        print(f" PREDICTION REPORT FOR: {ticker_symbol}")
        print("="*40)
        print(f"Current Stock Price: ${current_price:.2f}")
        print(f"Recommended Setup: {bias}")
        print(f"Contract Symbol: {best_option['contractSymbol']}")
        print(f"Expiration Date: {target_exp}")
        print(f"Strike Price: ${best_option['strike']:.2f}")
        print(f"Estimated Delta: {best_option['Delta']:.2f}")
        print("-" * 40)
        print(f"Suggested Entry: ${ask_price:.2f} per share (${total_cost:.2f} total cost)")
        print(f"Profit Target (+50%):${target_sell_50pct:.2f} per share (${target_sell_50pct * 100:.2f} total value)")
        print(f"Profit Target (+100%):${target_sell_100pct:.2f} per share (${target_sell_100pct * 100:.2f} total value)")
        print("="*40)

    except Exception as e:
        print(f"Error scanning options chain for {ticker_symbol}: {e}")

if __name__ == "__main__":
    user_ticker = input("Enter stock ticker symbol (e.g., AAPL, NVDA, MSFT): ")
    analyze_single_stock(user_ticker)
