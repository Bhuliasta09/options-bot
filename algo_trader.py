import os
import yfinance as yf
import requests
import numpy as np
from datetime import datetime
import scipy.stats as si
from transformers import pipeline

# --- Configuration ---
WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
WATCHLIST = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA", "AMZN"]
TARGET_DAYS_OUT = 30
RISK_FREE_RATE = 0.05

print("Loading AI Sentiment Model...")
sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert")

# --- 1. Math Function: Calculate Delta ---
def calculate_delta(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return si.norm.cdf(d1, 0.0, 1.0)
    else:
        return -si.norm.cdf(-d1, 0.0, 1.0) # Puts have negative Delta

# --- 2. Scoring System (Now returns -50 to +50) ---
def get_stock_score(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    fund_score, sentiment_score = 0, 0

    try:
       info = ticker.info
       calc = (info.get('profitMargins', 0) * 100) - (info.get('debtToEquity', 100) / 10)
       # Score fundamentals from -25 (Bad) to +25 (Good)
       fund_score = max(-25, min(25, calc - 25))
    except: pass

    try:
        news = ticker.news
        if news:
            headlines = [item['title'] for item in news[:5]]
            results = sentiment_analyzer(headlines)
            for res in results:
                if res['label'] == 'positive': sentiment_score += 5
                elif res['label'] == 'negative': sentiment_score -= 5
            # Score sentiment from -25 (Bearish) to +25 (Bullish)
            sentiment_score = max(-25, min(25, sentiment_score))
    except: pass

    # Total score ranges from -50 (Strong Sell) to +50 (Strong Buy)
    return fund_score + sentiment_score

# --- 3. Options Scanner (Find best Call OR Put) ---
def find_best_option(ticker_symbol, strategy="call"):
    stock = yf.Ticker(ticker_symbol)
    current_price = stock.fast_info['lastPrice']

    expirations = stock.options
    target_exp, best_diff = None, 9999

    for exp in expirations:
        days_to_exp = (datetime.strptime(exp, '%Y-%m-%d') - datetime.now()).days
        if TARGET_DAYS_OUT < days_to_exp < best_diff:
            best_diff, target_exp = days_to_exp, exp

    if not target_exp: return None

    chain = stock.option_chain(target_exp)
    options = chain.calls if strategy == "call" else chain.puts

    liquid_options = options[(options['volume'] > 50) & (options['openInterest'] > 100)].copy()
    if liquid_options.empty: return None

    liquid_options['Delta'] = liquid_options.apply(
    lambda row: calculate_delta(current_price, row['strike'], best_diff / 365.0, RISK_FREE_RATE, row['impliedVolatility'], strategy), axis=1
    )

    # Target ITM options: Calls (0.60 to 0.75), Puts (-0.60 to -0.75)
    if strategy == "call":
        ideal = liquid_options[(liquid_options['Delta'] >= 0.60) & (liquid_options['Delta'] <= 0.75)].copy()
    else:
        ideal = liquid_options[(liquid_options['Delta'] <= -0.60) & (liquid_options['Delta'] >= -0.75)].copy()

    if ideal.empty: return None

    ideal['Spread'] = ideal['ask'] - ideal['bid']
    best_option = ideal.sort_values('Spread').iloc[0]

    return {
        "contract": best_option['contractSymbol'],
        "expiration": target_exp,
        "strike": best_option['strike'],
        "ask_price": best_option['ask'],
        "cost": best_option['ask'] * 100,
        "delta": best_option['Delta'],
        "type": strategy.upper()
    }

# --- 4. Execution ---
print("Scanning market...")
scores = {symbol: get_stock_score(symbol) for symbol in WATCHLIST}

# Find most bullish and most bearish stock
most_bullish = max(scores, key=scores.get)
most_bearish = min(scores, key=scores.get)

bull_score = scores[most_bullish]
bear_score = scores[most_bearish]

option_data = None
strategy = None
target_stock = None
target_score = None

# If there is a very strong bullish signal (Score > 20), look for a Call
if bull_score > 20:
    print(f"Strong Bull Signal for {most_bullish} (Score: {bull_score}). Searching for Calls...")
    option_data = find_best_option(most_bullish, "call")
    strategy, target_stock, target_score = "CALL (Bullish)", most_bullish, bull_score

# If no good Call, check if there is a strong bearish signal (Score < -20) for a Put
elif bear_score < -20:
     print(f"Strong Bear Signal for {most_bearish} (Score: {bear_score}). Searching for Puts...")
     option_data = find_best_option(most_bearish, "put")
     strategy, target_stock, target_score = "PUT (Bearish)", most_bearish, bear_score

else:
    print("Market is mixed. No strong signals today.")

# --- 5. Discord Webhook ---
if option_data:
    color = 5763719 if option_data['type'] == "CALL" else 15548997 # Green for Call, Red for Put
    discord_message = {
        "username": "Algo-Trader Options Desk",
        "embeds": [{
            "title": f"🚨 {strategy} Setup: {target_stock}",
            "description": f"**AI Sentiment Score:** {target_score:.1f} \n*(+50 is Max Bullish, -50 is Max Bearish)*\n\n**Recommended Trade:**",
            "color": color,
            "fields": [
                {"name": "Contract Name", "value": f"`{option_data['contract']}`", "inline": False},
                {"name": "Expiration Date", "value": f"{option_data['expiration']}", "inline": True},
                {"name": "Strike Price", "value": f"${option_data['strike']:.2f}", "inline": True},
                {"name": "Delta", "value": f"{option_data['delta']:.2f}", "inline": True},
                {"name": "Entry Price (Ask)", "value": f"${option_data['ask_price']:.2f} per share", "inline": True},
                {"name": "Total Cost to Buy", "value": f"**${option_data['cost']:.2f}**", "inline": True},
            ]
        }]
    }
else:
    discord_message = {
        "username": "Algo-Trader Options Desk",
        "content": "📉 **Daily Scan Complete:** No stocks hit the strict +/- 20 score threshold, or no safe/liquid options contracts were found. Preserving capital today."
    }

requests.post(WEBHOOK_URL, json=discord_message)
