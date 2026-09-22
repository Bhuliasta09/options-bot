import asyncio
import aiohttp
import os
import yfinance as yf
import requests
import numpy as np
from datetime import datetime
import scipy.stats as si
from transformers import pipeline
import pandas as pd
from io import StringIO

# --- Configuration ---
WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
TARGET_DAYS_OUT = 7 # 1 week out
MAX_CONTRACT_COST = 150.0 # Max budget $150 per contract
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
        return si.norm.cdf(-d1, 0.0, 1.0) # Puts have negative Delta

def get_stock_score(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    fund_score, sentiment_score = 0, 0

    try:
        info = ticker.info
        calc = (info.get("profitMargins", 0) * 100) - (info.get("debtToEquity", 0) / 10)
        fund_score = max(-25, min(25, calc))
    except:
        pass

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

    return fund_score + sentiment_score

def find_best_option(ticker_symbol, strategy="call"):
    stock = yf.Ticker(ticker_symbol)
    current_price = stock.fast_info['lastPrice']

    expirations = stock.options
    target_exp, best_diff = None, 9999

    for exp in expirations:
        days_to_exp = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
        if TARGET_DAYS_OUT < days_to_exp < best_diff:
            best_diff, target_exp = days_to_exp, exp

    if not target_exp:
        return None

    chain = stock.option_chain(target_exp)
    options = chain.calls if strategy == "call" else chain.puts

    liquid_options = options[(options['volume'] > 50) & (options['openInterest'] > 100)].copy()
    if liquid_options.empty:
        return None

    liquid_options['Delta'] = liquid_options.apply(
        lambda row: calculate_delta(current_price, row['strike'], best_diff / 365.0, RISK_FREE_RATE, row['impliedVolatility'], strategy), axis=1
    )

    if strategy == "call":
        ideal = liquid_options[(liquid_options['Delta'] >= 0.60) & (liquid_options['Delta'] <= 0.75)].copy()
    else:
        ideal = liquid_options[(liquid_options['Delta'] <= -0.60) & (liquid_options['Delta'] >= -0.75)].copy()

    if ideal.empty:
        return None

    ideal['Spread'] = ideal['ask'] - ideal['bid']
    best_option = ideal.sort_values('Spread').iloc[0]

    if (best_option['ask'] * 100) > MAX_CONTRACT_COST:
        return None

    return {
        "contract": best_option['contractSymbol'],
        "expiration": target_exp,
        "strike": best_option['strike'],
        "ask_price": best_option['ask'],
        "cost": best_option['ask'] * 100,
        "delta": best_option['Delta'],
        "type": strategy.upper()
    }

async def process_ticker(symbol):
    """Asynchronous worker to process individual stocks concurrently."""
    loop = asyncio.get_running_loop()
    print(f"Scanning {symbol}...")

    score = await loop.run_in_executor(None, get_stock_score, symbol)
    trade_info = None

    if score > 20:
        print(f"Strong Bull Signal for {symbol} (Score: {score}). Searching for Calls...")
        trade_info = await loop.run_in_executor(None, lambda: find_best_option(symbol, "call"))
        if trade_info:
            return {"stock": symbol, "score": score, "strategy": "CALL", "option": trade_info}
    elif score < -20:
        print(f"Strong Bear Signal for {symbol} (Score: {score}). Searching for Puts...")
        trade_info = await loop.run_in_executor(None, lambda: find_best_option(symbol, "put"))
        if trade_info:
            return {"stock": symbol, "score": score, "strategy": "PUT", "option": trade_info}

    return None

async def main():
    print("Fetching dynamic ticker list...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    table = pd.read_html(StringIO(html))[0]
    watchlist = table[table['CIK'].notnull()]['Symbol'].tolist()[:50]

    print(f"Starting async scan across {len(watchlist)} tickers...")
    tasks = [process_ticker(symbol) for symbol in watchlist]
    results = await asyncio.gather(*tasks)

    trade_opportunities = [r for r in results if r is not None]
    trade_opportunities.sort(key=lambda x: abs(x['score']), reverse=True)
    top_3_trades = trade_opportunities[:3]

    if top_3_trades:
        discord_embeds = []
        for trade in top_3_trades:
            color = 5763719 if trade['strategy'] == "CALL" else 15548997
            discord_embeds.append({
                "title": f"📈 {trade['strategy']} Setup: {trade['stock']}",
                "description": f"AI Sentiment Score: {trade['score']}\nProjected Profit (100% Target): ${round(trade['option']['ask_price'] * 100, 2)}",
                "color": color,
                "fields": [
                    {"name": "Contract Name", "value": f"{trade['option']['contract']}", "inline": False},
                    {"name": "Expiration", "value": f"{trade['option']['expiration']}", "inline": True},
                    {"name": "Strike", "value": f"${trade['option']['strike']:.2f}", "inline": True},
                    {"name": "Delta", "value": f"{trade['option']['delta']:.2f}", "inline": True},
                    {"name": "Entry (Ask)", "value": f"${trade['option']['ask_price']:.2f}", "inline": True},
                    {"name": "Total Cost", "value": f"${trade['option']['cost']:.2f}", "inline": True},
                ]
            })
        discord_message = {"username": "Algo-Trader Options Desk", "embeds": discord_embeds}
    else:
        discord_message = {
            "username": "Algo-Trader Options Desk",
            "content": "⚠️ Daily Scan Complete: No stocks hit the strict +/- 20 score threshold, or no safe/liquid options contracts were found. Preserving capital today."
        }

    requests.post(WEBHOOK_URL, json=discord_message)

if __name__ == "__main__":
    asyncio.run(main())
