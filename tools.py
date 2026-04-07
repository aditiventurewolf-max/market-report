# tools.py
import httpx
import yfinance as yf
from datetime import datetime, timedelta

# ── TOOL 1: Earnings data ──────────────────────────────────────────────────
# yfinance is a free library that wraps Yahoo Finance's undocumented API.
# It gives us income statements, balance sheets, earnings history.
# The 'async def' means this function is a coroutine — it can pause 
# while waiting for network I/O and let other things run.

async def get_earnings(ticker: str) -> dict:
    """
    Fetch recent earnings data for a ticker.
    Returns EPS, revenue, beat/miss vs estimates.
    """
    # yfinance is synchronous (blocking), so we use a thread executor
    # in agent.py to not block the event loop. For now, keep it simple.
    stock = yf.Ticker(ticker)
    
    try:
        info = stock.info
        earnings_dates = stock.earnings_dates
        
        # Get last 4 quarters of earnings
        recent_earnings = []
        if earnings_dates is not None and not earnings_dates.empty:
            for date, row in earnings_dates.head(4).iterrows():
                recent_earnings.append({
                    "date": str(date.date()),
                    "eps_estimate": round(row.get("EPS Estimate", 0) or 0, 3),
                    "eps_actual": round(row.get("Reported EPS", 0) or 0, 3),
                    "surprise_pct": round(row.get("Surprise(%)", 0) or 0, 2),
                })
        
        return {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE"),
            "revenue_ttm": info.get("totalRevenue"),
            "profit_margin": info.get("profitMargins"),
            "recent_earnings": recent_earnings,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ── TOOL 2: News + Sentiment ──────────────────────────────────────────────
# We're using NewsAPI (free tier: 100 req/day) to pull recent headlines.
# Then we score sentiment with VADER — a rule-based NLP model that works
# great for financial news. No ML training required.
# 
# KEY CONCEPT: sentiment scoring gives a number (-1 to +1) for each headline.
# We aggregate these to get a "mood" signal for the stock.

async def get_news_sentiment(ticker: str, company_name: str) -> dict:
    """
    Fetch recent news headlines and score their sentiment.
    """
    import os
    from newsapi import NewsApiClient
    
    # Import VADER here — install with: pip install vaderSentiment
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    analyzer = SentimentIntensityAnalyzer()
    news_api_key = os.getenv("NEWS_API_KEY")
    
    if not news_api_key:
        return {"error": "NEWS_API_KEY not set"}
    
    newsapi = NewsApiClient(api_key=news_api_key)
    
    # Search for news about this company in the last 7 days
    week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    try:
        response = newsapi.get_everything(
            q=f"{ticker} OR {company_name}",
            from_param=week_ago,
            language='en',
            sort_by='relevancy',
            page_size=10,
        )
        
        articles = response.get("articles", [])
        scored = []
        
        for article in articles:
            headline = article.get("title", "")
            description = article.get("description", "")
            text = f"{headline}. {description}"
            
            # VADER returns: neg, neu, pos, compound
            # compound is the overall score: -1 (very negative) to +1 (very positive)
            scores = analyzer.polarity_scores(text)
            
            scored.append({
                "headline": headline,
                "source": article.get("source", {}).get("name"),
                "published": article.get("publishedAt"),
                "sentiment_compound": round(scores["compound"], 3),
                "sentiment_label": (
                    "positive" if scores["compound"] >= 0.05
                    else "negative" if scores["compound"] <= -0.05
                    else "neutral"
                ),
                "url": article.get("url"),
            })
        
        # Aggregate: what's the average sentiment?
        if scored:
            avg_sentiment = sum(a["sentiment_compound"] for a in scored) / len(scored)
        else:
            avg_sentiment = 0
        
        return {
            "ticker": ticker,
            "articles_analyzed": len(scored),
            "average_sentiment": round(avg_sentiment, 3),
            "overall_label": (
                "positive" if avg_sentiment >= 0.05
                else "negative" if avg_sentiment <= -0.05
                else "neutral"
            ),
            "articles": scored,
        }
        
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ── TOOL 3: SEC Filing reader ─────────────────────────────────────────────
# EDGAR is the SEC's free public database. Every public company must file
# quarterly (10-Q) and annual (10-K) reports. We use the EDGAR full-text
# search API — completely free, no key needed.
# 
# We're NOT reading the whole filing (they're 200+ pages). We pull the
# most recent filing metadata and a summary section.

async def get_sec_filings(ticker: str) -> dict:
    """
    Fetch recent SEC filings metadata from EDGAR.
    """
    # EDGAR's company search API
    headers = {"User-Agent": "market-research-agent research@example.com"}
    
    async with httpx.AsyncClient() as client:
        # Step 1: find the company's CIK number (EDGAR's internal ID)
        search_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2024-01-01&forms=10-K,10-Q"
        
        try:
            # First get CIK via ticker lookup
            cik_url = f"https://data.sec.gov/submissions/CIK{ticker.upper()}.json"
            
            # EDGAR has a ticker→CIK mapping endpoint
            tickers_url = "https://www.sec.gov/files/company_tickers.json"
            resp = await client.get(tickers_url, headers=headers, timeout=10)
            tickers_data = resp.json()
            
            # Find CIK for our ticker
            cik = None
            for entry in tickers_data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    break
            
            if not cik:
                return {"ticker": ticker, "error": "CIK not found"}
            
            # Step 2: get their recent filings
            filings_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            resp = await client.get(filings_url, headers=headers, timeout=10)
            data = resp.json()
            
            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            descriptions = recent.get("primaryDocument", [])
            accession = recent.get("accessionNumber", [])
            
            # Filter for 10-K and 10-Q only
            relevant = []
            for i, form in enumerate(forms[:50]):  # check last 50 filings
                if form in ("10-K", "10-Q"):
                    acc_clean = accession[i].replace("-", "")
                    edgar_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{descriptions[i]}"
                    relevant.append({
                        "form": form,
                        "date": dates[i],
                        "url": edgar_url,
                        "accession": accession[i],
                    })
                if len(relevant) >= 4:
                    break
            
            return {
                "ticker": ticker,
                "company_name": data.get("name"),
                "recent_filings": relevant,
                "sic_description": data.get("sicDescription"),
            }
            
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}


# ── TOOL 4: Price history ─────────────────────────────────────────────────
# Simple price data: 52-week range, current price, volume, RSI.
# RSI (Relative Strength Index) is a momentum indicator:
#   > 70 = overbought (possibly due for pullback)
#   < 30 = oversold (possibly due for bounce)

async def get_price_history(ticker: str) -> dict:
    """
    Fetch price history and compute basic technical indicators.
    """
    stock = yf.Ticker(ticker)
    
    try:
        hist = stock.history(period="1y")
        
        if hist.empty:
            return {"ticker": ticker, "error": "No price history"}
        
        current_price = round(hist["Close"].iloc[-1], 2)
        week_52_high = round(hist["Close"].max(), 2)
        week_52_low = round(hist["Close"].min(), 2)
        avg_volume_30d = int(hist["Volume"].tail(30).mean())
        
        # Calculate RSI (14-day)
        delta = hist["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = round(100 - (100 / (1 + rs.iloc[-1])), 1)
        
        # Price vs 52-week range (0% = at low, 100% = at high)
        position_in_range = round(
            (current_price - week_52_low) / (week_52_high - week_52_low) * 100, 1
        )
        
        return {
            "ticker": ticker,
            "current_price": current_price,
            "52_week_high": week_52_high,
            "52_week_low": week_52_low,
            "position_in_52w_range_pct": position_in_range,
            "avg_volume_30d": avg_volume_30d,
            "rsi_14d": rsi,
            "rsi_signal": (
                "overbought" if rsi > 70
                else "oversold" if rsi < 30
                else "neutral"
            ),
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}
    