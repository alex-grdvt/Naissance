"""
Fetches news headlines from Yahoo Finance and Google News, scores new
headlines with FinBERT, and merges the result with cached sentiment history
via portfolio_state. Called from main.py to build sentiment_df, which feeds
alpha_score (L3).
"""

import pandas as pd
import yfinance as yf
from transformers import pipeline
from gnews import GNews

_finbert = None


def _get_finbert():
    """Returns the shared FinBERT pipeline, loading it on first use because the transformer
    is expensive to load.
    """
    global _finbert
    if _finbert is None:
        _finbert = pipeline("text-classification", model="ProsusAI/finbert")
    return _finbert


def fetch_google_news(ticker):
    """Retrieve recent Google News headlines for a ticker."""
    google_news = GNews(language="en", max_results=30)
    articles = google_news.get_news(ticker)
    return [
        {"title": a["title"], "date": a["published date"]}
        for a in articles
    ]


def fetch_yfinance_news(ticker):
    """Retrieve recent news headlines from Yahoo Finance."""
    news = yf.Ticker(ticker).news
    out = []

    for article in news:
        content = article.get("content", {})
        title = content.get("title")
        raw_date = content.get("pubDate") or content.get("displayTime")

        if not title or not raw_date:
            continue

        date = pd.to_datetime(raw_date, utc=True, errors="coerce")
        if pd.isna(date):
            continue

        out.append({"title": title, "date": date.tz_localize(None)})

    return out


def fetch_sentiment_data(tickers, start_date, end_date):
    """
    Fetch headlines from Yahoo Finance and Google News, score new
    headlines with FinBERT, and merge with cached sentiment history.
    """
    from portfolio_state import log_sentiment, get_sentiment_history

    results = []

    for ticker in tickers:
        cached = get_sentiment_history(ticker, start_date, end_date)
        cached_dates = set(cached["date"].tolist()) if not cached.empty else set()

        # Each provider is fetched independently so one failure does not remove the other provider's headlines.
        headlines = []
        try:
            headlines += fetch_yfinance_news(ticker)
        except Exception as e:
            print(f"  [SENTIMENT] {ticker} — yfinance news failed: {e}")

        try:
            headlines += fetch_google_news(ticker)
        except Exception as e:
            print(f"  [SENTIMENT] {ticker} — google news failed: {e}")

        print(f"  [SENTIMENT] {ticker} — {len(headlines)} headlines "
              f"({len(cached_dates)} cached dates in window)")

        for item in headlines:
            date_str = pd.Timestamp(item["date"]).strftime("%Y-%m-%d")
            if date_str in cached_dates:
                continue

            try:
                sentiment = _get_finbert()(item["title"])[0]
                results.append({
                    "ticker": ticker,
                    "title": item["title"],
                    "date": date_str,
                    "label": sentiment["label"],
                    "score": sentiment["score"],
                })
                log_sentiment(ticker, date_str, sentiment["score"], sentiment["label"], 1)
            except Exception as e:
                print(f"  [SENTIMENT] {ticker} inference failed: {e}")

        for _, row in cached.iterrows():
            results.append({
                "ticker": ticker,
                "title": "",
                "date": row["date"],
                "label": row["label"],
                "score": row["score"],
            })

    columns = ["ticker", "title", "date", "label", "score"]
    return pd.DataFrame(results, columns=columns) if results else pd.DataFrame(columns=columns)


if __name__ == "__main__":
    df = fetch_sentiment_data(["AAPL", "MSFT"], "2026-05-06", "2026-06-07")
    print(df)