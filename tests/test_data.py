from stockai.data import parse_alpha_vantage_feed, parse_yahoo_news


def test_alpha_vantage_feed_picks_matching_ticker():
    feed = [
        {
            "title": "Apple beats estimates",
            "source": "Reuters",
            "time_published": "20260923T120000",
            "summary": "...",
            "ticker_sentiment": [
                {"ticker": "MSFT", "relevance_score": "0.1", "ticker_sentiment_score": "-0.5"},
                {"ticker": "AAPL", "relevance_score": "0.8", "ticker_sentiment_score": "0.42"},
            ],
        },
        {"title": "Market wrap", "ticker_sentiment": []},
    ]
    items = parse_alpha_vantage_feed(feed, "aapl")
    assert items[0]["sentiment"] == 0.42 and items[0]["relevance"] == 0.8
    assert items[1]["sentiment"] is None


def test_yahoo_news_handles_new_and_old_formats():
    raw = [
        {"content": {"title": "New format", "provider": {"displayName": "Yahoo"}, "pubDate": "2026-09-23"}},
        {"title": "Old format", "publisher": "Bloomberg", "providerPublishTime": 1758628800},
        {"content": {"title": ""}},
    ]
    items = parse_yahoo_news(raw)
    assert [i["title"] for i in items] == ["New format", "Old format"]
    assert items[0]["source"] == "Yahoo" and items[1]["source"] == "Bloomberg"
    assert all(i["sentiment"] is None for i in items)
