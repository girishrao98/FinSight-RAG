import ssl
import urllib.request
from dataclasses import dataclass
from time import sleep
from typing import List

import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup


@dataclass
class StockInfo:
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: int


def fetch_sp500_list() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
        },
    )
    with urllib.request.urlopen(req, context=ctx) as resp:
        html = resp.read().decode("utf-8")

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"class": "wikitable sortable mw-collapsible sticky-header"})
    if table is None:
        raise RuntimeError("Could not locate the S&P 500 company table on Wikipedia")

    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = []
    for tr in table.find_all("tr")[1:]:
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cols:
            rows.append(cols)

    df = pd.DataFrame(rows, columns=headers)
    df = df.rename(columns={
        "Symbol": "Ticker",
        "Security": "Name",
        "GICSSector": "Sector",
        "GICS Sub-Industry": "Industry",
    })
    df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)
    return df


def get_market_cap(ticker: str) -> int | None:
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    cap = info.get("marketCap") or info.get("market_cap")
    if cap is None:
        return None
    return int(cap)


def build_sector_rankings(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    results = []
    tickers = df["Ticker"].tolist()
    print(f"Fetching market cap data for {len(tickers)} tickers. This may take several minutes...")
    for idx, ticker in enumerate(tickers, start=1):
        market_cap = get_market_cap(ticker)
        if market_cap is None:
            print(f"  - skipping {ticker}: no market cap available")
        results.append({"Ticker": ticker, "MarketCap": market_cap})
        if idx % 25 == 0:
            print(f"  fetched {idx}/{len(tickers)}")
            sleep(0.3)

    caps = pd.DataFrame(results)
    merged = df.merge(caps, on="Ticker", how="left")
    merged = merged.dropna(subset=["MarketCap"]).copy()
    merged["MarketCap"] = merged["MarketCap"].astype(int)
    merged = merged.sort_values(["Sector", "MarketCap"], ascending=[True, False])

    grouped = merged.groupby("Sector").head(top_n).reset_index(drop=True)
    return grouped


def main() -> None:
    sp500_df = fetch_sp500_list()
    print("Loaded S&P 500 list with", len(sp500_df), "tickers")
    print(sp500_df[["Ticker", "Name", "Sector", "Industry"]].head())

    top_by_sector = build_sector_rankings(sp500_df, top_n=3)
    print("\nTop 3 stocks by market cap in each sector:")
    print(top_by_sector[["Sector", "Industry", "Ticker", "Name", "MarketCap"]].to_string(index=False))
    top_by_sector.to_csv("top_stocks_by_sector.csv", index=False)
    print("Saved top_stocks_by_sector.csv")


if __name__ == "__main__":
    main()
