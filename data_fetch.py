import requests
import pandas as pd
import os
from datetime import datetime

API_KEY = "q9lnyOSkS8OS8KlfCA8pTyHiiR2H5iDp"
TICKER = "GLD"
START_DATE = "2025-01-01"
END_DATE   = "2026-01-31"
OUTFILE    = f"./data/minute_data/{TICKER}.US_minute.csv"  # Output CSV path

def fetch_polygon_minute_data(ticker, from_date, to_date, api_key):
    """
    Fetch 1-minute aggregates from Polygon for the given ticker and date range.
    Returns a DataFrame with columns [timestamp, open, high, low, close, volume].
    """
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{from_date}/{to_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": "100000",
        "apiKey": api_key
    }

    all_rows = []
    while True:
        print(f"Requesting from {url} with limit={params['limit']}...")
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        # Extract results
        results = data.get("results", [])
        for bar in results:
            ts = datetime.utcfromtimestamp(bar["t"] / 1000.0)
            row = {
                "timestamp": ts,
                "open": bar["o"],
                "high": bar["h"],
                "low":  bar["l"],
                "close": bar["c"],
                "volume": bar["v"]
            }
            all_rows.append(row)

        # Stop if there's no pagination
        if "next_url" in data and data["next_url"]:
            url = data["next_url"]
        else:
            break

    df = pd.DataFrame(all_rows)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def main():
    df_new = fetch_polygon_minute_data(TICKER, START_DATE, END_DATE, API_KEY)
    if df_new.empty:
        print("No bars retrieved (possible future date or no coverage). Nothing to save.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

    if os.path.exists(OUTFILE):
        print(f"{OUTFILE} found. Loading existing data to append.")
        df_existing = pd.read_csv(OUTFILE, parse_dates=["timestamp"])
        last_time = df_existing["timestamp"].max()
        df_append = df_new[df_new["timestamp"] > last_time]
        if df_append.empty:
            print("No new bars to append.")
            return
        df_combined = pd.concat([df_existing, df_append], ignore_index=True)
        df_combined.sort_values("timestamp", inplace=True)
        df_combined.to_csv(OUTFILE, index=False)
        print(f"Appended {len(df_append)} new bars. Total rows: {len(df_combined)}")
    else:
        df_new.to_csv(OUTFILE, index=False)
        print(f"Created {OUTFILE} with {len(df_new)} rows.")

if __name__ == "__main__":
    main()
