import pandas as pd

paths = {
    'extended_stock_prices': r'Z:\jvand\QUINTIC_V3\data\stage\extended_stock_prices.parquet',
    'all_stocks_daily_panel': r'Z:\jvand\QUINTIC_V3\data\stage\all_stocks_daily_panel.parquet',
    'stocks_universe_merged': r'Z:\jvand\QUINTIC_V3\data\stage\stocks_universe_merged.parquet',
}

for name, p in paths.items():
    print(f'\n=== {name} ===')
    df = pd.read_parquet(p)
    print(f'  shape: {df.shape}')
    print(f'  cols ({len(df.columns)}): {list(df.columns)}')
    if 'date' in df.columns:
        print(f'  date range: {df["date"].min()} to {df["date"].max()}')
        print(f'  unique dates: {df["date"].nunique()}')
    if 'ticker' in df.columns:
        print(f'  unique tickers: {df["ticker"].nunique()}')
