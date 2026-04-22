python -c "
import pandas as pd
df = pd.read_parquet(r'Z:\\jvand\\QUINTIC_V3\\data\\stage\\stocks_universe_merged.parquet')
print('Shape:', df.shape)
print('Date range:', df['date'].min(), '→', df['date'].max())
print('Columns:', df.columns.tolist())
print('Sample tickers:', sorted(df['ticker'].unique()[:10]) if 'ticker' in df.columns else 'No ticker')
print('Has volume:', 'volume' in df.columns)
print('Has sector:', 'sector' in df.columns)
"