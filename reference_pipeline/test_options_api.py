from polygon import RESTClient
from quintic_paths import load_simple_dotenv, project_root
import os

load_simple_dotenv(project_root(), override=True)
key = os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY")
client = RESTClient(api_key=key)

print("Testing snapshot...")
chain = list(client.list_snapshot_options_chain("AAPL", params={"limit": 2}))
for c in chain:
    g = getattr(c, "greeks", None)
    print("gamma:", getattr(g, "gamma", None))
    print("theta:", getattr(g, "theta", None))
    print("delta:", getattr(g, "delta", None))
    print("IV:", getattr(c, "implied_volatility", None))
    print("OI:", getattr(c, "open_interest", None))
    print("---")
    print("\nTesting historical snapshot...")
try:
    hist = list(client.list_snapshot_options_chain("AAPL", params={"limit": 2, "as_of": "2025-01-15"}))
    for c in hist:
        g = getattr(c, "greeks", None)
        print("gamma:", getattr(g, "gamma", None))
        print("theta:", getattr(g, "theta", None))
        print("OI:", getattr(c, "open_interest", None))
        print("---")
    print(f"Historical contracts returned: {len(hist)}")
except Exception as e:
    print(f"Historical error: {e}")
print("Done")