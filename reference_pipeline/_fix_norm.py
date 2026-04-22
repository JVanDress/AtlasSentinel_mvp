import pandas as pd
p = r"C:\QUINTIC_V3\scripts\reference_pipeline\normalize_returns.py"
lines = open(p, encoding="utf-8").readlines()
out = []
for line in lines:
    out.append(line)
    if "_qc_fail" in line and "any(axis=1)" in line:
        out.append("\n")
        out.append('    bad_price_count = (df["close"] <= 0).sum()\n')
        out.append("    if bad_price_count > 0:\n")
        out.append('        df.loc[df["close"] <= 0, "close"] = float("nan")\n')
        out.append('        print(f"  Replaced {bad_price_count} zero/negative prices with NaN")\n')
open(p, "w", encoding="utf-8").writelines(out)
print("Fixed")
