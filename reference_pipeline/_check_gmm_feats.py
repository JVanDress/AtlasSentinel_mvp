import pandas as pd
df = pd.read_parquet(r"Z:\jvand\QUINTIC_V3\data\stage\cleanroom_ou_panel.parquet")
feats = [
    "close_frac_diff",
    "log_return_1d",
    "volume_robust_z_sector_rank",
    "trade_size_robust_z_sector_rank",
    "clv_sector_rank",
    "garman_klass_var_1d_sector_rank",
    "log_return_1d_sector_rank",
    "ou_log_close_stretch_63",
    "ou_garman_klass_var_1d_log1p_stretch_63",
    "ou_volume_robust_z_stretch_63",
    "ou_trade_size_robust_z_stretch_63",
]
for f in feats:
    if f in df.columns:
        print(f"{f}: EXISTS  non-null={df[f].notna().sum():,}")
    else:
        print(f"{f}: MISSING")
