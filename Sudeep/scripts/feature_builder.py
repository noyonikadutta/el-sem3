# FarmerML/scripts/feature_builder.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_CSV = os.path.join(PROJECT_ROOT, "output", "merged_weekly_canonical.csv")

# Feature columns that the models expect (same as training)
FEATURE_COLS = [
    "imd_rain_mm", "rain_1w", "rain_4w", "rain_12w", "rain_1w_lag1", "rain_4w_lag1",
    "t2m_4w", "t2mmax_4w", "t2m_4w_lag1", "rh_4w",
    "rain_imd_vs_nasa_ratio", "rain12w_anom_ratio"
]

# Load canonical weekly CSV once
_df_cache = None
def _load_df():
    global _df_cache
    if _df_cache is None:
        df = pd.read_csv(INPUT_CSV, parse_dates=["week_start"])
        # required columns check
        required = ["district", "week_start", "imd_rain_mm", "nasa_prectotcorr", "T2M", "T2M_MAX", "T2M_MIN", "RH2M"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in merged_weekly_canonical.csv: {missing}")
        df = df.sort_values(["district", "week_start"]).reset_index(drop=True)
        _df_cache = df
    return _df_cache.copy()

def build_features_for(district: str, week_start: str) -> dict:
    """
    Build model features for a given district and week_start (ISO 'YYYY-MM-DD').
    Uses only data <= week_start (no future leakage).
    Returns a dict with keys matching FEATURE_COLS.
    """
    df = _load_df()
    # parse week_start
    if isinstance(week_start, str):
        wk = pd.to_datetime(week_start)
    else:
        wk = pd.to_datetime(week_start)
    # filter district history up to and including wk
    h = df[(df["district"] == district) & (df["week_start"] <= wk)].sort_values("week_start")
    if h.empty:
        raise ValueError(f"No historical rows found for district={district} up to {wk.date()}")
    # We compute rolling windows that end at wk (include that week if present)
    # if the exact week_start row is missing, we still compute features from latest prior weeks;
    # for imd_rain_mm we take the value for wk if present, else 0 (you can change this)
    last_row = h[h["week_start"] == wk]
    if last_row.shape[0] == 1:
        imd_rain_mm = float(last_row["imd_rain_mm"].iloc[0])
        nasa_prec = float(last_row["nasa_prectotcorr"].iloc[0])
        t2m_max = float(last_row["T2M_MAX"].iloc[0])
        t2m_mean = float(last_row["T2M"].iloc[0])
        rh = float(last_row["RH2M"].iloc[0])
    else:
        # if target week row is missing, use zeros / last-known for some features
        imd_rain_mm = 0.0
        nasa_prec = 0.0
        t2m_max = float(h["T2M_MAX"].iloc[-1])
        t2m_mean = float(h["T2M"].iloc[-1])
        rh = float(h["RH2M"].iloc[-1])

    # helper: compute rolling sum/mean ending at last available index
    # ensure we consider up to 12 previous rows (including current if present)
    def rolling_sum(col, window):
        vals = h[col].fillna(0.0).values
        if len(vals) == 0:
            return 0.0
        # take last 'window' values
        return float(vals[-window:].sum())

    def rolling_mean(col, window):
        vals = h[col].dropna().values
        if len(vals) == 0:
            return 0.0
        return float(vals[-window:].mean()) if len(vals) >= 1 else 0.0

    rain_1w = imd_rain_mm
    rain_4w = rolling_sum("imd_rain_mm", 4)
    rain_12w = rolling_sum("imd_rain_mm", 12)

    # lags: previous week values (if available)
    rain_1w_lag1 = float(h["imd_rain_mm"].iloc[-2]) if len(h) >= 2 else 0.0
    rain_4w_lag1 = None
    if len(h) >= 2:
        # compute rolling 4w ending at prior week -> last 4 excluding current (so take last 5 and drop last)
        rain_4w_lag1 = float(h["imd_rain_mm"].iloc[-5:-1].sum()) if len(h) >= 5 else float(h["imd_rain_mm"].iloc[:-1].sum())
    else:
        rain_4w_lag1 = 0.0

    t2m_4w = rolling_mean("T2M", 4)
    t2mmax_4w = rolling_mean("T2M_MAX", 4)
    t2m_4w_lag1 = None
    if len(h) >= 2:
        t2m_4w_lag1 = float(h["T2M"].iloc[-5:-1].mean()) if len(h) >= 5 else float(h["T2M"].iloc[:-1].mean())
    else:
        t2m_4w_lag1 = t2m_4w

    rh_4w = rolling_mean("RH2M", 4)

    # consistency ratio (imd / nasa) for the week (avoid division by zero)
    rain_imd_vs_nasa_ratio = imd_rain_mm / (nasa_prec + 1e-6) if nasa_prec != 0 else 0.0

    # climatology: compute mean of 12-week rolling rainfall for same week-of-year historically
    h["week_of_year"] = h["week_start"].dt.isocalendar().week.astype(int)
    wom = wk.isocalendar()[1]
    clim_group = df[(df["district"] == district)].copy()
    clim_group["week_of_year"] = clim_group["week_start"].dt.isocalendar().week.astype(int)
    # use the same week_of_year entries across years to estimate typical 12-week sums
    clim_vals = []
    for _, row in clim_group[clim_group["week_of_year"] == wom].iterrows():
        # compute 12-week sum centered at that row (or ending at that row) if possible
        # we'll compute the 12-week sum ending at that row
        # find the index of row in clim_group and sum previous 12
        idxs = clim_group.index[clim_group["week_start"] == row["week_start"]].tolist()
        if not idxs:
            continue
        i = idxs[0]
        start_i = max(0, i-11)
        s = clim_group.iloc[start_i:i+1]["imd_rain_mm"].sum()
        clim_vals.append(s)
    rain12w_clim_mean = float(np.nanmean(clim_vals)) if len(clim_vals) > 0 else np.nan
    rain12w_anom_ratio = (rain_12w / rain12w_clim_mean) if (not np.isnan(rain12w_clim_mean) and rain12w_clim_mean != 0) else 0.0

    feat = {
        "imd_rain_mm": float(imd_rain_mm),
        "rain_1w": float(rain_1w),
        "rain_4w": float(rain_4w),
        "rain_12w": float(rain_12w),
        "rain_1w_lag1": float(rain_1w_lag1),
        "rain_4w_lag1": float(rain_4w_lag1),
        "t2m_4w": float(t2m_4w),
        "t2mmax_4w": float(t2mmax_4w),
        "t2m_4w_lag1": float(t2m_4w_lag1),
        "rh_4w": float(rh_4w),
        "rain_imd_vs_nasa_ratio": float(rain_imd_vs_nasa_ratio),
        "rain12w_anom_ratio": float(rain12w_anom_ratio)
    }

    # ensure all FEATURE_COLS present and ordered
    ordered = {k: feat.get(k, 0.0) for k in FEATURE_COLS}
    return ordered

# quick test
if __name__ == "__main__":
    print("Testing builder with sample district/week...")
    print(list(build_features_for("Bagalkot", "2015-01-05").items())[:12])
