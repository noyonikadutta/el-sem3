# FarmerML/scripts/make_features_and_labels.py
import os
import numpy as np
import pandas as pd

# ---------- CONFIG ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_CSV = os.path.join(PROJECT_ROOT, "output", "merged_weekly_canonical.csv")
OUT_FEATURES = os.path.join(PROJECT_ROOT, "output", "merged_features.csv")
OUT_TRAIN = os.path.join(PROJECT_ROOT, "output", "train.csv")
OUT_TEST = os.path.join(PROJECT_ROOT, "output", "test.csv")

# Train/test split by year (change as needed)
TRAIN_MAX_YEAR = 2022  # train <= 2022, test 2023-2024
# Flood threshold absolute mm (avoid tiny spikes being flagged as flood)
FLOOD_MIN_MM = 75

# ---------- LOAD ----------
print("Loading:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV, parse_dates=["week_start"], dayfirst=False)
df = df.sort_values(["district", "week_start"]).reset_index(drop=True)

# Safety check columns expected (adjust names if needed)
expected = ["district", "week_start", "imd_rain_mm", "nasa_prectotcorr", "T2M", "T2M_MAX", "T2M_MIN", "RH2M"]
missing = [c for c in expected if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns in input CSV: {missing}")

# ---------- BASIC TIME FIELDS ----------
df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
df["iso_year"] = df["week_start"].dt.isocalendar().year.astype(int)

# ---------- ROLLING FEATURES per district ----------
def make_roll_features(g):
    g = g.sort_values("week_start").copy()
    # 1-week values (already weekly aggregates)
    g["rain_1w"] = g["imd_rain_mm"].astype(float)

    # rolling sums for rainfall
    g["rain_4w"] = g["imd_rain_mm"].rolling(window=4, min_periods=1).sum()
    g["rain_12w"] = g["imd_rain_mm"].rolling(window=12, min_periods=1).sum()

    # rolling means for temperature and humidity
    g["t2m_4w"] = g["T2M"].rolling(window=4, min_periods=1).mean()
    g["t2mmax_4w"] = g["T2M_MAX"].rolling(window=4, min_periods=1).mean()
    g["rh_4w"] = g["RH2M"].rolling(window=4, min_periods=1).mean()

    # simple lag features (previous week)
    g["rain_1w_lag1"] = g["rain_1w"].shift(1)
    g["rain_4w_lag1"] = g["rain_4w"].shift(1)
    g["t2m_4w_lag1"] = g["t2m_4w"].shift(1)

    # consistency feature: IMD / NASA ratio for the week (add tiny eps)
    g["rain_imd_vs_nasa_ratio"] = g["imd_rain_mm"] / (g["nasa_prectotcorr"].replace(0, np.nan) + 1e-6)
    g["rain_imd_vs_nasa_ratio"] = g["rain_imd_vs_nasa_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return g

df = df.groupby("district", group_keys=False).apply(make_roll_features).reset_index(drop=True)

# ---------- CLIMATOLOGY & PERCENTILES (district x week_of_year) ----------
# We'll compute mean/std of the 12-week rolling rainfall and percentiles for weekly rainfall & Tmax
clim = df.groupby(["district", "week_of_year"]).agg(
    rain12w_clim_mean=("rain_12w", "mean"),
    rain12w_clim_std=("rain_12w", "std"),
    rain1w_p95=("rain_1w", lambda x: np.nanpercentile(x.dropna(), 95) if x.notna().any() else np.nan),
    t2mmax_p90=("T2M_MAX", lambda x: np.nanpercentile(x.dropna(), 90) if x.notna().any() else np.nan),
    rain1w_p50=("rain_1w", lambda x: np.nanpercentile(x.dropna(), 50) if x.notna().any() else np.nan),
).reset_index()

df = df.merge(clim, on=["district", "week_of_year"], how="left")

# ---------- LABEL RULES (simple, interpretable) ----------
# Drought: rain_12w much below climatology
# - require climatology to be non-zero / not NaN
df["rain12w_anom_ratio"] = df["rain_12w"] / (df["rain12w_clim_mean"].replace(0, np.nan))
df["drought_label"] = ((df["rain12w_anom_ratio"] < 0.6) & (df["rain12w_clim_mean"].notna())).astype(int)

# Flood: extreme weekly rainfall (>= 95th percentile for the district-week) AND above absolute threshold
df["flood_label"] = ((df["rain_1w"] >= df["rain1w_p95"]) & (df["rain_1w"] >= FLOOD_MIN_MM)).astype(int)

# Heatwave: weekly max temp exceeds local 90th percentile + buffer (3 deg C)
df["heatwave_label"] = ((df["T2M_MAX"] >= (df["t2mmax_p90"] + 3.0))).astype(int)

# If percentiles are NaN (insufficient history), set label to 0 (conservative)
df.loc[df["rain1w_p95"].isna(), "flood_label"] = 0
df.loc[df["t2mmax_p90"].isna(), "heatwave_label"] = 0

# ---------- SMALL CLEANUPS ----------
# Fill remaining NaNs in rolling features with sensible defaults
fill_cols = ["rain_4w", "rain_12w", "t2m_4w", "t2mmax_4w", "rh_4w", "rain_1w_lag1"]
for c in fill_cols:
    if c in df.columns:
        df[c] = df[c].fillna(0.0)

# Keep a compact set of columns for ML
feature_cols = [
    "district", "week_start", "iso_year", "week_of_year",
    # raw + rolling
    "imd_rain_mm", "rain_1w", "rain_4w", "rain_12w", "rain_1w_lag1", "rain_4w_lag1",
    "t2m_4w", "t2mmax_4w", "t2m_4w_lag1", "rh_4w",
    # consistency / climatology
    "rain_imd_vs_nasa_ratio", "rain12w_anom_ratio",
    # labels
    "drought_label", "flood_label", "heatwave_label"
]

# Drop duplicates and keep necessary
result = df[feature_cols].copy()
result = result.sort_values(["district", "week_start"]).reset_index(drop=True)

# ---------- SAVE features CSV ----------
print("Writing features to:", OUT_FEATURES)
os.makedirs(os.path.dirname(OUT_FEATURES), exist_ok=True)
result.to_csv(OUT_FEATURES, index=False)

# ---------- TRAIN/TEST SPLIT (time-based) ----------
train = result[result["iso_year"] <= TRAIN_MAX_YEAR].copy()
test = result[result["iso_year"] > TRAIN_MAX_YEAR].copy()

print(f"Train rows: {len(train)}  |  Test rows: {len(test)}")

print("Writing train/test:")
train.to_csv(OUT_TRAIN, index=False)
test.to_csv(OUT_TEST, index=False)

print("Done.")
