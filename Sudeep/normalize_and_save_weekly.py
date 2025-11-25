#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import numpy as np
import os

IN = "merged_weekly_by_isoweek.csv"   # or your merged filename
OUT_MERGED = "merged_weekly_canonical.csv"
OUT_IMD = "imd_weekly_canonical.csv"
OUT_NASA = "nasa_weekly_canonical.csv"

df = pd.read_csv(IN, parse_dates=["week_start_imd","week_end_sunday","week_start_nasa"], dayfirst=False)

# normalize district
df['district'] = df['district'].astype(str).str.strip()

# create canonical week_start:
# Prefer week_start_nasa (Monday). If missing, use week_end_sunday + 1 day (Sunday->Monday)
df['week_start_nasa'] = pd.to_datetime(df['week_start_nasa'], errors='coerce')
df['week_end_sunday'] = pd.to_datetime(df['week_end_sunday'], errors='coerce')
df['week_start_imd'] = pd.to_datetime(df['week_start_imd'], errors='coerce')

df['week_start'] = df['week_start_nasa'].copy()
missing_mask = df['week_start'].isna()
df.loc[missing_mask, 'week_start'] = df.loc[missing_mask, 'week_end_sunday'] + pd.Timedelta(days=1)
# final fallback: use week_start_imd
mask2 = df['week_start'].isna()
df.loc[mask2, 'week_start'] = df.loc[mask2, 'week_start_imd']

# ensure week_start is datetime
df['week_start'] = pd.to_datetime(df['week_start'])

# reorder columns: week_start, district, imd_rain_mm, nasa_prectotcorr, other features...
cols = df.columns.tolist()
keep_first = ['week_start','district','imd_rain_mm','nasa_prectotcorr','T2M_MAX','T2M_MIN','T2M','RH2M']
# keep only those that exist
keep_first = [c for c in keep_first if c in df.columns]
others = [c for c in cols if c not in keep_first and c not in ['week_start','district']]
final_cols = ['week_start','district'] + keep_first[2:] + others  # ensures week_start,district first
final_cols = [c for c in final_cols if c in df.columns]  # filter
df = df[final_cols]

# sanity check: how many rows
print("Rows:", len(df))
print("Unique districts:", df['district'].nunique())

# Save merged canonical
df.to_csv(OUT_MERGED, index=False)
print("Saved merged canonical ->", OUT_MERGED)

# Also produce separate IMD and NASA CSVs (weekly) for compatibility with prepare script
imd_df = df[['week_start','district','imd_rain_mm']].rename(columns={'imd_rain_mm':'rain_sum'})
nasa_cols = [c for c in ['week_start','district','nasa_prectotcorr','T2M_MAX','T2M_MIN','T2M','RH2M'] if c in df.columns]
nasa_df = df[nasa_cols].copy()
# rename nasa precip column to PRECTOTCORR to match earlier scripts if desired
if 'nasa_prectotcorr' in nasa_df.columns:
    nasa_df = nasa_df.rename(columns={'nasa_prectotcorr':'PRECTOTCORR'})

imd_df.to_csv(OUT_IMD, index=False)
nasa_df.to_csv(OUT_NASA, index=False)
print("Saved IMD ->", OUT_IMD, "and NASA ->", OUT_NASA)

# Quick checks: precip summary and ratio
if 'imd_rain_mm' in df.columns and 'nasa_prectotcorr' in df.columns:
    df['precip_ratio'] = df['nasa_prectotcorr'] / (df['imd_rain_mm'].replace(0, np.nan))
    print("IMD precip stats:", df['imd_rain_mm'].describe().to_dict())
    print("NASA precip stats:", df['nasa_prectotcorr'].describe().to_dict())
    print("Precipt ratio (median, mean):", df['precip_ratio'].median(), df['precip_ratio'].mean())

# Save a small sample to inspect
df.head(40).to_csv("merged_weekly_canonical_sample.csv", index=False)
print("Saved sample -> merged_weekly_canonical_sample.csv")
