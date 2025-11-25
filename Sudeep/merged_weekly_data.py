#!/usr/bin/env python3
"""
merge_weekly_check.py

Normalizes IMD weekly and NASA weekly CSVs to (district, iso_year, iso_week),
compares precipitation magnitudes, and writes merged CSV.

Adjust IMD_CSV and NASA_CSV paths as needed.
"""
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
IMD_CSV = "data/imd_rainfall_weekly_district.csv"       # has columns: district_name, date, imd_weekly_rainfall_mm
NASA_CSV = "karnataka_nasa_power_weekly.csv"            # has columns: week_start, district, PRECTOTCORR, T2M_MAX, ...
OUT_MERGED = "data/merged_weekly_by_isoweek.csv"
# --------------------------------------------------------

def load_imd(path):
    df = pd.read_csv(path, parse_dates=["date"])
    # canonicalize names
    df = df.rename(columns={"district_name":"district", "date":"week_end_sunday", "imd_weekly_rainfall_mm":"imd_rain_mm"})
    # Create ISO week/year and choose canonical week_start (Monday) as (week_end_sunday + 1 day)
    df["iso_year"] = df["week_end_sunday"].dt.isocalendar().year
    df["iso_week"] = df["week_end_sunday"].dt.isocalendar().week
    # Optional: compute week_start (Monday) from iso_year/iso_week
    df["week_start"] = pd.to_datetime(df["week_end_sunday"] + pd.Timedelta(days=1))  # Sunday->next Monday
    # Trim spaces in district
    df["district"] = df["district"].astype(str).str.strip()
    return df[["district","week_start","week_end_sunday","iso_year","iso_week","imd_rain_mm"]]

def load_nasa(path):
    df = pd.read_csv(path, parse_dates=["week_start"])
    # canonicalize column names if necessary
    # Ensure district col is named 'district'
    if "district" not in df.columns:
        # attempt common alternatives
        for c in df.columns:
            if c.lower().startswith("dist"):
                df = df.rename(columns={c:"district"})
                break
    df["district"] = df["district"].astype(str).str.strip()
    # create iso week/year from week_start (which looks like Monday)
    df["iso_year"] = df["week_start"].dt.isocalendar().year
    df["iso_week"] = df["week_start"].dt.isocalendar().week
    # Rename PRECTOTCORR to nasa_precip (keep original too)
    if "PRECTOTCORR" in df.columns:
        df = df.rename(columns={"PRECTOTCORR":"nasa_prectotcorr"})
    return df

def main():
    imd = load_imd(IMD_CSV)
    nasa = load_nasa(NASA_CSV)
    print("IMD sample:")
    print(imd.head())
    print("\nNASA sample:")
    print(nasa.head())

    # Merge on district + iso_year + iso_week (inner join)
    merged = pd.merge(imd, nasa, on=["district","iso_year","iso_week"], how="inner", suffixes=("_imd","_nasa"))
    print(f"\nMerged rows (inner): {len(merged)}")

    # Diagnostics: how many IMD rows found matches?
    imd_total = len(imd)
    matched_imd = merged[["district","iso_year","iso_week"]].drop_duplicates().shape[0]
    print(f"IMD rows: {imd_total}, matched by NASA: {matched_imd}")

    # Quick precip comparison if both columns exist
    if "imd_rain_mm" in merged.columns and "nasa_prectotcorr" in merged.columns:
        # Some NASA values may be very small; check stats
        print("\nPrecip stats (IMD imd_rain_mm):")
        print(merged["imd_rain_mm"].describe())
        print("\nPrecip stats (NASA nasa_prectotcorr):")
        print(merged["nasa_prectotcorr"].describe())

        # Compute correlation per district and overall
        overall_corr = merged[["imd_rain_mm","nasa_prectotcorr"]].corr().iloc[0,1]
        print(f"\nOverall Pearson correlation (IMD vs NASA precip): {overall_corr:.3f}")

        # Simple scatter for a sample district (first one)
        sample_district = merged["district"].unique()[0]
        sample = merged[merged["district"]==sample_district]
        print(f"\nSample district: {sample_district}, rows: {len(sample)}")
        if not sample.empty:
            try:
                # after creating the scatter plot
                plt.scatter(sample["imd_rain_mm"], sample["nasa_prectotcorr"], s=10)
                plt.xlabel("IMD weekly rain (mm)")
                plt.ylabel("NASA PRECTOTCORR (mm/week?)")
                plt.title("IMD vs NASA precip — Bagalkot")
                plt.grid(True)

                # Save instead of show (non-blocking)
                out_plot = "plots/imd_vs_nasa_Bagalkot.png"
                import os
                os.makedirs("plots", exist_ok=True)
                plt.savefig(out_plot, dpi=200, bbox_inches="tight")
                plt.close()
                print(f"Saved plot to {out_plot}")

            except Exception as e:
                print("Plot failed:", e)

    # Save merged
    merged.to_csv(OUT_MERGED, index=False)
    print(f"\nSaved merged file: {OUT_MERGED}")

if __name__ == "__main__":
    main()
