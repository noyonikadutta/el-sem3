#!/usr/bin/env python3
"""
run_full_pipeline.py

End-to-end pipeline:
1. Collect IMD daily rainfall into data/imd_rainfall_daily_district.csv
2. Aggregate IMD daily -> weekly --> data/imd_rainfall_weekly_district.csv
3. Fetch NASA POWER daily for centroids CSV -> aggregate weekly -> karnataka_nasa_power_weekly.csv
4. Normalize & merge IMD+NASA weekly -> merged_weekly_canonical.csv
5. Create features & labels -> merged_features.csv, train.csv, test.csv

Edit configuration below as needed.
"""
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import time
from typing import List

# ---------- CONFIG ----------
# Date range for full historical collection
START_DATE = pd.Timestamp("2015-01-01")
END_DATE   = pd.Timestamp("2024-12-31")

# IMD config (update shapefile path)
DISTRICT_SHP_PATH = r"C:\Users\SUDEEP XAVIER ROCHE\Downloads\District\District.shp"
DISTRICT_NAME_COLUMN = 'KGISDist_1'

# Centroids CSV (must contain district_name, latitude, longitude)
CENTROIDS_CSV = r"data\district_centroids_wgs84_lookup.csv"

# Output canonical files (used later)
OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMD_DAILY_OUT = OUT_DIR / "imd_rainfall_daily_district.csv"
IMD_WEEKLY_OUT = OUT_DIR / "imd_rainfall_weekly_district.csv"
NASA_WEEKLY_OUT = OUT_DIR / "karnataka_nasa_power_weekly.csv"
MERGED_WEEKLY_CANONICAL = OUT_DIR / "merged_weekly_canonical.csv"

# Final training-ready outputs
MERGED_FEATURES = OUT_DIR / "merged_features.csv"
TRAIN_CSV = OUT_DIR / "train.csv"
TEST_CSV = OUT_DIR / "test.csv"

# Prepare script settings (lags, rolling windows, split date)
SPLIT_DATE = "2023-01-01"
ROLL_WINDOWS = [3, 6]
LAGS = [1, 2]
MIN_ROWS_PER_DISTRICT = 20

# NASA parameters (you can extend)
NASA_PARAMS = [
    "T2M_MAX", "T2M_MIN", "T2M", "RH2M", "PRECTOTCORR"
]

# NASA polite settings (per request script; slow is OK)
SLEEP_BETWEEN = 1.0
MAX_RETRIES = 3
RETRY_SLEEP = 5

# --------------------------------------------------------

# ---------- IMD collection helpers (adapted from your IMD.py) ----------

def collect_imd_rainfall_wrapper(shp_path, name_col, start_year, end_year, output_file_daily):
    """
    Robust IMD daily collection (year-by-year) with integrated grid->district mapping
    and daily aggregation to district mean rainfall. Returns final daily DataFrame.
    """
    import geopandas as gpd
    import numpy as np
    import imdlib as imd
    import inspect
    from shapely.geometry import Point
    import pandas as pd
    import time
    import os

    COORD_DECIMALS = 6
    YEAR_MAX_RETRIES = 4
    YEAR_RETRY_DELAY = 10  # seconds initial backoff

    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    # read districts shapefile once
    districts_gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
    districts_gdf = districts_gdf.rename(columns={name_col: 'district_name'})
    districts_gdf['district_name'] = districts_gdf['district_name'].astype(str).str.strip()

    # flexible imd.get_data caller (tries keyword, then positional)
    def _try_imd_get_data(varname, start_yr, end_yr, fn_format='yearwise'):
        sig = inspect.signature(imd.get_data)
        param_names = list(sig.parameters.keys())
        kwargs = {}
        var_candidates = ['param', 'parameter', 'variable', 'var', 'product', 'data', 'param_name', 'varname', 'var_type', 'vname']
        start_candidates = ['start_yr', 'start_year', 'start', 'sy', 'startYear']
        end_candidates   = ['end_yr', 'end_year', 'end', 'ey', 'endYear']
        fn_candidates    = ['fn_format','format','file_format','fmt']
        for c in var_candidates:
            if c in param_names:
                kwargs[c] = varname
                break
        for c in start_candidates:
            if c in param_names:
                kwargs[c] = start_yr
                break
        for c in end_candidates:
            if c in param_names:
                kwargs[c] = end_yr
                break
        for c in fn_candidates:
            if c in param_names:
                kwargs[c] = fn_format
                break
        try:
            return imd.get_data(**kwargs)
        except Exception:
            positional_attempts = [
                (varname, start_yr, end_yr, fn_format),
                (varname, start_yr, end_yr),
                (start_yr, end_yr, varname),
                (start_yr, end_yr),
                (start_yr, end_yr, fn_format),
            ]
            for args in positional_attempts:
                try:
                    return imd.get_data(*args)
                except TypeError:
                    continue
            raise

    # helpers to extract lat/lon/dates/arrays from imd object (robust)
    def _extract_lats(data):
        if hasattr(data, 'lat_array'):
            return np.array(getattr(data, 'lat_array'))
        for name in ('get_lats', 'get_latitudes', 'get_lat'):
            if hasattr(data, name) and callable(getattr(data, name)):
                return np.array(getattr(data, name)())
        if hasattr(data, 'get_xarray') and callable(getattr(data, 'get_xarray')):
            ds = data.get_xarray()
            for coord in ('lat','latitude','lats'):
                if coord in ds.coords:
                    return np.array(ds.coords[coord].values)
        if hasattr(data, 'ds'):
            ds = getattr(data, 'ds')
            for coord in ds.coords:
                if 'lat' in str(coord).lower():
                    return np.array(ds.coords[coord].values)
        raise RuntimeError("Couldn't extract lats from IMD data")

    def _extract_lons(data):
        if hasattr(data, 'lon_array'):
            return np.array(getattr(data, 'lon_array'))
        for name in ('get_lons', 'get_longitudes', 'get_lon'):
            if hasattr(data, name) and callable(getattr(data, name)):
                return np.array(getattr(data, name)())
        if hasattr(data, 'get_xarray') and callable(getattr(data, 'get_xarray')):
            ds = data.get_xarray()
            for coord in ('lon','longitude','lons'):
                if coord in ds.coords:
                    return np.array(ds.coords[coord].values)
        if hasattr(data, 'ds'):
            ds = getattr(data, 'ds')
            for coord in ds.coords:
                if 'lon' in str(coord).lower():
                    return np.array(ds.coords[coord].values)
        raise RuntimeError("Couldn't extract lons from IMD data")

    def _extract_dates(data):
        if hasattr(data, 'start_day') and hasattr(data, 'no_days'):
            start = getattr(data, 'start_day')
            n = getattr(data, 'no_days')
            return list(pd.date_range(start=pd.to_datetime(start), periods=int(n), freq='D'))
        if hasattr(data, 'end_day') and hasattr(data, 'no_days'):
            end = getattr(data, 'end_day')
            n = getattr(data, 'no_days')
            return list(pd.date_range(end=pd.to_datetime(end), periods=int(n), freq='D'))
        if hasattr(data, 'get_xarray') and callable(getattr(data, 'get_xarray')):
            ds = data.get_xarray()
            for d in ('time','date','dates'):
                if d in ds.coords:
                    return list(pd.to_datetime(ds.coords[d].values))
        if hasattr(data, 'dates'):
            return list(pd.to_datetime(getattr(data,'dates')))
        raise RuntimeError("Couldn't extract dates from IMD data")

    def _get_data_array_for_day(data, day):
        for name in ('get_data_array', 'get_array', 'get_daily_array', 'get_data_for_date', 'get_daily'):
            if hasattr(data, name) and callable(getattr(data, name)):
                return np.array(getattr(data, name)(day))
        if hasattr(data, 'get_xarray') and callable(getattr(data, 'get_xarray')):
            ds = data.get_xarray()
            varnames = list(ds.data_vars)
            if not varnames:
                raise RuntimeError("xarray dataset has no data vars")
            var = varnames[0]
            if 'time' in ds.coords:
                times = pd.to_datetime(ds.coords['time'].values)
                target = pd.to_datetime(day)
                idx = (times == target).nonzero()[0]
                if len(idx) > 0:
                    return np.array(ds[var].isel(time=int(idx[0])).values)
        if hasattr(data, 'data'):
            arr = getattr(data, 'data')
            arr = np.array(arr)
            if arr.ndim == 3:
                if hasattr(data, 'start_day') and hasattr(data, 'no_days'):
                    start_ts = pd.to_datetime(getattr(data, 'start_day'))
                    times = pd.date_range(start=start_ts, periods=int(getattr(data,'no_days')), freq='D')
                    idxs = (times == pd.to_datetime(day)).nonzero()[0]
                    if len(idxs) > 0:
                        return arr[int(idxs[0]), :, :]
            return arr
        raise RuntimeError(f"No method to extract daily array for {day}")

    all_district_rainfall = []
    grid_to_district = None
    base_grid_hash = None

    successful_years = []
    failed_years = []

    for year in range(start_year, end_year + 1):
        attempt = 0
        year_success = False
        while attempt < YEAR_MAX_RETRIES and not year_success:
            attempt += 1
            try:
                print(f"[IMD] Downloading: rain for year {year}")
                data = _try_imd_get_data('rain', year, year, fn_format='yearwise')
                # extract grid metadata & dates
                lats = _extract_lats(data)
                lons = _extract_lons(data)
                dates = _extract_dates(data)
                lats = np.round(np.array(lats).astype(float), COORD_DECIMALS)
                lons = np.round(np.array(lons).astype(float), COORD_DECIMALS)

                # Build grid DataFrame and mapping *once* (or if grid changed)
                grid_hash = (tuple(lats.tolist()), tuple(lons.tolist()))
                if (grid_to_district is None) or (grid_hash != base_grid_hash):
                    print(f"[IMD] Building grid -> district mapping for year {year} (this may take a moment)...")
                    grid_points = []
                    for lat in lats:
                        for lon in lons:
                            grid_points.append({
                                'latitude': float(lat),
                                'longitude': float(lon),
                                'geometry': Point(float(lon), float(lat))
                            })
                    grid_gdf = gpd.GeoDataFrame(grid_points, geometry='geometry', crs="EPSG:4326")
                    # spatial join
                    grid_to_district_gdf = gpd.sjoin(grid_gdf, districts_gdf[['district_name','geometry']], how='inner', predicate='within')
                    # reduce to lat/lon -> district mapping table
                    grid_to_district = grid_to_district_gdf[['district_name','latitude','longitude']].drop_duplicates().reset_index(drop=True)
                    grid_to_district['latitude'] = np.round(grid_to_district['latitude'].astype(float), COORD_DECIMALS)
                    grid_to_district['longitude'] = np.round(grid_to_district['longitude'].astype(float), COORD_DECIMALS)
                    base_grid_hash = grid_hash
                    print(f"[IMD] grid->district mapping created: {len(grid_to_district)} grid points mapped.")

                # iterate daily and aggregate for this year
                for day in dates:
                    try:
                        arr = _get_data_array_for_day(data, day)
                    except Exception as e:
                        print(f"[IMD] year {year} skip day {day}: {e}")
                        continue
                    arr = np.array(arr)
                    # fix orientation if necessary
                    if arr.shape != (len(lats), len(lons)):
                        if arr.T.shape == (len(lats), len(lons)):
                            arr = arr.T
                        else:
                            print(f"[IMD] year {year} day {day} shape mismatch {arr.shape} -> skipping")
                            continue
                    daily_df = pd.DataFrame({
                        'latitude': np.repeat(lats, len(lons)),
                        'longitude': np.tile(lons, len(lats)),
                        'rainfall': arr.flatten()
                    })
                    daily_df['latitude'] = np.round(daily_df['latitude'].astype(float), COORD_DECIMALS)
                    daily_df['longitude'] = np.round(daily_df['longitude'].astype(float), COORD_DECIMALS)
                    # merge with mapping table
                    daily_district = pd.merge(grid_to_district, daily_df, on=['latitude','longitude'], how='inner')
                    if daily_district.empty:
                        # no mapping for this day -> skip
                        continue
                    daily_avg = daily_district.groupby('district_name')['rainfall'].mean().reset_index()
                    daily_avg['date'] = pd.to_datetime(day)
                    all_district_rainfall.append(daily_avg)

                successful_years.append(year)
                year_success = True
                print(f"[IMD] year {year} fetched successfully.")

            except Exception as e:
                print(f"[IMD] year={year} attempt {attempt} failed: {e}")
                if attempt < YEAR_MAX_RETRIES:
                    wait = YEAR_RETRY_DELAY * (2 ** (attempt - 1))
                    print(f"[IMD] retrying year {year} after {wait}s ...")
                    time.sleep(wait)
                else:
                    print(f"[IMD] year {year} failed after {YEAR_MAX_RETRIES} attempts — skipping year.")
                    failed_years.append(year)
                    break

    if len(all_district_rainfall) == 0:
        raise RuntimeError("[IMD] No daily data aggregated across all years.")

    final_df = pd.concat(all_district_rainfall, ignore_index=True)
    final_df = final_df.rename(columns={'rainfall':'imd_rainfall_mm','district_name':'district_name'})
    # Save
    out_dir = os.path.dirname(output_file_daily)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    final_df.to_csv(output_file_daily, index=False)
    print(f"[IMD] saved daily rainfall -> {output_file_daily}")
    print(f"[IMD] successful years: {successful_years} | failed years: {failed_years}")
    return final_df


def aggregate_to_weekly_wrapper(daily_df, output_weekly):
    if daily_df.empty:
        raise RuntimeError("[IMD] daily_df is empty")
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    weekly_df = daily_df.set_index('date').groupby('district_name').resample('W')['imd_rainfall_mm'].sum().reset_index()
    weekly_df = weekly_df.rename(columns={'date':'week_end_sunday', 'imd_rainfall_mm':'imd_weekly_rainfall_mm'})
    weekly_df.to_csv(output_weekly, index=False)
    print(f"[IMD] saved weekly rainfall -> {output_weekly}")
    return weekly_df

# ---------- NASA collection helpers (adapted from your NasaApi.py) ----------
import requests
from tqdm import tqdm

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

def build_request_url(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp, parameters: List[str]) -> str:
    params = {
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "latitude": str(lat),
        "longitude": str(lon),
        "parameters": ",".join(parameters),
        "community": "AG",
        "format": "JSON"
    }
    query = "&".join(f"{k}={v}" for k,v in params.items())
    return f"{BASE_URL}?{query}"

def fetch_power_json(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp, parameters: List[str]) -> dict:
    url = build_request_url(lat, lon, start, end, parameters)
    last_exc = None
    for attempt in range(1, MAX_RETRIES+1):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return r.json()
            else:
                last_exc = Exception(f"Bad status {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_exc = e
        time.sleep(RETRY_SLEEP * attempt)
    raise last_exc

def json_to_daily_df(data_json: dict) -> pd.DataFrame:
    props = data_json.get("properties", {})
    params = props.get("parameter", {})
    if not params:
        raise RuntimeError("No properties.parameter in NASA JSON")
    df = pd.DataFrame(params)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.sort_index()
    return df

def aggregate_daily_to_weekly(daily_df: pd.DataFrame, week_rule="W-MON") -> pd.DataFrame:
    if daily_df.empty:
        return daily_df
    df = daily_df.astype(float).copy()
    precip_cols = [c for c in df.columns if "PREC" in c.upper() or "PRECTOT" in c.upper()]
    other_cols = [c for c in df.columns if c not in precip_cols]
    agg = {}
    for c in precip_cols:
        agg[c] = 'sum'
    for c in other_cols:
        agg[c] = 'mean'
    weekly = df.resample(week_rule).agg(agg).reset_index().rename(columns={'index':'week_start'})
    # ensure week_start column
    if 'week_start' not in weekly.columns:
        weekly = weekly.rename(columns={weekly.columns[0]:'week_start'})
    return weekly

def fetch_nasa_for_centroids(centroids_csv, start, end, parameters, out_weekly_csv):
    cent = pd.read_csv(centroids_csv)
    # expected columns: district_name or district / latitude / longitude
    # normalize names
    if 'district_name' in cent.columns:
        district_col = 'district_name'
    elif 'district' in cent.columns:
        district_col = 'district'
    else:
        # find likely district column
        district_col = [c for c in cent.columns if 'dist' in c.lower()][0]
    lat_col = [c for c in cent.columns if 'lat' in c.lower()][0]
    lon_col = [c for c in cent.columns if 'lon' in c.lower() or 'long' in c.lower()][0]
    cent[district_col] = cent[district_col].astype(str).str.strip()
    weekly_accum = []
    for _, row in tqdm(cent.iterrows(), total=len(cent), desc="NASA districts"):
        district = str(row[district_col]).strip()
        lat = float(row[lat_col]); lon = float(row[lon_col])
        # fetch
        try:
            j = fetch_power_json(lat=lat, lon=lon, start=start, end=end, parameters=parameters)
            daily = json_to_daily_df(j)
            # save raw daily if desired (skipped here)
            # aggregate weekly
            weekly = aggregate_daily_to_weekly(daily)
            weekly['district'] = district
            weekly_accum.append(weekly)
        except Exception as e:
            print(f"[NASA] skip {district}: {e}")
        time.sleep(SLEEP_BETWEEN)
    if not weekly_accum:
        raise RuntimeError("[NASA] no weekly data fetched")
    combined = pd.concat(weekly_accum, ignore_index=True)
    # ensure week_start column name and dtypes
    if 'week_start' in combined.columns:
        combined['week_start'] = pd.to_datetime(combined['week_start'])
    combined.to_csv(out_weekly_csv, index=False)
    print(f"[NASA] saved weekly -> {out_weekly_csv}")
    return combined

# ---------- Normalization & merge ----------
def normalize_and_merge(imd_weekly_csv, nasa_weekly_csv, out_merged_canonical):
    imd = pd.read_csv(imd_weekly_csv, parse_dates=['week_end_sunday'])
    imd = imd.rename(columns={'district_name':'district','imd_weekly_rainfall_mm':'imd_rain_mm'})
    # compute iso week/year and canonical week_start (Monday)
    imd['iso_year'] = imd['week_end_sunday'].dt.isocalendar().year
    imd['iso_week'] = imd['week_end_sunday'].dt.isocalendar().week
    imd['week_start_imd'] = (imd['week_end_sunday'] + pd.Timedelta(days=1)).dt.floor('D')

    nasa = pd.read_csv(nasa_weekly_csv, parse_dates=['week_start'])
    if 'district' not in nasa.columns:
        guess = [c for c in nasa.columns if 'dist' in c.lower()]
        if guess:
            nasa = nasa.rename(columns={guess[0]:'district'})
    nasa = nasa.rename(columns={'PRECTOTCORR':'nasa_prectotcorr'} if 'PRECTOTCORR' in nasa.columns else {})
    nasa['iso_year'] = nasa['week_start'].dt.isocalendar().year
    nasa['iso_week'] = nasa['week_start'].dt.isocalendar().week
    # merge on district + iso_year + iso_week
    merged = pd.merge(imd, nasa, on=['district','iso_year','iso_week'], how='inner', suffixes=('_imd','_nasa'))
    # create canonical week_start: prefer nasa.week_start, else imd week_start_imd
    if 'week_start' in merged.columns:
        merged['week_start'] = pd.to_datetime(merged['week_start'])
    else:
        merged['week_start'] = merged['week_start_imd']
    # reorder & save
    keep_cols = ['district','week_start','week_end_sunday','iso_year','iso_week','imd_rain_mm']
    keep_cols += [c for c in merged.columns if c not in keep_cols]
    merged = merged[keep_cols]
    merged.to_csv(out_merged_canonical, index=False)
    print(f"[MERGE] saved canonical merged -> {out_merged_canonical}")
    return merged

# ---------- Feature engineering & labeling (adapted from prepare_training_data_weekly.py) ----------
def prepare_features_from_merged(merged_canonical_csv, out_merged_features, out_train, out_test,
                                 split_date=SPLIT_DATE, roll_windows=ROLL_WINDOWS, lags=LAGS, min_rows=MIN_ROWS_PER_DISTRICT):
    df = pd.read_csv(merged_canonical_csv, parse_dates=['week_start'])
    # unify district
    df['district'] = df['district'].astype(str).str.strip()
    # rename nasa precipitation column if present
    if 'nasa_prectotcorr' in df.columns:
        pass
    # canonicalize NASA temp columns to known names if present
    col_map = {}
    for c in df.columns:
        if c.upper().startswith('T2M_MAX'):
            col_map[c] = 'T2M_MAX'
        if c.upper().startswith('T2M_MIN'):
            col_map[c] = 'T2M_MIN'
        if c.upper() == 'T2M':
            col_map[c] = 'T2M'
        if c.upper().startswith('RH'):
            col_map[c] = 'RH2M'
    if col_map:
        df = df.rename(columns=col_map)

    # derived temperature features
    if 'T2M_MAX' in df.columns and 'T2M_MIN' in df.columns:
        df['temp_avg'] = (df['T2M_MAX'] + df['T2M_MIN']) / 2.0
        df['temp_range'] = df['T2M_MAX'] - df['T2M_MIN']
    elif 'T2M' in df.columns:
        df['temp_avg'] = df['T2M']

    # Sort
    df = df.sort_values(['district','week_start']).reset_index(drop=True)

    # Lags
    for lag in lags:
        df[f"rain_lag_{lag}w"] = df.groupby('district')['imd_rain_mm'].shift(lag)
        if 'temp_avg' in df.columns:
            df[f"temp_avg_lag_{lag}w"] = df.groupby('district')['temp_avg'].shift(lag)
        if 'T2M_MAX' in df.columns:
            df[f"T2M_MAX_lag_{lag}w"] = df.groupby('district')['T2M_MAX'].shift(lag)
        if 'RH2M' in df.columns:
            df[f"RH2M_lag_{lag}w"] = df.groupby('district')['RH2M'].shift(lag)

    # Rolling means
    for w in roll_windows:
        df[f"rain_roll_{w}w"] = df.groupby('district')['imd_rain_mm'].rolling(window=w, min_periods=1).mean().reset_index(0,drop=True)
        if 'temp_avg' in df.columns:
            df[f"temp_roll_{w}w"] = df.groupby('district')['temp_avg'].rolling(window=w, min_periods=1).mean().reset_index(0,drop=True)
        if 'T2M_MAX' in df.columns:
            df[f"T2M_MAX_roll_{w}w"] = df.groupby('district')['T2M_MAX'].rolling(window=w, min_periods=1).mean().reset_index(0,drop=True)

    # Differences
    df['rain_diff_1w'] = df['imd_rain_mm'] - df['rain_lag_1w']

    # Filter small districts
    counts = df['district'].value_counts()
    valid = counts[counts >= min_rows].index.tolist()
    df = df[df['district'].isin(valid)].copy()

    # Labels: per-district thresholds
    drought_q = df.groupby('district')['imd_rain_mm'].quantile(0.20).rename('drought_thr').reset_index()
    flood_q = df.groupby('district')['imd_rain_mm'].quantile(0.95).rename('flood_thr').reset_index()
    df = df.merge(drought_q, on='district', how='left').merge(flood_q, on='district', how='left')
    df['drought_label'] = (df['imd_rain_mm'] < df['drought_thr']).astype(int)
    df['flood_label'] = (df['imd_rain_mm'] > df['flood_thr']).astype(int)
    if 'T2M_MAX' in df.columns:
        df['heatwave_label'] = ((df['T2M_MAX'] > 40.0) & (df.groupby('district')['T2M_MAX'].shift(1) > 40.0)).astype(int)
    else:
        df['heatwave_label'] = (df['temp_avg'] > 38.0).astype(int)

    # cleanup
    df = df.drop(columns=[c for c in ['drought_thr','flood_thr'] if c in df.columns])
    critical = ['imd_rain_mm']
    if 'temp_avg' in df.columns:
        critical.append('temp_avg')
    df_before = len(df)
    df = df.dropna(subset=critical)
    print(f"[FEATURES] dropped {df_before - len(df)} rows due to NAs in critical cols")

    # save merged features
    df.to_csv(out_merged_features, index=False)
    print(f"[FEATURES] saved merged features -> {out_merged_features}")

    # train/test split
    df['week_start'] = pd.to_datetime(df['week_start'])
    train = df[df['week_start'] < pd.to_datetime(split_date)].copy()
    test  = df[df['week_start'] >= pd.to_datetime(split_date)].copy()
    train.to_csv(out_train, index=False)
    test.to_csv(out_test, index=False)
    print(f"[SPLIT] train rows: {len(train)} | test rows: {len(test)}")
    return df, train, test

# ---------- Orchestration ----------
def main():
    # 1) IMD fetch & weekly aggregation
    if not IMD_DAILY_OUT.exists():
        print("[RUN] starting IMD daily collection (this uses imdlib; may take time)...")
        daily = collect_imd_rainfall_wrapper(DISTRICT_SHP_PATH, DISTRICT_NAME_COLUMN, START_DATE.year, END_DATE.year, str(IMD_DAILY_OUT))
    else:
        print("[RUN] IMD daily exists, loading.")
        daily = pd.read_csv(IMD_DAILY_OUT, parse_dates=['date'])

    if not IMD_WEEKLY_OUT.exists():
        print("[RUN] aggregating IMD -> weekly")
        weekly = aggregate_to_weekly_wrapper(daily, str(IMD_WEEKLY_OUT))
    else:
        print("[RUN] IMD weekly exists, loading.")
        weekly = pd.read_csv(IMD_WEEKLY_OUT, parse_dates=['week_end_sunday'])

    # 2) NASA fetch & weekly aggregation
    if not NASA_WEEKLY_OUT.exists():
        print("[RUN] fetching NASA weekly data for centroids...")
        nasawd = fetch_nasa_for_centroids(CENTROIDS_CSV, START_DATE, END_DATE, NASA_PARAMS, str(NASA_WEEKLY_OUT))
    else:
        print("[RUN] NASA weekly exists, loading.")
        nasawd = pd.read_csv(NASA_WEEKLY_OUT, parse_dates=['week_start'])

    # 3) normalize and merge
    print("[RUN] normalizing and merging IMD + NASA weekly")
    merged_weekly = normalize_and_merge(str(IMD_WEEKLY_OUT), str(NASA_WEEKLY_OUT), str(MERGED_WEEKLY_CANONICAL))

    # 4) feature engineering and labels -> merged_features + train/test
    print("[RUN] creating features & labels")
    prepare_features_from_merged(str(MERGED_WEEKLY_CANONICAL),
                                str(MERGED_FEATURES),
                                str(TRAIN_CSV),
                                str(TEST_CSV),
                                split_date=SPLIT_DATE,
                                roll_windows=ROLL_WINDOWS,
                                lags=LAGS,
                                min_rows=MIN_ROWS_PER_DISTRICT)

    print("\n\nALL DONE. Outputs:")
    print(" - IMD daily:", IMD_DAILY_OUT)
    print(" - IMD weekly:", IMD_WEEKLY_OUT)
    print(" - NASA weekly:", NASA_WEEKLY_OUT)
    print(" - Merged weekly canonical:", MERGED_WEEKLY_CANONICAL)
    print(" - Training features:", MERGED_FEATURES)
    print(" - Train set:", TRAIN_CSV)
    print(" - Test set:", TEST_CSV)

if __name__ == "__main__":
    main()
