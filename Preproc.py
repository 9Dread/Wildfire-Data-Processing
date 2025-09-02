#Python script for preprocessing data for modeling.
#This includes combining the covariates I crawled (in Output) with the NOAA Covs (in Data/NOAA). It also includes adjusting my covariates for one-timestep-ahead
#prediction.
#To run this code, the files here https://drive.google.com/drive/u/1/folders/1a0kcB2FzTs5uLb4Wnd1M15fE1iyw5Atc must be in Output,
#and the files here https://drive.google.com/drive/folders/12y_q1weaaF-ymQpaso2PBtlecjfMWvdx?usp=sharing must be in Data/NOAA. This means that
#the lengthy processing of these covariates does *not* need to be rerun to reproduce the preprocessing of the data
#for modeling unless desired.

#Problem:
#  We want to make our dataset fit for one-timestep-ahead prediction. This is because, although the
#  variables have dates attached to them, they are not available for same-day prediction in real time.
#  Specifically, fm100 and fm1000 are from gridMET (https://www.climatologylab.org/gridmet.html).
#  For these datasets, the fuel moisture for a given day is provided the day after. For instance,
#  the fuel moisture on 8/08/2025 is provided on 8/09/2025. This means we cannot assume that we have
#  the fuel moisture for 8/08/2025 to predict the intensity of events on 8/08/2025. Instead we must
#  use the information we have available in real-time, and we want the dataset to reflect this.

#Desired state of datasets:
#  Daily:
#    For covariates that update daily, use previous day. For 6hr covariates, use observed values
#    at noon of the previous day.
#  6hr:
#    For predicting at some given time step (e.g. jan 1 for 12-6pm)
#    Use most recent covs (previous day for MarcosCovs, and for instance 12pm for MichellesCovs for 12-6pm)

#Current state of dataset:
#  Output:
#    Elevation - static, does not change. Does not need to be adjusted.
#    NDVI/EVI - Uses most recent satellite observation in each grid cell. Updated every ~8 days.
#      These were already crawled by taking the *most recent observation* for any given day. Thus it
#      already reflects using the most recent available information, so no adjustment is needed.
#    fm100/fm1000 - Currently has each day's observation. Since updated one day ahead, everything
#      needs to be shifted backwards one day. Thus we need fm100/fm1000 of 2019 for jan 1 of 2020.

import xarray as xr
import pandas as pd
from Functions import get_point24deg_grid, aggregate_tif_to_cells, ensure_grid_order_matches, shift_forward_one_year
from pathlib import Path
import numpy as np

#FIRST: adjust fm100 and fm1000 forward by one day. as an intermediate step, the adjusted datasets will be
#put into a new directory, Data/AdjustedOutput
years = [2020, 2021, 2022, 2023, 2024]
varnames = ["fm100", "fm1000"]

nc_dir = Path("Output")
nc_pattern = "daily_gridded_CA_{year}.nc"

out_dir = Path("Data/AdjustedOutput")
out_dir.mkdir(parents=True, exist_ok=True)

#2019 december 31 fuel moisture paths
seed_tifs = {
    "fm100":  Path("Data/fuelmoisture_2019_dec31/fm100_2019_dec31.tif"),
    "fm1000": Path("Data/fuelmoisture_2019_dec31/fm1000_2019_dec31.tif"),
}

#make grid
grid_gdf = get_point24deg_grid()
grid_gdf_base = grid_gdf.copy() #for realigning if necessary

seed_agg_cache = {}
for v in varnames:
    seed_agg_cache[v] = aggregate_tif_to_cells(seed_tifs[v], grid_gdf_base, stats="mean")

for v in varnames:
    carry = None #last-day array carried into the next year

    for y in years:
        in_path  = nc_dir / nc_pattern.format(year=y)
        out_path = out_dir / nc_pattern.format(year=y)

        ds = xr.load_dataset(in_path)

        #align polygons to this file's cell ordering
        grid_gdf_aligned, idxs = ensure_grid_order_matches(ds, grid_gdf_base, id_col="cell_id")

        #build the seed for the first day of this year:
        if y == years[0]:
            #2020: use aggregated 2019-12-31 TIFF
            seed_full_order = seed_agg_cache[v]
            seed_aligned = seed_full_order[idxs]
            first_day_value = seed_aligned
        else:
            #2021...: use carry (which we saved as the last day of previous year's original data)
            if carry is None:
                raise RuntimeError("Carry is None for a year > first. Logic error.")
            first_day_value = carry

        #apply shift within this year
        old = ds[v].values #shape (T, C)
        new = shift_forward_one_year(old, first_day_value)  #shape (T, C)

        #update the dataset variable values
        ds[v].values[:] = new

        #prepare carry for next year; the last day of this year's ORIGINAL data becomes first day of next year
        carry = old[-1, :].copy()

        #write out the fixed file
        ds.to_netcdf(out_path)
        ds.close()

#NEXT: using adjusted datasets from above, combine with NOAA for each year in years. First make daily dataset:

nc_dir = Path("Data/AdjustedOutput")
nc_pattern = "daily_gridded_CA_{year}.nc"

csv_dir = Path("Data/NOAA")
csv_pattern = "California_HRRR_daily{year}06.csv"

out_dir = Path("Output_Preprocessed")
out_dir.mkdir(parents=True, exist_ok=True)

for y in years:
    in_path_nc = nc_dir / nc_pattern.format(year=y)
    in_path_csv = csv_dir / csv_pattern.format(year=y)
    out_path = out_dir / nc_pattern.format(year=y)

    ds = xr.load_dataset(in_path_nc)
    df = pd.read_csv(in_path_csv)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    #drop useless columns (most are all zeros)
    df = df.drop(columns=["Cloud mixing ratio", 
    "Fraction of cloud cover", 
    "Graupel (snow pellets)",
    "Rain mixing ratio",
    "Snow mixing ratio",
    "unknown",
    "Latitude",
    "Longitude"])

    #loop thru all columns except date and convert to numeric
    exclude = "Date"
    include = [col for col in df.columns if col != exclude]
    for col in include: 
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    #drop na and drop duplicates; usually resolves errors where we have extra rows
    df = df.dropna()
    df = df.drop_duplicates()
    df["Cell_ID"] = df["Cell_ID"].astype(ds["cell"].dtype) #make cell ids match datatypes

    #make xarray dataset out of the csv
    df_xr = (
            df.set_index(["Date", "Cell_ID"])
              .to_xarray()
              .rename({"Date": "time", "Cell_ID": "cell"})
              .transpose("time", "cell", ...)
        )
    #Align exactly to ds coords
    df_xr = df_xr.reindex_like(ds)
    #align indices with the other ds
    #df_xr = df_xr.reindex(time=ds.time, cell=ds.cell)
    ds_out = xr.merge([ds, df_xr], compat="no_conflicts")
    for v in set(df_xr.data_vars) - set(ds.data_vars):
         frac = 1 - np.isnan(ds_out[v]).mean().item()
         print(y, v, "non-NaN coverage:", f"{100*frac:.5f}%")
    ds_out.to_netcdf(out_path)