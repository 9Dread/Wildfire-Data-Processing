# Wildfire-Data-Processing

Data processing for some environmental covariates in GRID-BENCH. Dead fuel moisture (fm100 and fm1000) are taken from [gridMET](https://www.climatologylab.org/gridmet.html), and NDVI/EVI are taken from NASA VIIRS. Elevation is taken from USGS 3DEP.

This code takes these variables for a given year, aggregates them to a 0.24-degree grid in California, and outputs a netCDF dataset of the raster containing all daily observations.

GEE_NDVI_EVI.py runs an earth engine script to get NDVI and EVI for the given year. It takes Google Cloud days to process each year; the code is here for documentation purposes.

Combine_Covariates.py combines all of the datasets into the 0.24-degree grid and outputs the netCDF file.

Instructions are provided in each file for setup.

Fully processed grids are in this [drive](https://drive.google.com/drive/folders/1a0kcB2FzTs5uLb4Wnd1M15fE1iyw5Atc?usp=sharing).

The rest of the data processing pipeline (Preproc.py) prepares data for wildfire risk modeling in my [Wildfire-Point-Process](https://github.com/9Dread/Wildfire-Point-Process) repository. This includes combining the data with processed NOAA covariates (processed by another student to the same grid). The time resolution of the NOAA covariates is 6 hours. We make two separate datasets: a daily one, which uses covariates at 6am from NOAA; and a 6 hour one, which uses the most recent observation of daily covariates.

To run Preproc.py, make sure that the Output folder contains all of the fully processed datasets from Combine_Covariates.py. If you would like to download them instead of running the code, the link is above. Also ensure that the NOAA covariates are in Data/NOAA; these can be found at this other [drive](https://drive.google.com/drive/folders/12y_q1weaaF-ymQpaso2PBtlecjfMWvdx?usp=sharing).

This final pipeline produces two kinds of datasets:

6h_gridded_CA_{year}.nc are the 6hr resolution datasets for each year. The daily covariates (fm100,fm1000) are taken to be from the previous day to prevent data leakage in prediction. The 6hr NOAA covariates are provided for 00:00, 06:00, 12:00, and 18:00 every day. These are used to predict the wildfire risk for the next 6 hours. Missing values were small in quantity and filled using the most recent observation.

daily_gridded_CA_{year}.nc are the daily datasets. In this case all variables, including NOAA, are treated as daily resolution and thus must be adjusted to prevent leakage. We use 18:00 observations to predict the risk of the next day. Here I dropped the first day of 2020 because I do not have NOAA covariates on 12-31-2019 (as they were processed by another student).

The since the daily dataset is relatively small, it is in the [Output_Preprocessed](Output_Preprocessed) folder and included in this repository. The 6-hr datasets are in this [drive](https://drive.google.com/drive/folders/1KGSzYTvCHx74Bo6F5DgDdEgatKCvEukw?usp=sharing).
