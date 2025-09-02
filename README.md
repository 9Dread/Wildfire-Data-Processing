# Wildfire-Data-Processing

Data processing for some environmental covariates in GRID-BENCH. Dead fuel moisture (fm100 and fm1000) are taken from [gridMET](https://www.climatologylab.org/gridmet.html), and NDVI/EVI are taken from NASA VIIRS. Elevation is taken from USGS 3DEP.

This code takes these variables for a given year, aggregates them to a 0.24-degree grid in California, and outputs a netCDF dataset of the raster containing all daily observations.

GEE_NDVI_EVI.py runs an earth engine script to get NDVI and EVI for the given year. It takes Google Cloud days to process each year; the code is here for documentation purposes.

Combine_Covariates.py combines all of the datasets into the 0.24-degree grid and outputs the netCDF file.

Instructions are provided in each file for setup.

Fully processed grids are in this [drive](https://drive.google.com/drive/folders/1a0kcB2FzTs5uLb4Wnd1M15fE1iyw5Atc?usp=sharing).

The rest of the data processing pipeline (Preproc.py) prepares data for wildfire risk modeling in my [Wildfire-Point-Process](https://github.com/9Dread/Wildfire-Point-Process) repository. This includes combining the data with processed NOAA covariates (processed by another student to the same grid). The time resolution of the NOAA covariates is 6 hours. We make two separate datasets: a daily one, which uses covariates at 6am from NOAA; and a 6 hour one, which uses the most recent observation of daily covariates.

To run Preproc.py, make sure that the Output folder contains all of the fully processed datasets from Combine_Covariates.py. If you would like to download them instead of running the code, the link is above. Also ensure that the NOAA covariates are in Data/NOAA; these can be found at this other [drive](https://drive.google.com/drive/folders/12y_q1weaaF-ymQpaso2PBtlecjfMWvdx?usp=sharing).

The output files of this final pipeline are in the Output_Preprocessed folder and included in this repository.