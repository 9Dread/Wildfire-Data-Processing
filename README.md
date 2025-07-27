# Wildfire-Data-Processing

Data processing for some environmental covariates in GRID-BENCH. Dead fuel moisture (fm100 and fm1000) are taken from [gridMET](https://www.climatologylab.org/gridmet.html), and NDVI/EVI are taken from NASA VIIRS. Elevation is taken from USGS 3DEP.

This code takes these variables for a given year, aggregates them to a 0.24-degree grid in California, and outputs a netCDF dataset of the raster containing all daily observations.

GEE_NDVI_EVI.py runs an earth engine script to get NDVI and EVI for the given year. It takes Google Cloud days to process each year; the code is here for documentation purposes.

Combine_Covariates.py combines all of the datasets into the 0.24-degree grid and outputs the netCDF file.

Instructions are provided in each file for setup.
