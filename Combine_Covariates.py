#Example/Instructions for combining fuel moisture (fm100,fm1000), elevation, EVI, and NDVI into a 0.24-degree grid
#with daily observations in California for any given year

#Outputs a .nc dataset in Output folder

#To start, you should have the fm100 and fm1000 netCDF files in Data/fm100 and Data/fm1000 for the year you wish to process
#The file names should be in the form fm100_{year}.nc and fm1000_{year}.nc
#These can be downloaded from gridMET at https://www.climatologylab.org/gridmet.html

#You should also have the processed NDVI and EVI tiff stack for the years from GEE_NDVI_EVI.py in Data/NDVI_EVI
import Functions

#If everything is set up correctly, you should just have to run:
Functions.combine_data(2021)
#(for any given year)

#Functions.combine_data_from_np(2020) if NDVI/EVI were processed and saved in Output folder as numpy but an error occurred so the whole dataset was not saved properly
