#Example/Instructions for processing GEE NDVI/EVI data
import Functions

#ee.Initialize a project first

#call this with year, bucket name (mine was my-fire-research)
#the data will show up in the bucket (with many TIFFs) with the file prefix viirs_daily_{year}

#Note: it takes 1-2 days to fully process the data for one year (VERY slow)
Functions.gee_NDVI_EVI_year(2023, 'my-fire-research')

#These will have to be downloaded from your machine, and then combined into one file. The easiest way to do this is:
# Put all the downloaded tiffs into a folder;
# Open conda powershell in an environment that has gdal installed;
# Run the following:
#   Set-Location "{PATH TO TIFF FOLDER}"
#   gdalbuildvrt viirs_{year}.vrt viirs_daily_{year}*.tif
#   gdal_translate viirs_{year}.vrt viirs_daily_{year}_CA.tif `
#       -co TILED=YES -co COMPRESS=LZW -co BIGTIFF=YES

#This will put a new tiff viirs_daily_{year}_CA.tif in the folder that has the NDVI and EVI bands for each day in the given year.
#The tiff will have 2*days_in_year bands (one observation for NDVI and EVI each day)
#The bands alternate between variables (the first band is NDVI for jan 1, second band is EVI for jan 1, etc.)
#The rest of the processing can be handled by other functions, so this is not too important for the user.