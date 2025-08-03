#Function implementations for data processing.
import ee, geemap
import pandas as pd, numpy as np, geopandas as gpd
from shapely.geometry import box
import xarray as xr, rasterio as rio
from rasterstats import zonal_stats
from tqdm import tqdm
import seamless_3dep as s3dep
import rioxarray as rxr
from pathlib import Path

def extract_vegetation(nc_dataset, var_name, grid_gdf, year):
    """
    Helper function for combine_data. Extracts vegetation to the grid.
    """
    #Attach CRS,
    #Reproject GeoDataFrame to match raster CRS just in case
    if not nc_dataset.rio.crs:
        nc_dataset = nc_dataset.rio.write_crs("EPSG:4326")

    grid = grid_gdf.to_crs(nc_dataset.rio.crs)


    n_days = nc_dataset.sizes["day"]
    n_cells = len(grid)
    means = np.full((n_days, n_cells), np.nan)


    for t in tqdm(range(n_days), desc=var_name):
        day_da = nc_dataset.isel(day=t)
        arr = day_da[var_name].values
        transform = day_da.rio.transform()


        stats = zonal_stats(
            vectors=grid.geometry,
            raster=arr,
            affine=transform,
            stats=["mean"],
            nodata=np.nan,
            all_touched=False
        )


        means[t] = [s["mean"] if s["mean"] is not None else np.nan for s in stats]

    #save numpy array
    np.save(f'Output/{var_name}_{year}_means.npy', means)

    return means, nc_dataset["day"].values

def combine_data(year):
    """
    Function to make a netcdf dataset over a 0.24 degree grid of California for the given year.
    This downsamples data from the other covariate files using an average, so it can be very slow.
    Output is in Output folder.
    """

    #First make the california grid with 0.24 degree resolution
    pad, dx = 0.30, 0.24 #Grid dimensions

    #Note that the CA bounding box in epsg4326 is (-124.409591, 32.534156, -114.131211, 42.009518)
    xmin, xmax = -124.409591-pad, -114.131211+pad
    ymin, ymax = 32.534156-pad, 42.009518+pad

    cells = [box(x, y, x+dx, y+dx) #square cells
         for x in np.arange(xmin, xmax, dx)
         for y in np.arange(ymin, ymax, dx)]
    grid_gdf = gpd.GeoDataFrame({"cell_id": range(len(cells))}, geometry=cells, crs=4326)
    #California outline
    #downloading county polygons, "FIPS : 06" is California
    ca = (gpd.read_file("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json")
      .loc[lambda d: d.id.str.startswith("06")]
      .to_crs(4326) # ensure same CRS
      .dissolve()) # single polygon
    
    #Keep only grid cells that intersect California
    grid_gdf = grid_gdf.loc[grid_gdf.geometry.intersects(ca.geometry.iloc[0])].reset_index(drop=True)

    #get Digital Elevation Map for elevation covariate
    CA_BBOX = (-124.409591, 32.534156, -114.131211, 42.009518)
    data_dir = Path("Data")
    elev = s3dep.get_map("DEM", CA_BBOX, data_dir, res = 500)
    elev = rxr.open_rasterio(elev[0].as_posix()).rio.reproject("EPSG:4326")
    elev = elev.squeeze()

    #Get fm100 for the given year
    fm100 = xr.open_dataset(f"Data/fm100/fm100_{year}.nc")
    fm100 = fm100["dead_fuel_moisture_100hr"]
    fm100 = fm100.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    fm100 = fm100.rio.write_crs("EPSG:4326", inplace=False)

    #Get fm1000 for the given year
    fm1000 = xr.open_dataset(f"Data/fm1000/fm1000_{year}.nc")
    fm1000 = fm1000["dead_fuel_moisture_1000hr"]
    fm1000 = fm1000.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    fm1000 = fm1000.rio.write_crs("EPSG:4326", inplace=False)

    #Get NDVI and EVI for the given year
    veg_stack = rxr.open_rasterio(f"Data/NDVI_EVI/viirs_daily_{year}_CA.tif", masked=True)
    n_bands = veg_stack.sizes["band"]
    days = n_bands // 2
    dates = pd.date_range(f"{year}-01-01", periods=days, freq="D")
    even = veg_stack.isel(band=slice(0, None, 2)) #0,2,4… = NDVI
    odd = veg_stack.isel(band=slice(1, None, 2)) #1,3,5… = EVI
    ndvi = even.assign_coords(band=dates).rename({'band':'day'})
    evi = odd.assign_coords(band=dates).rename({'band':'day'})
    ndvi_set = xr.Dataset({'NDVI': ndvi}) \
           .rio.write_crs(veg_stack.rio.crs, inplace=False)
    evi_set = xr.Dataset({'EVI': evi}) \
           .rio.write_crs(veg_stack.rio.crs, inplace=False)

    n_days = ndvi_set.sizes["day"]

    #Now compute results
    results = {}
    time_ref = None
    veg_vars = {
        "NDVI": ndvi_set,
        "EVI": evi_set
    }
    fm_vars = {
        "fm100": fm100,
        "fm1000": fm1000
    }
    #vegetation
    for var_name, dataset in veg_vars.items():
        means, times = extract_vegetation(dataset, var_name, grid_gdf, year)
        results[var_name] = means
        if time_ref is None:
            time_ref = times  #use time from first file
    #fuel moisture
    for var_name, dataset in fm_vars.items():
        means, times = extract_fuel_moisture(dataset, var_name, grid_gdf)
        results[var_name] = means
        if time_ref is None:
            time_ref = times  #use time from first variable
    #elevation
    results["elevation"] = extract_elevation(elev, n_days, grid_gdf)
    #Combine everything
    n_cells = len(grid_gdf)
    ds_out = xr.Dataset(
        {
            var_name: (("time", "cell"), results[var_name])
            for var_name in results
        },
        coords={
            "time": time_ref,
            "cell": np.arange(n_cells),
            "lon": ("cell", grid_gdf.to_crs('EPSG:26910').geometry.centroid.to_crs('EPSG:4326').x),
            "lat": ("cell", grid_gdf.to_crs('EPSG:26910').geometry.centroid.to_crs('EPSG:4326').y),
        }
    )
    #save as NetCDF
    ds_out.to_netcdf(f"Output/daily_gridded_CA_{year}.nc")

def combine_data_from_np(year):
    """
    This thing basically combines the data if combine_data fails at some point but NDVI/EVI processing (the costly part) was finished and saved as a numpy array.
    This happened for leap years (366 days instead of 365). Should be fixed now but this patches things together if something happens for whatever reason.
    """
    #First make the california grid with 0.24 degree resolution
    pad, dx = 0.30, 0.24 #Grid dimensions

    #Note that the CA bounding box in epsg4326 is (-124.409591, 32.534156, -114.131211, 42.009518)
    xmin, xmax = -124.409591-pad, -114.131211+pad
    ymin, ymax = 32.534156-pad, 42.009518+pad

    cells = [box(x, y, x+dx, y+dx) #square cells
         for x in np.arange(xmin, xmax, dx)
         for y in np.arange(ymin, ymax, dx)]
    grid_gdf = gpd.GeoDataFrame({"cell_id": range(len(cells))}, geometry=cells, crs=4326)
    #California outline
    #downloading county polygons, "FIPS : 06" is California
    ca = (gpd.read_file("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json")
      .loc[lambda d: d.id.str.startswith("06")]
      .to_crs(4326) # ensure same CRS
      .dissolve()) # single polygon
    #Keep only grid cells that intersect California
    grid_gdf = grid_gdf.loc[grid_gdf.geometry.intersects(ca.geometry.iloc[0])].reset_index(drop=True)

    #get Digital Elevation Map for elevation covariate
    CA_BBOX = (-124.409591, 32.534156, -114.131211, 42.009518)
    data_dir = Path("Data")
    elev = s3dep.get_map("DEM", CA_BBOX, data_dir, res = 500)
    elev = rxr.open_rasterio(elev[0].as_posix()).rio.reproject("EPSG:4326")
    elev = elev.squeeze()

    #Get fm100 for the given year
    fm100 = xr.open_dataset(f"Data/fm100/fm100_{year}.nc")
    fm100 = fm100["dead_fuel_moisture_100hr"]
    fm100 = fm100.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    fm100 = fm100.rio.write_crs("EPSG:4326", inplace=False)

    #Get fm1000 for the given year
    fm1000 = xr.open_dataset(f"Data/fm1000/fm1000_{year}.nc")
    fm1000 = fm1000["dead_fuel_moisture_1000hr"]
    fm1000 = fm1000.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    fm1000 = fm1000.rio.write_crs("EPSG:4326", inplace=False)
    n_days = fm1000.sizes["day"]

    results = {}
    time_ref = None
    veg_vars = {
        "NDVI": f"Output/NDVI_{year}_means.npy",
        "EVI": f"Output/EVI_{year}_means.npy"
    }
    fm_vars = {
        "fm100": fm100,
        "fm1000": fm1000
    }
    #vegetation
    for var_name, path in veg_vars.items():
        means = np.load(path)
        results[var_name] = means
    #fuel moisture
    for var_name, dataset in fm_vars.items():
        means, times = extract_fuel_moisture(dataset, var_name, grid_gdf)
        results[var_name] = means
        if time_ref is None:
            time_ref = times  #use time from first variable
    #elevation
    results["elevation"] = extract_elevation(elev, n_days, grid_gdf)
    #Combine everything
    n_cells = len(grid_gdf)

    ds_out = xr.Dataset(
        {
            var_name: (("time", "cell"), results[var_name])
            for var_name in results
        },
        coords={
            "time": time_ref,
            "cell": np.arange(n_cells),
            "lon": ("cell", grid_gdf.to_crs('EPSG:26910').geometry.centroid.to_crs('EPSG:4326').x),
            "lat": ("cell", grid_gdf.to_crs('EPSG:26910').geometry.centroid.to_crs('EPSG:4326').y),
        }
    )
    #save as NetCDF
    ds_out.to_netcdf(f"Output/daily_gridded_CA_{year}.nc")

def fill_forward(collection, start, end):
    """Return an ImageCollection with one frame per calendar day (for getting GEE covariate data).
       Each day inherits the most recent composite."""
    start = ee.Date(start); end = ee.Date(end)
    n = end.difference(start, 'day')
    #list of days
    dates = ee.List.sequence(0, n).map(lambda d:
        start.advance(ee.Number(d), 'day'))
    def make_day(d):
        #Helper function for GEE code
        d = ee.Date(d)
        #last composite whose start <= this day
        img = collection.filterDate('2000-01-01', d.advance(1,'day')) \
                    .sort('system:time_start', False) \
                    .first()
        return ee.Image(img).set({'system:time_start': d.millis()})
    return ee.ImageCollection(ee.List(dates.map(make_day)))

def extract_fuel_moisture(nc_dataset, var_name, grid_gdf):
    """
    Helper function for combine_data. Extracts fuel moisture to the grid.
    """
    #Attach CRS,
    #Reproject GeoDataFrame to match raster CRS just in case
    if not nc_dataset.rio.crs:
        nc_dataset = nc_dataset.rio.write_crs("EPSG:4326")

    grid = grid_gdf.to_crs(nc_dataset.rio.crs)

    n_days = nc_dataset.sizes["day"]
    n_cells = len(grid)
    means = np.full((n_days, n_cells), np.nan)

    for t in tqdm(range(n_days), desc=var_name):
        day_da = nc_dataset.isel(day=t)
        arr = day_da.values
        transform = day_da.rio.transform()


        stats = zonal_stats(
            vectors=grid.geometry,
            raster=arr,
            affine=transform,
            stats=["mean"],
            nodata=np.nan,
            all_touched=False
        )


        means[t] = [s["mean"] if s["mean"] is not None else np.nan for s in stats]
    return means, nc_dataset["day"].values


def extract_elevation(elevation_da, n_days, grid_gdf):
    """
    Helper function for combine_data. Extracts elevation to the grid.
    """
    n_cells = len(grid_gdf)
    means = np.full((n_days, n_cells), np.nan)
    #just do this once now
    arr = elevation_da.values
    transform = elevation_da.rio.transform()


    stats = zonal_stats(
        vectors=grid_gdf.geometry,
        raster=arr,
        affine=transform,
        stats=["mean"],
        nodata=np.nan,
        all_touched=False
    )
    for t in tqdm(range(n_days), desc="elevation"):
        means[t] = [s["mean"] if s["mean"] is not None else np.nan for s in stats]
    return means


def gee_NDVI_EVI_year(year, bucket_name):
    """
    Starts an Earth Engine task that grabs NDVI and EVI across california for the given year, with daily observations.
    Sends all data into a collection of GeoTIFFs (with 1 band per each observation) in the given cloud storage bucket.
    You must do ee.Initialize first
    """
    file_pref = f'viirs_daily_{year}_1_'
    start = f'{year}-01-01'
    end = f'{year}-12-31'
    #region of interest  (California outline from Census TIGER)
    ca = ee.FeatureCollection('TIGER/2018/States') \
        .filter(ee.Filter.eq('NAME', 'California'))
    
    vnp = ee.ImageCollection('NASA/VIIRS/002/VNP13A1') \
          .filterDate(start, end) \
          .select(['NDVI', 'EVI']) \
          .map(lambda img: img.clip(ca)) #auto–mosaic tiles
    daily = fill_forward(vnp, start, end)
    #export to a Cloud Storage (bucket_name) as a Cloud-Optimised TIFF stack
    task = ee.batch.Export.image.toCloudStorage(
        image  = daily.toBands(),
        description = f'CA_VIIRS_NDVI_EVI_{year}_daily',
        bucket = bucket_name,
        fileNamePrefix = file_pref,
        region = ca.geometry(),
        scale = 500, #500m resolution
        crs = 'EPSG:4326',
        formatOptions = {'cloudOptimized': True}
    )
    task.start()
