# %% [markdown]
# Composite analysis is a commonly used statistical technique to determine some of the basic structural characteristics of a meteorological or climatological phenomenon that are difficult to observe in totality (such as a hurricane, a squall line thunderstorm, or a cold front), or phenomenon which occur over time (e.g., the weather/climate over a given geographic area). Composite Analysis is one of those very recurrent and celebrated methods of climate science, which could be quite useful for exploring the large scale impacts of teleconnections from modes of atmospheric variability such as El Nino.
# 
# There are a number of steps necessary to form composites of any given phenomenon (Lee, 2011).
# 
# The first step is choosing a basis for the analysis, more specifically a positive and negative basis must be chosen. For example, in some work on ENSO composite analysis, generally the positive basis is used to describe El Niño events, and the negative basis is used to describe La Niña events.
# After the basis is formed, and events chosen, these events are then averaged, and the positive and negative averaged events are subtracted from one another.
# Finally, statistical significance is determined by using a two tailed student-t test, and for the cases evaluated, the confidence interval is generally set at 0.95.
# In this notebook, we will carry out a primary composite study of the El Niño influence on the precipitation over the Southern Africa. This notebook extended the idea from Tristan Hauser, so credit should also go to him.

# %% [markdown]
# Author Marie-Aude Pradal

# %%
import numpy as np               
import pandas as pd              
import xarray as xr

import matplotlib.pyplot as plt  
import matplotlib as mpl          
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from helpers import *

%matplotlib inline                
_ = plt.xkcd()

# %% [markdown]
# 1. Identify El Niño events
# 
# El Niño/Southern Oscillation (ENSO) is an irregularly periodic variation in winds and sea surface temperatures over the tropical eastern Pacific Ocean, which is the most important coupled ocean-atmosphere phenomenon to cause global climate variability on interannual time scales. The warming phase of the sea temperature is known as El Niño and the cooling phase as La Niña. The extremes of this climate pattern's oscillations cause extreme weather (such as floods and droughts) in many regions of the world. Developing countries dependent upon agriculture and fishing, particularly those bordering the Pacific Ocean, are the most affected.
# 
# There are many separate indices available to describe ENSO events. Here Multivariate ENSO Index (MEI) was used. In the interest of determining a full affect of both the atmospheric and oceanic aspects of ENSO, the MEI has been utilized as equatorial Pacific variables of both the atmosphere and ocean go into the formation of this index (Wolter, 1998). As the MEI indices are scaled by variability, we take any MEI value>1 to be an El Niño event.

# %%
plt.rcParams['font.family'] = 'Arial'

# %%
# Define the column names (Year + 12 months)
columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Read the file into a DataFrame
df = pd.read_csv('/Users/marie-audepradal/Documents/enso_indices.txt', delim_whitespace=True, header=None, names=columns)

# Display the first few rows
print(df.head())

# Convert to long format
df_long = df.melt(id_vars='Year', var_name='Month', value_name='ENSO_Index')

# Convert to datetime for proper x-axis
month_to_num = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
df_long['Month_Num'] = df_long['Month'].map(month_to_num)
df_long['Date'] = pd.to_datetime(dict(year=df_long['Year'], month=df_long['Month_Num'], day=1))

# Sort by date
df_long = df_long.sort_values('Date')
# Plot the continuous time series
plt.figure(figsize=(14, 6))
plt.plot(df_long['Date'], df_long['ENSO_Index'], label='ENSO Index', color='blue')
plt.axhline(0, color='gray', linestyle='--')
plt.title('ENSO Index Over Time')
plt.xlabel('Year')
plt.ylabel('ENSO Index')
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# fill with color the events corresponding to nino index GT 1

# %%
# Step 4: Plot the data with highlighting
plt.figure(figsize=(15, 5))
plt.plot(df_long['Date'], df_long['ENSO_Index'], label='ENSO Index', color='black')

# Highlight region between -1 and 1
in_range = df_long['ENSO_Index'].between(-1, 1)
plt.fill_between(df_long['Date'], df_long['ENSO_Index'], where=in_range, color='skyblue', label='-1 < ENSO < 1')

plt.axhline(1, color='gray', linestyle='--')
plt.axhline(-1, color='gray', linestyle='--')
plt.title('ENSO Index Over Time with Range Highlighted')
plt.xlabel('Date')
plt.ylabel('ENSO Index')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# save the filtered values into a new variable and in a netcdf file

# %%
# Step 1: Load the data
columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df = pd.read_csv('/Users/marie-audepradal/Documents/enso_indices.txt', delim_whitespace=True, header=None, names=columns)

# Step 2: Reshape into long format
enso_long = df.melt(id_vars='Year', var_name='Month', value_name='ENSO')
enso_long['Date'] = pd.to_datetime(enso_long['Year'].astype(str) + enso_long['Month'], format='%Y%b')
enso_long = enso_long.sort_values('Date').reset_index(drop=True)

# Step 3: Plotting
plt.figure(figsize=(15, 5))
plt.plot(enso_long['Date'], enso_long['ENSO'], label='ENSO Index', color='gray')

# Highlight values between -1 and 1
mask = enso_long['ENSO'].between(-1, 1)
plt.fill_between(enso_long['Date'], enso_long['ENSO'],
                 where=mask, color='lightblue', label='-1 < ENSO < 1')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.axhline(-1, color='red', linestyle='--', linewidth=0.8)
plt.axhline(1, color='red', linestyle='--', linewidth=0.8)
plt.title('ENSO Index Over Time')
plt.xlabel('Date')
plt.ylabel('ENSO Index')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# Step 4: Save filtered data to NetCDF using xarray
# Create a DataArray
filtered_data = enso_long[mask]
da = xr.DataArray(filtered_data['ENSO'].values,
                  coords={'time': filtered_data['Date'].values},
                  dims=['time'],
                  name='enso_index')

# Save to NetCDF
#ds = da.to_dataset()
#ds.to_netcdf('/Users/marie-audepradal/Documents/filtered_enso_indices.nc')

# %% [markdown]
# plot extreme values: nino index <=1.5 or >1.5

# %%
# Conditions for highlighting la Nina extreme events
extreme_mask_nino = (df_long['ENSO_Index'] > 2) #| (df_long['ENSO_Index'] < -1.5)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(df_long['Date'], df_long['ENSO_Index'], color='gray', label='ENSO Index')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
years = pd.date_range(start='1979', end='2025', freq='YS')
for year in years:
    plt.axvline(x=year, color='gray', linestyle='--', linewidth=0.5)

# Highlight extremes
plt.plot(df_long.loc[extreme_mask_nino, 'Date'],
         df_long.loc[extreme_mask_nino, 'ENSO_Index'],
         color='red', linestyle='none', marker='o', label='Extreme El Nino')

plt.title('ENSO Index Over Time')
plt.xlabel('Date')
plt.ylabel('ENSO Index')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Conditions for highlighting neutral events
extreme_mask_neutral = (df_long['ENSO_Index'] < 0.5) & (df_long['ENSO_Index'] > -0.5)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(df_long['Date'], df_long['ENSO_Index'], color='gray', label='ENSO Index')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
years = pd.date_range(start='1979', end='2025', freq='YS')
for year in years:
    plt.axvline(x=year, color='gray', linestyle='--', linewidth=0.5)

# Highlight extremes
plt.plot(df_long.loc[extreme_mask_neutral, 'Date'],
         df_long.loc[extreme_mask_neutral, 'ENSO_Index'],
         color='red', linestyle='none', marker='o', label='neutral phase')

plt.title('ENSO Index Over Time')
plt.xlabel('Date')
plt.ylabel('ENSO Index')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# save extreme values as new variable in netcdf format

# %%
# Select extreme values El Nino
extreme_nino_df = df_long[extreme_mask_nino].copy()
extreme_nino_df.set_index('Date', inplace=True)

# Select extreme values neutral phase
extreme_neutral_df = df_long[extreme_mask_neutral].copy()
extreme_neutral_df.set_index('Date', inplace=True)
# Convert to xarray Dataset
ds_nino = xr.Dataset(
    {"enso_index": ("time", extreme_nino_df['ENSO_Index'].values)},
    coords={"time": extreme_nino_df.index}
)

# Save to NetCDF
ds_nino.to_netcdf("/Users/marie-audepradal/Documents/nino.nc")

# Convert to xarray Dataset
ds_neutral = xr.Dataset(
    {"enso_index": ("time", extreme_neutral_df['ENSO_Index'].values)},
    coords={"time": extreme_neutral_df.index}
)

# Save to NetCDF
ds_neutral.to_netcdf("/Users/marie-audepradal/Documents/neutral.nc")

# %% [markdown]
# read ERA5 data from a netcdf file

# %%
file_path = "/Users/marie-audepradal/Documents/ERA5SST.nc"
era5_ds = xr.open_dataset(file_path)

# %%
file_path=  "/Users/marie-audepradal/Documents/nino.nc"
nino15_ds = xr.open_dataset(file_path)
file_path=  "/Users/marie-audepradal/Documents/neutral.nc"
neutral_ds = xr.open_dataset(file_path)

# %%
# Extract time values from both datasets
era5_times = era5_ds['valid_time']
nino15_times = nino15_ds['time']
neutral_times = neutral_ds['time']
# Find common timestamps
matching_nino = np.intersect1d(era5_times.values, nino15_times.values)
matching_neutral = np.intersect1d(era5_times.values, neutral_times.values)
# Select the corresponding entries from the enso2023 dataset
matching_nino15_data = era5_ds.sel(valid_time=matching_nino)
matching_neutral_data = era5_ds.sel(valid_time=matching_neutral)


# %%
matching_neutral_data

# %%
# Compute the average over time of the matched nina data

mean_sstKn = matching_neutral_data['sst'].mean(dim='valid_time')
mean_sstn = mean_sstKn - 273.15
# Plot the 2D map
plt.figure(figsize=(12, 6))
plt.pcolormesh(matching_neutral_data['longitude'], matching_neutral_data['latitude'], mean_sstn, shading='auto', cmap='viridis')
plt.colorbar(label='SST')
plt.title('Composite map of SST for the neutral phase')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.show()

# %%
# Compute the average over time of the matched nino extreme events

mean_sstKo = matching_nino15_data['sst'].mean(dim='valid_time')
mean_ssto = mean_sstKo.sel(latitude=slice(5, -5), longitude=slice(210, 270)) - 273.15
mean_sstKn = matching_neutral_data['sst'].mean(dim='valid_time')
mean_sstn = mean_sstKn.sel(latitude=slice(5, -5), longitude=slice(210, 270)) - 273.15

# For better visualization, focus on Nino 3 region. 
# plot the composite SST for extreme Nino and for Neutral Phase

# Plot the 2D map
plt.figure(figsize=(12, 6))
plt.pcolormesh(mean_ssto['longitude'], mean_ssto['latitude'], mean_ssto, shading='auto', cmap='viridis', vmin=22,
    vmax=30)
plt.colorbar(label='SST')
plt.title('Composite map of SST for extreme Nino Events')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.show()



# Plot the 2D map
plt.figure(figsize=(12, 6))
plt.pcolormesh(mean_sstn['longitude'], mean_sstn['latitude'], mean_sstn, shading='auto', cmap='viridis', vmin=22,
    vmax=30)
plt.colorbar(label='SST')
plt.title('Composite map of SST for the neutral phase')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.show()



# %%

# Load NetCDF data
ds_ERA5L = xr.open_dataset("/Users/marie-audepradal/Documents/1970-2024_tpe_ERA5Land_monthly.nc")  # Replace with your file path

# 2. Identify which variable holds temperature
#    (in your file it’s “t2m”, but this will print all data_vars)
print("Data variables in the file:", list(ds_ERA5L.data_vars))
# → e.g. ['t2m']

# 3. Select the temperature DataArray
temp = ds_ERA5L['t2m']
tp = ds_ERA5L['tp']

# 4. Compute the monthly climatology over valid_time
#    (group by the month of the timestamp and take the mean across all years)
temp_clim = temp.groupby('valid_time.month').mean(dim='valid_time')
tp_clim = tp.groupby('valid_time.month').mean(dim='valid_time')
temp_clim.name = 't2m_climatology'
tp_clim.name = 'precip_climatology'

# 5. Save the result to a new NetCDF
temp_output_path = '/Users/marie-audepradal/Documents/temp_climatology_1970-2024_ERA5Land_monthly.nc'
temp_clim.to_netcdf(temp_output_path)
print(f"Monthly climatology written to: {temp_output_path}")
tp_output_path = '/Users/marie-audepradal/Documents/precip_climatology_1970-2024_ERA5Land_monthly.nc'
tp_clim.to_netcdf(tp_output_path)
print(f"Monthly climatology written to: {tp_output_path}")
print(temp_clim)
print(tp_clim)


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio import features
from rasterio.transform import from_origin
import fiona
from shapely.geometry import shape, Polygon, MultiPolygon, box
from shapely.ops import transform
import xarray as xr
import numpy as np

# Re-open the datasets
ds_nino = xr.open_dataset('/Users/marie-audepradal/Documents/nino.nc')
ds_t2 = xr.open_dataset('/Users/marie-audepradal/Documents/t2_1983_ERA5Land_monthly.nc')

# Identify the appropriate time variables
times_nino = ds_nino['time'].values
# ERA5-Land t2m uses 'valid_time' instead of 'time'
times_t2 = ds_t2['valid_time'].values

print("Nino dataset time coordinates:")
print(times_nino)

print("\nERA5-Land t2m dataset valid_time coordinates:")
print(times_t2)

# Find the common times between the two datasets
common_times = np.intersect1d(times_t2, times_nino)

print(f"\nCommon times ({len(common_times)} entries):")
print(common_times)

# Select t2m data for those common times
if 't2m' in ds_t2:
    t2m_var = 't2m'
else:
    # List variables if 't2m' not found
    print("\nVariables in ERA5-Land dataset:", list(ds_t2.data_vars))
    t2m_var = list(ds_t2.data_vars)[0]  # fallback

# Subset using valid_time as the coordinate
ds_t2_sel = ds_t2.sel(valid_time=common_times)[[t2m_var]]

# Save the subset to a new NetCDF file
output_path = '/Users/marie-audepradal/Downloads/t2m_matched_times.nc'
ds_t2_sel.to_netcdf(output_path)

print(f"\nSubset t2m data saved to: {output_path}")
print(ds_t2_sel)


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from rasterio import features
from rasterio.transform import from_origin

# ─────────────────────────────────────────────────────────────────────────────
#  1.  LOAD “t2m_matched_times.nc” (only once).  We will pull out the 't2m' 
#     DataArray rather than calling .values on the entire Dataset. 
#     That avoids the “method object is not subscriptable” error.
# ─────────────────────────────────────────────────────────────────────────────
ds_t2_sel = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')

# The variable is named 't2m' in ds_t2_sel.  Fetch it as a DataArray:
t2m_da = ds_t2_sel['t2m']  # dims are (valid_time, latitude, longitude)

# ─────────────────────────────────────────────────────────────────────────────
#  2.  EXTRACT lon/lat and do the 0–360 → (–180,180) conversion exactly as before
# ─────────────────────────────────────────────────────────────────────────────
lons = ds_t2_sel['longitude'].values.copy()
lats = ds_t2_sel['latitude'].values

# If the longitudes run 0..359.9, shift them into −180..+180 and reorder the DataArray
if lons.min() >= 0:
    lons_mod = (lons + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
    # Reindex the DataArray along the 'longitude' axis in the same order:
    t2m_da = t2m_da.isel(longitude=order)
else:
    # Already in −180..+180, nothing to do
    pass

# ─────────────────────────────────────────────────────────────────────────────
#  3.  PICK “the first time step” (e.g. Jan 1983 if your matched times start there)
#     and convert from K → °C.  This is a (lat × lon) array.
# ─────────────────────────────────────────────────────────────────────────────
t2m_first = t2m_da.isel(valid_time=0).values   # shape: (latitude, longitude)
t2m_first = t2m_first - 273.15                  # now in °C

# ─────────────────────────────────────────────────────────────────────────────
#  4.  DEFINE bounding‐boxes (approximate) for Chile, France, and contiguous US.
#     (If you prefer the true polygons, you can plug in your own shapefile; 
#      here we just box‐approximate them to avoid any shapefile import errors.)
# ─────────────────────────────────────────────────────────────────────────────
# ––  Mainland Chile (approx. lon: [–75, –66], lat: [–56, –17])
mainland_chile = box(-75.0, -56.0, -66.0, -17.0)

# ––  Mainland France (approx. lon: [–5, +8], lat: [42, 51])
mainland_france = box(-5.0, 42.0, 8.0, 51.0)

# ––  Contiguous USA (lon: [–125, –66], lat: [24, 50])
contiguous_us = box(-125.0, 24.0, -66.0, 50.0)

# ─────────────────────────────────────────────────────────────────────────────
#  5.  BUILD THE AFFINE TRANSFORM for rasterio.rasterize:
#     assumes uniform grid spacing.
# ─────────────────────────────────────────────────────────────────────────────
dx = abs(lons[1] - lons[0])
dy = abs(lats[1] - lats[0])

west  = float(np.min(lons))
north = float(np.max(lats))
transform_grid = from_origin(west, north, dx, dy)

# ─────────────────────────────────────────────────────────────────────────────
#  6.  RASTERIZE each country’s bounding‐box at the global resolution:
# ─────────────────────────────────────────────────────────────────────────────
mask_chile = features.rasterize(
    [(mainland_chile, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)

mask_france = features.rasterize(
    [(mainland_france, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)

mask_usa = features.rasterize(
    [(contiguous_us, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)

# ─────────────────────────────────────────────────────────────────────────────
#  7.  APPLY EACH MASK to the t2m‐in‐°C array ⇒ NaN outside the country:
# ─────────────────────────────────────────────────────────────────────────────
t2m_chile  = np.where(mask_chile  == 1, t2m_first, np.nan)
t2m_france = np.where(mask_france == 1, t2m_first, np.nan)
t2m_usa    = np.where(mask_usa    == 1, t2m_first, np.nan)

# ─────────────────────────────────────────────────────────────────────────────
#  8.  FIND a common vmin/vmax so that all three overlays use the same color scale:
# ─────────────────────────────────────────────────────────────────────────────
all_vals = np.concatenate([
    t2m_chile[np.isfinite(t2m_chile)].ravel(),
    t2m_france[np.isfinite(t2m_france)].ravel(),
    t2m_usa[np.isfinite(t2m_usa)].ravel()
])
vmin = np.nanmin(all_vals)
vmax = np.nanmax(all_vals)

# ─────────────────────────────────────────────────────────────────────────────
#  9.  PLOT EVERYTHING:
#     – a global (–180..+180, –90..+90) basemap of the three temperature fields
#     – thick black boxes for Chile, France, contiguous USA
#     – a single colorbar (°C) shared by all overlays
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

# Prepare lon/lat meshgrid only once:
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Plot each country’s t2m field (°C) with “coolwarm”:
ax.pcolormesh(
    lon_grid, lat_grid, t2m_chile,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)
ax.pcolormesh(
    lon_grid, lat_grid, t2m_france,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)
ax.pcolormesh(
    lon_grid, lat_grid, t2m_usa,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)

# Draw the black box “borders” for each country:
for country_box in [mainland_chile, mainland_france, contiguous_us]:
    x, y = country_box.exterior.xy
    ax.plot(x, y, color='black', linewidth=1.0)

# Add one colorbar for everything:
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax)),
                    ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('2 m Temperature (°C)')

# Set global x/y limits, labels, title:
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.set_title(
    'World Map with First Time Step 2 m Temperature\n'
    'for Chile, France, and Contiguous USA (Bounding‐Box Approximation)'
)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from rasterio import features
from rasterio.transform import from_origin
from mpl_toolkits.basemap import Basemap

# ─────────────────────────────────────────────────────────────────────────────
# 1.  OPEN THE MATCHED‐TIMES FILE (“t2m_matched_times.nc”) LAZILY
# ─────────────────────────────────────────────────────────────────────────────
ds = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')   # ← update path as needed
t2m_da = ds['t2m']  # this is in Kelvin; we’ll subtract 273.15 for °C below

# Extract coords
lons = ds['longitude'].values.copy()
lats = ds['latitude'].values

# ─────────────────────────────────────────────────────────────────────────────
# 2.  IF LONGITUDE RUNS 0…360, SHIFT IT TO −180…+180 AND REORDER THE DATAARRAY
# ─────────────────────────────────────────────────────────────────────────────
if lons.min() >= 0:
    lons_mod = (lons + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
    t2m_da = t2m_da.isel(longitude=order)
# else: assume lons are already −180..+180

# Build a meshgrid once for pcolormesh
lon_grid, lat_grid = np.meshgrid(lons, lats)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DEFINE (APPROXIMATE) BOUNDING BOXES FOR CHILE, FRANCE, AND CONTIGUOUS USA
# ─────────────────────────────────────────────────────────────────────────────
mainland_chile = box(-75.0, -56.0, -66.0, -17.0)
mainland_france = box(-5.0, 42.0, 8.0, 51.0)
contiguous_us = box(-125.0, 24.0, -66.0, 50.0)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  RASTERIZE THOSE BOXES ONTO OUR GRID ONCE (so we can reuse the masks)
# ─────────────────────────────────────────────────────────────────────────────
dx = abs(lons[1] - lons[0])
dy = abs(lats[1] - lats[0])
west  = float(np.min(lons))
north = float(np.max(lats))
transform_grid = from_origin(west, north, dx, dy)

mask_chile = features.rasterize(
    [(mainland_chile, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)
mask_france = features.rasterize(
    [(mainland_france, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)
mask_usa = features.rasterize(
    [(contiguous_us, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  STREAM THROUGH ALL TIME STEPS TO FIND A COMMON vmin/vmax (°C) FOR PLOTTING
#    • We load one slice at a time (thus never filling RAM with the full 3D array)
# ─────────────────────────────────────────────────────────────────────────────
vmin = np.inf
vmax = -np.inf
times = t2m_da['valid_time'].values  # e.g. ['1983-01-01', '1983-02-01', …]

for t_idx in range(len(times)):
    # Load the tth slice in Kelvin → convert to °C
    arr_K = t2m_da.isel(valid_time=t_idx).values
    arr_C = arr_K - 273.15

    # Mask out everything except our three boxes
    a_chile = np.where(mask_chile == 1, arr_C, np.nan)
    a_fr   = np.where(mask_france == 1, arr_C, np.nan)
    a_usa  = np.where(mask_usa == 1, arr_C, np.nan)

    # Compute that slice's finite min/max and update running vmin/vmax
    combined = np.concatenate([
        a_chile[np.isfinite(a_chile)].ravel(),
        a_fr[np.isfinite(a_fr)].ravel(),
        a_usa[np.isfinite(a_usa)].ravel()
    ])
    if combined.size:
        vmin = min(vmin, np.nanmin(combined))
        vmax = max(vmax, np.nanmax(combined))

# If for some reason no finite values were found (unlikely), fall back:
if not np.isfinite(vmin) or not np.isfinite(vmax):
    vmin, vmax = 0.0, 1.0

# ─────────────────────────────────────────────────────────────────────────────
# 6.  NOW PLOT EACH TIME STEP IN ITS OWN FIGURE
#    • Basemap 'cyl' coastlines + country borders
#    • pcolormesh of each country‐masked array (°C), all sharing (vmin,vmax)
#    • Black box outlines for Chile / France / USA
# ─────────────────────────────────────────────────────────────────────────────
for t_idx, dt in enumerate(times):
    arr_K = t2m_da.isel(valid_time=t_idx).values
    arr_C = arr_K - 273.15

    a_chile = np.where(mask_chile == 1, arr_C, np.nan)
    a_fr   = np.where(mask_france == 1, arr_C, np.nan)
    a_usa  = np.where(mask_usa == 1, arr_C, np.nan)

    date_str = np.datetime_as_string(dt, unit='D')

    fig, ax = plt.subplots(figsize=(10, 5))
    m = Basemap(
        projection='cyl',
        llcrnrlat=-90, urcrnrlat=90,
        llcrnrlon=-180, urcrnrlon=180,
        resolution='c',
        ax=ax
    )
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)

    # Plot each country’s masked field
    m.pcolormesh(lon_grid, lat_grid, a_chile, cmap='coolwarm',
                 shading='auto', vmin=vmin, vmax=vmax)
    m.pcolormesh(lon_grid, lat_grid, a_fr,   cmap='coolwarm',
                 shading='auto', vmin=vmin, vmax=vmax)
    m.pcolormesh(lon_grid, lat_grid, a_usa,  cmap='coolwarm',
                 shading='auto', vmin=vmin, vmax=vmax)

    # Outline the bounding boxes in black
    for box_geom in [mainland_chile, mainland_france, contiguous_us]:
        x, y = box_geom.exterior.xy
        m.plot(x, y, color='black', linewidth=1.0)

    cbar = m.colorbar(location='right', pad='2%')
    cbar.set_label('2 m Temperature (°C)')

    ax.set_title(f'2 m Temperature on {date_str}\n(Chile, France, Contiguous USA)')
    plt.tight_layout()
    plt.show()


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# 1. Load the dataset
ds = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')
t2m_da = ds['t2m']

# 2. Adjust longitudes (0–360 to –180..180) and reorder if needed
lons = ds['longitude'].values.copy()
lats = ds['latitude'].values.copy()
if lons.min() >= 0:
    lons_mod = (lons + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
    t2m_da = t2m_da.isel(longitude=order)

# 3. Extract the first time step and convert from Kelvin to °C
t2m_first = t2m_da.isel(valid_time=0).values - 273.15  # shape: (lat, lon)

# 4. Create a lon/lat meshgrid
lon_grid, lat_grid = np.meshgrid(lons, lats)

# 5. Define bounding‐box masks for Chile, France, and contiguous USA
mask_chile = (lon_grid >= -75) & (lon_grid <= -66) & (lat_grid >= -56) & (lat_grid <= -17)
mask_france = (lon_grid >= -5) & (lon_grid <= 8) & (lat_grid >= 42) & (lat_grid <= 51)
mask_usa = (lon_grid >= -125) & (lon_grid <= -66) & (lat_grid >= 24) & (lat_grid <= 50)

# 6. Apply each mask to the temperature array (NaN outside the country)
t2m_chile = np.where(mask_chile, t2m_first, np.nan)
t2m_france = np.where(mask_france, t2m_first, np.nan)
t2m_usa = np.where(mask_usa, t2m_first, np.nan)

# 7. Compute a common vmin/vmax for consistent color scaling
all_vals = np.concatenate([
    t2m_chile[np.isfinite(t2m_chile)].ravel(),
    t2m_france[np.isfinite(t2m_france)].ravel(),
    t2m_usa[np.isfinite(t2m_usa)].ravel()
])
vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

# 8. Plot the world map with each country’s temperature overlay
fig, ax = plt.subplots(figsize=(12, 6))
cmap = 'coolwarm'

# Plot each country’s masked temperature field
ax.pcolormesh(lon_grid, lat_grid, t2m_chile, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
ax.pcolormesh(lon_grid, lat_grid, t2m_france, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
ax.pcolormesh(lon_grid, lat_grid, t2m_usa, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)

# Draw thick black bounding‐box borders for each country
# Chile
x_chile = [-75, -66, -66, -75, -75]
y_chile = [-56, -56, -17, -17, -56]
ax.plot(x_chile, y_chile, color='black', linewidth=1.0)

# France
x_france = [-5, 8, 8, -5, -5]
y_france = [42, 42, 51, 51, 42]
ax.plot(x_france, y_france, color='black', linewidth=1.0)

# USA
x_usa = [-125, -66, -66, -125, -125]
y_usa = [24, 24, 50, 50, 24]
ax.plot(x_usa, y_usa, color='black', linewidth=1.0)

# Add a single colorbar for all overlays
norm = Normalize(vmin=vmin, vmax=vmax)
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('2 m Temperature (°C)')

# Set global map limits and labels
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.set_title(
    'World Map with First Time Step 2 m Temperature\n'
    'for Chile, France, and Contiguous USA (Bounding‐Box Approximation)'
)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# 1. Load the dataset
ds = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')
t2m_da = ds['t2m']

# 2. Adjust longitudes (0–360 to –180..180) and reorder if needed
lons = ds['longitude'].values.copy()
lats = ds['latitude'].values.copy()
if lons.min() >= 0:
    lons_mod = (lons + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
    t2m_da = t2m_da.isel(longitude=order)

# 3. Extract the first time step and convert from Kelvin to °C
t2m_first = t2m_da.isel(valid_time=0).values - 273.15  # shape: (lat, lon)

# 4. Create a lon/lat meshgrid
lon_grid, lat_grid = np.meshgrid(lons, lats)

# 5. Define bounding‐box masks for Chile, France, and contiguous USA
mask_chile = (lon_grid >= -75) & (lon_grid <= -66) & (lat_grid >= -56) & (lat_grid <= -17)
mask_france = (lon_grid >= -5) & (lon_grid <= 8) & (lat_grid >= 42) & (lat_grid <= 51)
mask_usa = (lon_grid >= -125) & (lon_grid <= -66) & (lat_grid >= 24) & (lat_grid <= 50)

# 6. Apply each mask to the temperature array (NaN outside the country)
t2m_chile = np.where(mask_chile, t2m_first, np.nan)
t2m_france = np.where(mask_france, t2m_first, np.nan)
t2m_usa = np.where(mask_usa, t2m_first, np.nan)

# 7. Compute a common vmin/vmax for consistent color scaling
all_vals = np.concatenate([
    t2m_chile[np.isfinite(t2m_chile)].ravel(),
    t2m_france[np.isfinite(t2m_france)].ravel(),
    t2m_usa[np.isfinite(t2m_usa)].ravel()
])
vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

# 8. Plot the world map with each country’s temperature overlay (no bounding boxes)
fig, ax = plt.subplots(figsize=(12, 6))
cmap = 'coolwarm'

# Plot each country’s masked temperature field
ax.pcolormesh(lon_grid, lat_grid, t2m_chile, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
ax.pcolormesh(lon_grid, lat_grid, t2m_france, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
ax.pcolormesh(lon_grid, lat_grid, t2m_usa, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)

# Add a single colorbar for all overlays
norm = Normalize(vmin=vmin, vmax=vmax)
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('2 m Temperature (°C)')

# Set global map limits and labels
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.set_title(
    'World Map with First Time Step 2 m Temperature\n'
    'for Chile, France, and Contiguous USA (No Bounding Boxes)'
)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from rasterio import features
from rasterio.transform import from_origin
from mpl_toolkits.basemap import Basemap

# ─────────────────────────────────────────────────────────────────────────────
# 1.  OPEN THE MATCHED‐TIMES FILE (“t2m_matched_times.nc”) LAZILY
# ─────────────────────────────────────────────────────────────────────────────
ds = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')
t2m_da = ds['t2m']  # in Kelvin; will convert to °C on the fly

# Extract longitude and latitude arrays
lons = ds['longitude'].values.copy()
lats = ds['latitude'].values

# ─────────────────────────────────────────────────────────────────────────────
# 2.  SHIFT LONGITUDE IF NEEDED (0..360 → -180..180) AND REORDER
# ─────────────────────────────────────────────────────────────────────────────
if lons.min() >= 0:
    lons_mod = (lons + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
    t2m_da = t2m_da.isel(longitude=order)

# Build meshgrid for pcolormesh
lon_grid, lat_grid = np.meshgrid(lons, lats)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DEFINE (APPROXIMATE) BOXES FOR CHILE, FRANCE, AND USA (for masking only)
# ─────────────────────────────────────────────────────────────────────────────
mainland_chile = box(-75.0, -56.0, -66.0, -17.0)
mainland_france = box(-5.0, 42.0, 8.0, 51.0)
contiguous_us = box(-125.0, 24.0, -66.0, 50.0)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  RASTERIZE MASKS ONCE
# ─────────────────────────────────────────────────────────────────────────────
dx = abs(lons[1] - lons[0])
dy = abs(lats[1] - lats[0])
west = float(np.min(lons))
north = float(np.max(lats))
transform_grid = from_origin(west, north, dx, dy)

mask_chile = features.rasterize(
    [(mainland_chile, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)
mask_france = features.rasterize(
    [(mainland_france, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)
mask_usa = features.rasterize(
    [(contiguous_us, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMPUTE COMMON VMIN/VMAX (°C) ACROSS ALL TIMES AND COUNTRIES
# ─────────────────────────────────────────────────────────────────────────────
vmin = np.inf
vmax = -np.inf
times = t2m_da['valid_time'].values

for t_idx in range(len(times)):
    arr_K = t2m_da.isel(valid_time=t_idx).values
    arr_C = arr_K - 273.15
    
    a_chile = np.where(mask_chile == 1, arr_C, np.nan)
    a_fr = np.where(mask_france == 1, arr_C, np.nan)
    a_usa = np.where(mask_usa == 1, arr_C, np.nan)
    
    combined = np.concatenate([
        a_chile[np.isfinite(a_chile)].ravel(),
        a_fr[np.isfinite(a_fr)].ravel(),
        a_usa[np.isfinite(a_usa)].ravel()
    ])
    if combined.size:
        vmin = min(vmin, np.nanmin(combined))
        vmax = max(vmax, np.nanmax(combined))

if not np.isfinite(vmin) or not np.isfinite(vmax):
    vmin, vmax = 0.0, 1.0  # fallback

# ─────────────────────────────────────────────────────────────────────────────
# 6.  PLOT EACH TIME STEP: WORLD BOUNDARIES + COUNTRY BOUNDARIES + MASKED T2M
# ─────────────────────────────────────────────────────────────────────────────
for t_idx, dt in enumerate(times):
    arr_K = t2m_da.isel(valid_time=t_idx).values
    arr_C = arr_K - 273.15
    
    a_chile = np.where(mask_chile == 1, arr_C, np.nan)
    a_fr = np.where(mask_france == 1, arr_C, np.nan)
    a_usa = np.where(mask_usa == 1, arr_C, np.nan)
    date_str = np.datetime_as_string(dt, unit='D')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    m = Basemap(
        projection='cyl',
        llcrnrlat=-90, urcrnrlat=90,
        llcrnrlon=-180, urcrnrlon=180,
        resolution='c', ax=ax
    )
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    
    # Plot t2m for each country (masked fields) on world map
    m.pcolormesh(lon_grid, lat_grid, a_chile,
                 cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
    m.pcolormesh(lon_grid, lat_grid, a_fr,
                 cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
    m.pcolormesh(lon_grid, lat_grid, a_usa,
                 cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = m.colorbar(location='right', pad='2%')
    cbar.set_label('2 m Temperature (°C)')
    
    ax.set_title(f'2 m Temperature on {date_str}\n(Chile, France, Contiguous USA)')
    plt.tight_layout()
    plt.show()


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from rasterio import features
from rasterio.transform import from_origin
from mpl_toolkits.basemap import Basemap

# Load the subsetted t2m data lazily
ds = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')
t2m_da = ds['t2m']  # in Kelvin

# Extract longitude and latitude arrays
lons = ds['longitude'].values.copy()
lats = ds['latitude'].values

# Handle longitudes 0–360 if needed
if lons.min() >= 0:
    lons_mod = (lons + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
    t2m_da = t2m_da.isel(longitude=order)

# Build meshgrid for plotting
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Define bounding-box masks for each country (for masking only)
mainland_chile = box(-75.0, -56.0, -66.0, -17.0)
mainland_france = box(-5.0, 42.0, 8.0, 51.0)
contiguous_us = box(-125.0, 24.0, -66.0, 50.0)

# Compute affine transform for rasterizing
dx = abs(lons[1] - lons[0])
dy = abs(lats[1] - lats[0])
west = float(np.min(lons))
north = float(np.max(lats))
transform_grid = from_origin(west, north, dx, dy)

# Rasterize the bounding-box masks
mask_chile = features.rasterize(
    [(mainland_chile, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)
mask_france = features.rasterize(
    [(mainland_france, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)
mask_usa = features.rasterize(
    [(contiguous_us, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)

# Determine common vmin/vmax across all times for consistent color scale
vmin = np.inf
vmax = -np.inf
times = t2m_da['valid_time'].values

for t_idx in range(len(times)):
    arr_C = t2m_da.isel(valid_time=t_idx).values - 273.15
    a_chile = np.where(mask_chile == 1, arr_C, np.nan)
    a_fr = np.where(mask_france == 1, arr_C, np.nan)
    a_usa = np.where(mask_usa == 1, arr_C, np.nan)
    combined = np.concatenate([
        a_chile[np.isfinite(a_chile)].ravel(),
        a_fr[np.isfinite(a_fr)].ravel(),
        a_usa[np.isfinite(a_usa)].ravel()
    ])
    if combined.size:
        vmin = min(vmin, np.nanmin(combined))
        vmax = max(vmax, np.nanmax(combined))

if not np.isfinite(vmin) or not np.isfinite(vmax):
    vmin, vmax = 0.0, 1.0

# Plot each time step: world coastline only, no boxes drawn
for t_idx, dt in enumerate(times):
    arr_C = t2m_da.isel(valid_time=t_idx).values - 273.15
    a_chile = np.where(mask_chile == 1, arr_C, np.nan)
    a_fr = np.where(mask_france == 1, arr_C, np.nan)
    a_usa = np.where(mask_usa == 1, arr_C, np.nan)
    date_str = np.datetime_as_string(dt, unit='D')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    m = Basemap(
        projection='cyl',
        llcrnrlat=-90, urcrnrlat=90,
        llcrnrlon=-180, urcrnrlon=180,
        resolution='c', ax=ax
    )
    m.drawcoastlines(linewidth=0.5)
    
    # Plot temperature overlays for countries without drawing their boxes
    m.pcolormesh(lon_grid, lat_grid, a_chile, cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
    m.pcolormesh(lon_grid, lat_grid, a_fr,   cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
    m.pcolormesh(lon_grid, lat_grid, a_usa,  cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
    
    cbar = m.colorbar(location='right', pad='2%')
    cbar.set_label('2 m Temperature (°C)')
    
    ax.set_title(f'2 m Temperature on {date_str}\n(Chile, France, USA)')
    plt.tight_layout()
    plt.show()


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box
from rasterio import features
from rasterio.transform import from_origin

# ─────────────────────────────────────────────────────────────────────────────
# 1.  OPEN t2m_matched_times.nc LAZILY
# ─────────────────────────────────────────────────────────────────────────────
ds = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')  # ← adjust path if needed
t2m_da = ds['t2m']  # in Kelvin

# Extract coords (1D arrays)
lons = ds['longitude'].values.copy()
lats = ds['latitude'].values

# ─────────────────────────────────────────────────────────────────────────────
# 2.  SHIFT 0–360 LONGS ⇒ (–180..+180) IF NECESSARY, REORDER THE DATAARRAY
# ─────────────────────────────────────────────────────────────────────────────
if lons.min() >= 0:
    lons_mod = (lons + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
    t2m_da = t2m_da.isel(longitude=order)

# Build a meshgrid for pcolormesh
lon_grid, lat_grid = np.meshgrid(lons, lats)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DEFINE (APPROXIMATE) BOUNDING BOXES FOR THREE COUNTRIES
# ─────────────────────────────────────────────────────────────────────────────
#    (Use real polygons if you have a shapefile; these boxes are just a quick mask.)
mainland_chile  = box(-75.0, -56.0, -66.0, -17.0)
mainland_france = box(-5.0, 42.0,   8.0, 51.0)
contiguous_us   = box(-125.0, 24.0, -66.0, 50.0)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  RASTERIZE EACH BOX ONTO OUR GRID (lat × lon) SO WE CAN MASK t2m VALUES
# ─────────────────────────────────────────────────────────────────────────────
dx = abs(lons[1] - lons[0])
dy = abs(lats[1] - lats[0])
west  = float(np.min(lons))
north = float(np.max(lats))
transform_grid = from_origin(west, north, dx, dy)

mask_chile = features.rasterize(
    [(mainland_chile, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)
mask_france = features.rasterize(
    [(mainland_france, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)
mask_usa = features.rasterize(
    [(contiguous_us, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMPUTE A COMMON vmin/vmax (°C) BY STREAMING THROUGH ALL TIME STEPS
# ─────────────────────────────────────────────────────────────────────────────
vmin = np.inf
vmax = -np.inf
times = t2m_da['valid_time'].values  # e.g. ['1983-01-01', '1983-02-01', …]

for t_idx in range(len(times)):
    arr_K = t2m_da.isel(valid_time=t_idx).values
    arr_C = arr_K - 273.15  # convert K → °C

    a_chile = np.where(mask_chile == 1, arr_C, np.nan)
    a_fr    = np.where(mask_france == 1, arr_C, np.nan)
    a_usa   = np.where(mask_usa == 1, arr_C, np.nan)

    combined = np.concatenate([
        a_chile[np.isfinite(a_chile)].ravel(),
        a_fr[np.isfinite(a_fr)].ravel(),
        a_usa[np.isfinite(a_usa)].ravel()
    ])
    if combined.size:
        vmin = min(vmin, np.nanmin(combined))
        vmax = max(vmax, np.nanmax(combined))

if not np.isfinite(vmin) or not np.isfinite(vmax):
    vmin, vmax = 0.0, 1.0  # fallback in case no finite data found

# ─────────────────────────────────────────────────────────────────────────────
# 6.  LOAD WORLD OUTLINES FROM GeoPANDAS (naturalearth_lowres)
# ─────────────────────────────────────────────────────────────────────────────
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# We'll use this just for the coastline/country borders—and then overlay t2m.
# ─────────────────────────────────────────────────────────────────────────────
for t_idx, dt in enumerate(times):
    arr_K = t2m_da.isel(valid_time=t_idx).values
    arr_C = arr_K - 273.15

    a_chile = np.where(mask_chile == 1, arr_C, np.nan)
    a_fr    = np.where(mask_france == 1, arr_C, np.nan)
    a_usa   = np.where(mask_usa == 1, arr_C, np.nan)

    date_str = np.datetime_as_string(dt, unit='D')

    # ─────────────────────────────────────────────────────────────────────────
    # Create a GeoPandas plot of all landmasses/coastlines:
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    world.boundary.plot(ax=ax, color='lightgray', linewidth=0.5)

    # ─────────────────────────────────────────────────────────────────────────
    # Overlay each country's masked t2m (°C) via pcolormesh:
    # ─────────────────────────────────────────────────────────────────────────
    pcm_ch = ax.pcolormesh(
        lon_grid, lat_grid, a_chile,
        cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
    )
    pcm_fr = ax.pcolormesh(
        lon_grid, lat_grid, a_fr,
        cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
    )
    pcm_us = ax.pcolormesh(
        lon_grid, lat_grid, a_usa,
        cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Adjust plot limits, add colorbar, title, labels:
    # ─────────────────────────────────────────────────────────────────────────
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title(f'2 m Temperature on {date_str}\n(Chile, France, USA)')

    # Add one colorbar on the right:
    cbar = fig.colorbar(pcm_us, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('2 m Temperature (°C)')

    plt.tight_layout()
    plt.show()


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import box
from rasterio import features
from rasterio.transform import from_origin

# ─────────────────────────────────────────────────────────────────────────────
# 1.  OPEN “t2m_matched_times.nc” LAZILY
# ─────────────────────────────────────────────────────────────────────────────
ds = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')  # ← adjust this path
t2m_da = ds['t2m']  # in Kelvin

# Extract 1D lon/lat
lons = ds['longitude'].values.copy()
lats = ds['latitude'].values

# ─────────────────────────────────────────────────────────────────────────────
# 2.  SHIFT LONGITUDES 0..360 → (–180..+180) IF NECESSARY, REORDER t2m_da
# ─────────────────────────────────────────────────────────────────────────────
if lons.min() >= 0:
    lons_mod = (lons + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
    t2m_da = t2m_da.isel(longitude=order)
# else: already in (–180..+180)

lon_grid, lat_grid = np.meshgrid(lons, lats)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DEFINE BOUNDING BOXES (approximate) FOR Chile, France, USA
# ─────────────────────────────────────────────────────────────────────────────
mainland_chile  = box(-75.0, -56.0, -66.0, -17.0)
mainland_france = box(-5.0,   42.0,   8.0, 51.0)
contiguous_us   = box(-125.0, 24.0, -66.0, 50.0)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  RASTERIZE THOSE BOXES ONTO THE (lat × lon) GRID ONCE
# ─────────────────────────────────────────────────────────────────────────────
dx = abs(lons[1] - lons[0])
dy = abs(lats[1] - lats[0])
west  = float(np.min(lons))
north = float(np.max(lats))
transform_grid = from_origin(west, north, dx, dy)

mask_chile = features.rasterize(
    [(mainland_chile, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)
mask_france = features.rasterize(
    [(mainland_france, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)
mask_usa = features.rasterize(
    [(contiguous_us, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FIND A COMMON vmin/vmax (°C) BY STREAMING THROUGH EACH TIME SLICE
# ─────────────────────────────────────────────────────────────────────────────
vmin = np.inf
vmax = -np.inf
times = t2m_da['valid_time'].values  # e.g. ['1983-01-01', '1983-02-01', …]

for t_idx in range(len(times)):
    arr_K = t2m_da.isel(valid_time=t_idx).values
    arr_C = arr_K - 273.15

    # Mask each country region; anything outside becomes NaN
    a_chile = np.where(mask_chile == 1, arr_C, np.nan)
    a_fr    = np.where(mask_france == 1, arr_C, np.nan)
    a_usa   = np.where(mask_usa == 1, arr_C, np.nan)

    combined = np.concatenate([
        a_chile[np.isfinite(a_chile)].ravel(),
        a_fr[np.isfinite(a_fr)].ravel(),
        a_usa[np.isfinite(a_usa)].ravel()
    ])
    if combined.size:
        vmin = min(vmin, np.nanmin(combined))
        vmax = max(vmax, np.nanmax(combined))

# Fallback if no finite data found (unlikely)
if not np.isfinite(vmin) or not np.isfinite(vmax):
    vmin, vmax = -50.0, 50.0

# ─────────────────────────────────────────────────────────────────────────────
# 6.  LOOP OVER EACH TIME STEP → PLOT WITH CARTOPY
#     • Draw full‐world coastlines & borders in light gray
#     • Overlay only the masked t2m for Chile, France, USA (no boxes drawn)
# ─────────────────────────────────────────────────────────────────────────────
for t_idx, dt in enumerate(times):
    arr_K = t2m_da.isel(valid_time=t_idx).values
    arr_C = arr_K - 273.15

    a_chile = np.where(mask_chile == 1, arr_C, np.nan)
    a_fr    = np.where(mask_france == 1, arr_C, np.nan)
    a_usa   = np.where(mask_usa == 1, arr_C, np.nan)

    date_str = np.datetime_as_string(dt, unit='D')  # e.g. '1983-01-01'

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Draw coastlines & country borders in light gray
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.5, edgecolor='lightgray')
    ax.add_feature(cfeature.BORDERS.with_scale('110m'), linewidth=0.5, edgecolor='lightgray')

    # Set global extent
    ax.set_global()

    # Plot each country's masked t2m‐in‐°C via pcolormesh
    pcm1 = ax.pcolormesh(
        lon_grid, lat_grid, a_chile,
        transform=ccrs.PlateCarree(),
        cmap='coolwarm',
        shading='auto',
        vmin=vmin, vmax=vmax
    )
    pcm2 = ax.pcolormesh(
        lon_grid, lat_grid, a_fr,
        transform=ccrs.PlateCarree(),
        cmap='coolwarm',
        shading='auto',
        vmin=vmin, vmax=vmax
    )
    pcm3 = ax.pcolormesh(
        lon_grid, lat_grid, a_usa,
        transform=ccrs.PlateCarree(),
        cmap='coolwarm',
        shading='auto',
        vmin=vmin, vmax=vmax
    )

    # Add a colorbar on the right side
    cbar = plt.colorbar(pcm3, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
    cbar.set_label('2 m Temperature (°C)')

    ax.set_title(f'2 m Temperature on {date_str}\n(Chile, France, USA)')
    plt.tight_layout()
    plt.show()


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from rasterio import features
from rasterio.transform import from_origin

# Attempt to import Cartopy; if unavailable, fall back to Basemap
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    use_cartopy = True
except ImportError:
    from mpl_toolkits.basemap import Basemap
    use_cartopy = False

# ─────────────────────────────────────────────────────────────────────────────
# 1. OPEN “t2m_matched_times.nc” AND EXTRACT THE 't2m' DataArray
# ─────────────────────────────────────────────────────────────────────────────
ds = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')
t2m_da = ds['t2m']  # in Kelvin

# Extract longitude and latitude arrays
lons = ds['longitude'].values.copy()
lats = ds['latitude'].values

# ─────────────────────────────────────────────────────────────────────────────
# 2. SHIFT LONGITUDES 0–360 TO –180..+180 IF NECESSARY, AND REORDER t2m_da
# ─────────────────────────────────────────────────────────────────────────────
if lons.min() >= 0:
    lons_mod = (lons + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
    t2m_da = t2m_da.isel(longitude=order)

# Build meshgrid for plotting
lon_grid, lat_grid = np.meshgrid(lons, lats)

# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPUTE THE TIME‐MEAN IN DEGREES CELSIUS
# ─────────────────────────────────────────────────────────────────────────────
mean_t2m_C = (t2m_da - 273.15).mean(dim='valid_time').values  # 2D array (lat, lon)

# ─────────────────────────────────────────────────────────────────────────────
# 4. DEFINE BOUNDING BOXES FOR CHILE, FRANCE, AND CONTIGUOUS USA
# ─────────────────────────────────────────────────────────────────────────────
mainland_chile  = box(-75.0,  -56.0,  -66.0,  -17.0)
mainland_france = box(-5.0,    42.0,    8.0,    51.0)
contiguous_us   = box(-125.0,   24.0,   -66.0,   50.0)

# Rasterize these boxes onto the (lat × lon) grid
dx = abs(lons[1] - lons[0])
dy = abs(lats[1] - lats[0])
west  = float(np.min(lons))
north = float(np.max(lats))
transform_grid = from_origin(west, north, dx, dy)

mask_chile = features.rasterize(
    [(mainland_chile, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)
mask_france = features.rasterize(
    [(mainland_france, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)
mask_usa = features.rasterize(
    [(contiguous_us, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. MASK THE MEAN FIELD FOR EACH COUNTRY (NaN OUTSIDE)
# ─────────────────────────────────────────────────────────────────────────────
mean_chile  = np.where(mask_chile == 1,  mean_t2m_C, np.nan)
mean_france = np.where(mask_france == 1, mean_t2m_C, np.nan)
mean_usa    = np.where(mask_usa == 1,    mean_t2m_C, np.nan)

# Compute vmin/vmax across the three masked arrays
combined_vals = np.concatenate([
    mean_chile[np.isfinite(mean_chile)].ravel(),
    mean_france[np.isfinite(mean_france)].ravel(),
    mean_usa[np.isfinite(mean_usa)].ravel()
])
if combined_vals.size:
    vmin = np.nanmin(combined_vals)
    vmax = np.nanmax(combined_vals)
else:
    vmin, vmax = 0.0, 1.0

# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOT AVERAGED T2M ON A GLOBAL MAP (COASTLINES + BORDERS)
# ─────────────────────────────────────────────────────────────────────────────
if use_cartopy:
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Draw coastlines and country borders in light gray
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'),
                   linewidth=0.5, edgecolor='lightgray')
    ax.add_feature(cfeature.BORDERS.with_scale('110m'),
                   linewidth=0.5, edgecolor='lightgray')

    ax.set_global()

    # Overlay the masked mean‐temperature fields
    ax.pcolormesh(lon_grid, lat_grid, mean_chile,
                  transform=ccrs.PlateCarree(),
                  cmap='coolwarm', shading='auto',
                  vmin=vmin, vmax=vmax)
    ax.pcolormesh(lon_grid, lat_grid, mean_france,
                  transform=ccrs.PlateCarree(),
                  cmap='coolwarm', shading='auto',
                  vmin=vmin, vmax=vmax)
    ax.pcolormesh(lon_grid, lat_grid, mean_usa,
                  transform=ccrs.PlateCarree(),
                  cmap='coolwarm', shading='auto',
                  vmin=vmin, vmax=vmax)

    # Colorbar
    cbar = plt.colorbar(ax.pcolormesh(lon_grid, lat_grid, mean_usa,
                                      transform=ccrs.PlateCarree(),
                                      cmap='coolwarm',
                                      vmin=vmin, vmax=vmax),
                        ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
    cbar.set_label('Mean 2 m Temperature (°C)')

    ax.set_title('Mean 2 m Temperature\n(Chile, France, Contiguous USA) Over Matching Nino Times')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    plt.tight_layout()
    plt.show()

else:
    fig, ax = plt.subplots(figsize=(12, 6))
    m = Basemap(projection='cyl',
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180,
                resolution='c', ax=ax)
    m.drawcoastlines(linewidth=0.5, color='lightgray')
    m.drawcountries(linewidth=0.5, color='lightgray')

    # Overlay the masked mean‐temperature fields
    # Basemap's pcolormesh can map lon/lat directly if latlon=True
    m.pcolormesh(lon_grid, lat_grid, mean_chile,
                 cmap='coolwarm', shading='auto',
                 vmin=vmin, vmax=vmax, latlon=True)
    m.pcolormesh(lon_grid, lat_grid, mean_france,
                 cmap='coolwarm', shading='auto',
                 vmin=vmin, vmax=vmax, latlon=True)
    m.pcolormesh(lon_grid, lat_grid, mean_usa,
                 cmap='coolwarm', shading='auto',
                 vmin=vmin, vmax=vmax, latlon=True)

    # Colorbar
    cbar = m.colorbar(location='right', pad='2%')
    cbar.set_label('Mean 2 m Temperature (°C)')

    ax.set_title('Mean 2 m Temperature\n(Chile, France, Contiguous USA) Over Matching Nino Times')
    plt.tight_layout()
    plt.show()


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from rasterio import features
from rasterio.transform import from_origin

# Attempt to import Cartopy; if unavailable, fall back to Basemap
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    use_cartopy = True
except ImportError:
    from mpl_toolkits.basemap import Basemap
    use_cartopy = False

# ─────────────────────────────────────────────────────────────────────────────
# 1. OPEN “t2m_matched_times.nc” AND EXTRACT THE 't2m' DATAARRAY
# ─────────────────────────────────────────────────────────────────────────────
ds = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')
t2m_da = ds['t2m']  # Kelvin

# Extract lon/lat
lons = ds['longitude'].values.copy()
lats = ds['latitude'].values

# ─────────────────────────────────────────────────────────────────────────────
# 2. SHIFT LONGITUDES 0–360 TO –180..+180 IF NECESSARY AND REORDER t2m_da
# ─────────────────────────────────────────────────────────────────────────────
if lons.min() >= 0:
    lons_mod = (lons + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
    t2m_da = t2m_da.isel(longitude=order)

# Build meshgrid
lon_grid, lat_grid = np.meshgrid(lons, lats)

# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPUTE THE TIME‐MEAN IN °C
# ─────────────────────────────────────────────────────────────────────────────
mean_t2m_C = (t2m_da - 273.15).mean(dim='valid_time').values  # 2D array (lat, lon)

# ─────────────────────────────────────────────────────────────────────────────
# 4. DEFINE BOUNDING BOXES FOR THE REQUESTED COUNTRIES
#    – Colombia, Chile, Brazil, Indonesia, Philippines
# ─────────────────────────────────────────────────────────────────────────────
forest_boxes = {
    'Colombia':    box(-79.0,  -4.0,  -66.0,  13.0),
    'Chile':       box(-75.0, -56.0,  -66.0, -17.0),  # same as before
    'Brazil':      box(-74.0, -34.0,  -34.0,   5.0),
    'Indonesia':   box( 95.0, -11.0,  141.0,   6.0),
    'Philippines': box(116.0,   5.0,  127.0,  19.0)
}

# Rasterize each box onto the lat×lon grid
dx = abs(lons[1] - lons[0])
dy = abs(lats[1] - lats[0])
west  = float(np.min(lons))
north = float(np.max(lats))
transform_grid = from_origin(west, north, dx, dy)

masks = {}
for country, bbox in forest_boxes.items():
    masks[country] = features.rasterize(
        [(bbox, 1)],
        out_shape=(len(lats), len(lons)),
        transform=transform_grid,
        fill=0,
        dtype=np.uint8
    )

# ─────────────────────────────────────────────────────────────────────────────
# 5. MASK THE MEAN FIELD FOR EACH COUNTRY (NaN OUTSIDE)
# ─────────────────────────────────────────────────────────────────────────────
masked_means = {}
for country, mask in masks.items():
    masked_means[country] = np.where(mask == 1, mean_t2m_C, np.nan)

# Compute vmin/vmax across all masked countries
all_vals = np.concatenate([
    masked_means[country][np.isfinite(masked_means[country])].ravel()
    for country in masked_means
])
if all_vals.size:
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)
else:
    vmin, vmax = 0.0, 1.0

# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOT ON A WORLD MAP (with Cartopy if available, else Basemap)
# ─────────────────────────────────────────────────────────────────────────────
if use_cartopy:
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Draw coastlines and borders
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.5, edgecolor='lightgray')
    ax.add_feature(cfeature.BORDERS.with_scale('110m'), linewidth=0.5, edgecolor='lightgray')
    ax.set_global()

    # Overlay each country's masked mean temperature
    for country, data in masked_means.items():
        ax.pcolormesh(
            lon_grid, lat_grid, data,
            transform=ccrs.PlateCarree(),
            cmap='coolwarm', shading='auto',
            vmin=vmin, vmax=vmax
        )

    # Add a colorbar
    pcm = ax.pcolormesh(lon_grid, lat_grid, list(masked_means.values())[0],
                        transform=ccrs.PlateCarree(),
                        cmap='coolwarm', shading='auto',
                        vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
    cbar.set_label('Mean 2 m Temperature (°C)')

    ax.set_title('Mean 2 m Temperature\n(Colombia, Chile, Brazil, Indonesia, Philippines)')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    plt.tight_layout()
    plt.show()

else:
    fig, ax = plt.subplots(figsize=(12, 6))
    m = Basemap(projection='cyl',
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180,
                resolution='c', ax=ax)
    m.drawcoastlines(linewidth=0.5, color='lightgray')
    m.drawcountries(linewidth=0.5, color='lightgray')

    # Overlay each country's masked mean temperature
    for data in masked_means.values():
        m.pcolormesh(lon_grid, lat_grid, data,
                     cmap='coolwarm', shading='auto',
                     vmin=vmin, vmax=vmax, latlon=True)

    # Add a colorbar
    pcm = m.pcolormesh(lon_grid, lat_grid, list(masked_means.values())[0],
                       cmap='coolwarm', shading='auto',
                       vmin=vmin, vmax=vmax, latlon=True)
    cbar = m.colorbar(location='right', pad='2%')
    cbar.set_label('Mean 2 m Temperature (°C)')

    ax.set_title('Mean 2 m Temperature\n(Colombia, Chile, Brazil, Indonesia, Philippines)')
    plt.tight_layout()
    plt.show()


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from rasterio import features
from rasterio.transform import from_origin

# Attempt to import Cartopy; if unavailable, fall back to Basemap
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    use_cartopy = True
except ImportError:
    from mpl_toolkits.basemap import Basemap
    use_cartopy = False

# ─────────────────────────────────────────────────────────────────────────────
# 1. OPEN “t2m_matched_times.nc” AND EXTRACT 't2m' DataArray
# ─────────────────────────────────────────────────────────────────────────────
ds = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')
t2m_da = ds['t2m']  # in Kelvin

# Extract longitude and latitude arrays
lons = ds['longitude'].values.copy()
lats = ds['latitude'].values

# ─────────────────────────────────────────────────────────────────────────────
# 2. SHIFT LONGITUDES 0–360 → −180..+180 IF NECESSARY, AND REORDER t2m_da
# ─────────────────────────────────────────────────────────────────────────────
if lons.min() >= 0:
    lons_mod = (lons + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
    t2m_da = t2m_da.isel(longitude=order)

# Build meshgrid for plotting
lon_grid, lat_grid = np.meshgrid(lons, lats)

# ─────────────────────────────────────────────────────────────────────────────
# 3. DEFINE BOUNDING BOXES FOR FOUR COUNTRIES
# ─────────────────────────────────────────────────────────────────────────────
# Colombia: lon [-79, -66], lat [-4, 13]
colombia_box = box(-79.0, -4.0, -66.0, 13.0)

# Chile: lon [-75, -66], lat [-56, -17]
chile_box = box(-75.0, -56.0, -66.0, -17.0)

# Indonesia: lon [95, 141], lat [-11, 6]
indonesia_box = box(95.0, -11.0, 141.0, 6.0)

# Philippines: lon [116, 126], lat [5, 20]
philippines_box = box(116.0, 5.0, 126.0, 20.0)

# ─────────────────────────────────────────────────────────────────────────────
# 4. RASTERIZE BOUNDING BOXES ONTO THE (lat × lon) GRID TO CREATE MASKS
# ─────────────────────────────────────────────────────────────────────────────
dx = abs(lons[1] - lons[0])
dy = abs(lats[1] - lats[0])
west  = float(np.min(lons))
north = float(np.max(lats))
transform_grid = from_origin(west, north, dx, dy)

mask_colombia = features.rasterize(
    [(colombia_box, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)
mask_chile = features.rasterize(
    [(chile_box, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)
mask_indonesia = features.rasterize(
    [(indonesia_box, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)
mask_philippines = features.rasterize(
    [(philippines_box, 1)],
    out_shape=(len(lats), len(lons)),
    transform=transform_grid,
    fill=0,
    dtype=np.uint8
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. STREAM THROUGH TIME STEPS AND COMPUTE SUMS FOR EACH COUNTRY
#    (one slice at a time to avoid MemoryError)
# ─────────────────────────────────────────────────────────────────────────────
nlat = len(lats)
nlon = len(lons)
ntimes = t2m_da.sizes['valid_time']

# Initialize sum arrays
sum_colombia = np.zeros((nlat, nlon), dtype=float)
sum_chile = np.zeros((nlat, nlon), dtype=float)
sum_indonesia = np.zeros((nlat, nlon), dtype=float)
sum_philippines = np.zeros((nlat, nlon), dtype=float)

# Loop over each time index
for t_idx in range(ntimes):
    arr_K = t2m_da.isel(valid_time=t_idx).values      # one 2D slice in Kelvin
    arr_C = arr_K - 273.15                             # convert to °C

    # Accumulate sums inside each mask
    sum_colombia    += np.where(mask_colombia    == 1, arr_C, 0.0)
    sum_chile       += np.where(mask_chile       == 1, arr_C, 0.0)
    sum_indonesia   += np.where(mask_indonesia   == 1, arr_C, 0.0)
    sum_philippines += np.where(mask_philippines == 1, arr_C, 0.0)

# Compute means by dividing by ntimes, then mask outside
mean_colombia    = np.where(mask_colombia    == 1, sum_colombia / ntimes,    np.nan)
mean_chile       = np.where(mask_chile       == 1, sum_chile / ntimes,       np.nan)
mean_indonesia   = np.where(mask_indonesia   == 1, sum_indonesia / ntimes,   np.nan)
mean_philippines = np.where(mask_philippines == 1, sum_philippines / ntimes, np.nan)

# ─────────────────────────────────────────────────────────────────────────────
# 6. DETERMINE COLOR SCALE LIMITS ACROSS THE FOUR MEAN ARRAYS
# ─────────────────────────────────────────────────────────────────────────────
all_vals = np.concatenate([
    mean_colombia[np.isfinite(mean_colombia)].ravel(),
    mean_chile[np.isfinite(mean_chile)].ravel(),
    mean_indonesia[np.isfinite(mean_indonesia)].ravel(),
    mean_philippines[np.isfinite(mean_philippines)].ravel()
])
if all_vals.size:
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)
else:
    vmin, vmax = 0.0, 1.0

# ─────────────────────────────────────────────────────────────────────────────
# 7. PLOT THE FOUR COUNTRY MEANS ON A WORLD MAP
#    (Cartopy if available; otherwise Basemap)
# ─────────────────────────────────────────────────────────────────────────────
if use_cartopy:
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Draw coastlines and borders in light gray
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'),
                   linewidth=0.5, edgecolor='lightgray')
    ax.add_feature(cfeature.BORDERS.with_scale('110m'),
                   linewidth=0.5, edgecolor='lightgray')
    ax.set_global()

    # Overlay each country’s mean‐t2m field
    ax.pcolormesh(
        lon_grid, lat_grid, mean_colombia,
        transform=ccrs.PlateCarree(), cmap='coolwarm',
        shading='auto', vmin=vmin, vmax=vmax
    )
    ax.pcolormesh(
        lon_grid, lat_grid, mean_chile,
        transform=ccrs.PlateCarree(), cmap='coolwarm',
        shading='auto', vmin=vmin, vmax=vmax
    )
    ax.pcolormesh(
        lon_grid, lat_grid, mean_indonesia,
        transform=ccrs.PlateCarree(), cmap='coolwarm',
        shading='auto', vmin=vmin, vmax=vmax
    )
    ax.pcolormesh(
        lon_grid, lat_grid, mean_philippines,
        transform=ccrs.PlateCarree(), cmap='coolwarm',
        shading='auto', vmin=vmin, vmax=vmax
    )

    # Colorbar on the right
    cbar = plt.colorbar(ax.pcolormesh(
        lon_grid, lat_grid, mean_chile,
        transform=ccrs.PlateCarree(), cmap='coolwarm',
        shading='auto', vmin=vmin, vmax=vmax
    ), ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
    cbar.set_label('Mean 2 m Temperature (°C)')

    ax.set_title('Mean 2 m Temperature\n(Colombia, Chile, Indonesia, Philippines)')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    plt.tight_layout()
    plt.show()

else:
    fig, ax = plt.subplots(figsize=(12, 6))
    m = Basemap(
        projection='cyl',
        llcrnrlat=-90, urcrnrlat=90,
        llcrnrlon=-180, urcrnrlon=180,
        resolution='c', ax=ax
    )
    m.drawcoastlines(linewidth=0.5, color='lightgray')
    m.drawcountries(linewidth=0.5, color='lightgray')

    # Overlay each country’s mean‐t2m field
    m.pcolormesh(
        lon_grid, lat_grid, mean_colombia,
        cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax, latlon=True
    )
    m.pcolormesh(
        lon_grid, lat_grid, mean_chile,
        cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax, latlon=True
    )
    m.pcolormesh(
        lon_grid, lat_grid, mean_indonesia,
        cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax, latlon=True
    )
    m.pcolormesh(
        lon_grid, lat_grid, mean_philippines,
        cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax, latlon=True
    )

    # Colorbar on the right
    cbar = m.colorbar(location='right', pad='2%')
    cbar.set_label('Mean 2 m Temperature (°C)')

    ax.set_title('Mean 2 m Temperature\n(Colombia, Chile, Indonesia, Philippines)')
    plt.tight_layout()
    plt.show()


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio import features
from rasterio.transform import from_origin
import fiona
from shapely.geometry import shape, Polygon, MultiPolygon, box
from shapely.ops import transform

# --------------------------------------------------------------------------------
# 1. Load the “matched‐times” t2m file (with times already aligned to your nino.nc)
# --------------------------------------------------------------------------------
ds_matched = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')
# The variable name should still be “t2m” and “valid_time” is aligned to nino.nc
print(ds_matched)
# e.g.: 
# <xarray.Dataset>
# Dimensions:     (valid_time: 5, latitude: 1801, longitude: 3600)
# Coordinates:
#   * valid_time  (valid_time) datetime64[ns] 1983-01-01 1983-02-01 ... 1983-05-01
#   * latitude    (latitude) float64  90.0 89.9 89.8 ... -89.9 -90.0
#   * longitude   (longitude) float64   0.0 0.1 0.2 ... 359.7 359.8 359.9
# Data variables:
#     t2m         (valid_time, latitude, longitude) float32 ...
#
# We will loop over each `valid_time` in ds_matched.

# --------------------------------------------------------------------------------
# 2. Load country shapefile and extract geometries (same as before)
# --------------------------------------------------------------------------------
shapefile_path = '/Users/marie-audepradal/Documents/WB_countries_Admin0_10m.shp'
with fiona.open(shapefile_path) as shp:
    chile_geom = None
    france_geom = None
    usa_geom = None
    world_geoms = []
    for feature in shp:
        props = feature['properties']
        geom = shape(feature['geometry'])
        world_geoms.append(geom)
        name = props.get('NAME_EN') or props.get('WB_NAME')
        if name == 'Chile':
            chile_geom = geom
        elif name == 'France':
            france_geom = geom
        elif name in ('United States of America', 'United States'):
            usa_geom = geom

if chile_geom is None:
    raise ValueError("Chile geometry not found in shapefile.")
if france_geom is None:
    raise ValueError("France geometry not found in shapefile.")
if usa_geom is None:
    raise ValueError("USA geometry not found in shapefile.")

# Optionally clip Chile to mainland, and USA to contiguous (lower 48)
mainland_bbox_cl = box(-75.0, -56.0, -66.0, -17.0)
mainland_chile = chile_geom.intersection(mainland_bbox_cl)

contiguous_bbox_us = box(-125.0, 24.0, -66.0, 50.0)
contiguous_us = usa_geom.intersection(contiguous_bbox_us)

# --------------------------------------------------------------------------------
# 3. Prepare a single (lon, lat) grid & transform once
# --------------------------------------------------------------------------------
# Extract the first time slice just to get lon/lat arrays; they don't change over time
lons_full = ds_matched['longitude'].values.copy()
lats = ds_matched['latitude'].values

# If lon runs 0–360, convert to –180–180 and reorder
if lons_full.min() >= 0:
    lons_mod = (lons_full + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
else:
    lons = lons_full.copy()
    order = np.arange(len(lons))

# Compute grid resolution (assumes uniform spacing)
dx = float(lons[1] - lons[0])
dy = float(abs(lats[1] - lats[0]))

# Build affine transform (west, north) origin
west = float(lons.min())
north = float(lats.max())
transform_grid = from_origin(west, north, dx, dy)

# Rasterize country polygons to masks once—same size each time
shape_out = (len(lats), len(lons))
mask_chile = features.rasterize(
    [(mainland_chile, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)
mask_france = features.rasterize(
    [(france_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)
mask_usa = features.rasterize(
    [(contiguous_us, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)

# Precompute a meshgrid of lon/lat for pcolormesh
lon_grid, lat_grid = np.meshgrid(lons, lats)

# --------------------------------------------------------------------------------
# 4. Loop over each valid_time in ds_matched and plot
# --------------------------------------------------------------------------------
for ti, timestamp in enumerate(ds_matched['valid_time'].values):
    # 4.1 Select the t2m slice at this time, reorder longitudes if needed, convert K→°C
    t2m_slice = ds_matched['t2m'].isel(valid_time=ti).values  # shape: (lat, lon_original)
    if lons_full.min() >= 0:
        # reorder columns from original 0–360 to –180–180
        t2m_reordered = t2m_slice[:, order]
    else:
        t2m_reordered = t2m_slice.copy()
    t2m_celsius = t2m_reordered - 273.15

    # 4.2 Apply masks to isolate each country
    t2m_chile = np.where(mask_chile == 1, t2m_celsius, np.nan)
    t2m_france = np.where(mask_france == 1, t2m_celsius, np.nan)
    t2m_usa = np.where(mask_usa == 1, t2m_celsius, np.nan)

    # 4.3 Compute vmin/vmax across all three for consistent colorbar
    all_vals = np.concatenate([
        t2m_chile[np.isfinite(t2m_chile)].ravel(),
        t2m_france[np.isfinite(t2m_france)].ravel(),
        t2m_usa[np.isfinite(t2m_usa)].ravel()
    ])
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)

    # 4.4 Begin plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # 4.4.1 Plot all world boundaries in light gray
    for geom in world_geoms:
        if isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color='lightgray', linewidth=0.5)
        elif isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            ax.plot(x, y, color='lightgray', linewidth=0.5)

    # 4.4.2 Plot t2m for each country as a pcolormesh
    pcm_chile = ax.pcolormesh(
        lon_grid, lat_grid, t2m_chile,
        cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
    )
    pcm_france = ax.pcolormesh(
        lon_grid, lat_grid, t2m_france,
        cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
    )
    pcm_usa = ax.pcolormesh(
        lon_grid, lat_grid, t2m_usa,
        cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
    )

    # 4.4.3 Overlay thick black boundaries for Chile, France, USA
    def plot_boundary(geom, color='black', lw=1.0):
        if isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color=color, linewidth=lw)
        elif isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            ax.plot(x, y, color=color, linewidth=lw)

    plot_boundary(mainland_chile, lw=1.0)
    plot_boundary(france_geom, lw=1.0)
    plot_boundary(contiguous_us, lw=1.0)

    # 4.4.4 Colorbar
    cbar = plt.colorbar(pcm_usa, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('2 m Temperature (°C)')

    # 4.4.5 Set global extent & labels
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title(f'World Map with {np.datetime_as_string(timestamp, unit="D")} 2 m Temperature\n'
                 'for Chile, France, and Contiguous USA')

    plt.tight_layout()
    plt.show()


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio import features
from rasterio.transform import from_origin
import fiona
from shapely.geometry import shape, Polygon, MultiPolygon, box

# --------------------------------------------------------------------------------
# 1. Load the “matched‐times” t2m file (with times already aligned to your nino.nc)
# --------------------------------------------------------------------------------
ds_matched = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')
# The variable name should still be “t2m” and “valid_time” is aligned to nino.nc
print(ds_matched)
# e.g.: 
# <xarray.Dataset>
# Dimensions:     (valid_time: 5, latitude: 1801, longitude: 3600)
# Coordinates:
#   * valid_time  (valid_time) datetime64[ns] 1983-01-01 1983-02-01 ... 1983-05-01
#   * latitude    (latitude) float64  90.0 89.9 89.8 ... -89.9 -90.0
#   * longitude   (longitude) float64   0.0 0.1 0.2 ... 359.7 359.8 359.9
# Data variables:
#     t2m         (valid_time, latitude, longitude) float32 ...
#
# We will loop over each `valid_time` in ds_matched.

# --------------------------------------------------------------------------------
# 2. Load country shapefile and extract geometries for six countries
# --------------------------------------------------------------------------------
shapefile_path = '/Users/marie-audepradal/Documents/WB_countries_Admin0_10m.shp'
with fiona.open(shapefile_path) as shp:
    mexico_geom = None
    colombia_geom = None
    chile_geom = None
    indonesia_geom = None
    south_africa_geom = None
    brazil_geom = None
    world_geoms = []   # for plotting all country outlines in light gray
    for feature in shp:
        props = feature['properties']
        geom = shape(feature['geometry'])
        world_geoms.append(geom)
        name = props.get('NAME_EN') or props.get('WB_NAME')
        if name == 'Mexico':
            mexico_geom = geom
        elif name == 'Colombia':
            colombia_geom = geom
        elif name == 'Chile':
            chile_geom = geom
        elif name == 'Indonesia':
            indonesia_geom = geom
        elif name in ('South Africa', 'Republic of South Africa'):
            south_africa_geom = geom
        elif name == 'Brazil':
            brazil_geom = geom

# Sanity checks
for country_name, country_geom in [
    ('Mexico', mexico_geom),
    ('Colombia', colombia_geom),
    ('Chile', chile_geom),
    ('Indonesia', indonesia_geom),
    ('South Africa', south_africa_geom),
    ('Brazil', brazil_geom),
]:
    if country_geom is None:
        raise ValueError(f"{country_name} geometry not found in shapefile.")

# (Optional) If you want to clip any large trans‐continental countries to a bounding box,
# you could define boxes here. For example, if you only care about mainland Chile:
# mainland_bbox_cl = box(-75.0, -56.0, -66.0, -17.0)
# mainland_chile = chile_geom.intersection(mainland_bbox_cl)
# Then use mainland_chile instead of chile_geom below.
#
# In this example, we’ll just use the full polygons as they appear in the shapefile.

# --------------------------------------------------------------------------------
# 3. Prepare a single (lon, lat) grid & affine transform once
# --------------------------------------------------------------------------------
lons_full = ds_matched['longitude'].values.copy()
lats = ds_matched['latitude'].values

# Convert 0–360 → -180–180 if needed, and get sorting index
if lons_full.min() >= 0:
    lons_mod = (lons_full + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
else:
    lons = lons_full.copy()
    order = np.arange(len(lons))

# Grid spacing (assuming uniform)
dx = float(lons[1] - lons[0])
dy = float(abs(lats[1] - lats[0]))

# Build affine transform (west, north) origin
west = float(lons.min())
north = float(lats.max())
transform_grid = from_origin(west, north, dx, dy)

# Raster shape = (n_lat, n_lon)
shape_out = (len(lats), len(lons))

# --------------------------------------------------------------------------------
# 4. Rasterize each country polygon into a mask (0/1)
# --------------------------------------------------------------------------------
mask_mexico = features.rasterize(
    [(mexico_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)
mask_colombia = features.rasterize(
    [(colombia_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)
mask_chile = features.rasterize(
    [(chile_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)
mask_indonesia = features.rasterize(
    [(indonesia_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)
mask_south_africa = features.rasterize(
    [(south_africa_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)
mask_brazil = features.rasterize(
    [(brazil_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)

# Pre‐compute a meshgrid of lon/lat for pcolormesh
lon_grid, lat_grid = np.meshgrid(lons, lats)

# --------------------------------------------------------------------------------
# 5. Loop over each valid_time in ds_matched and plot all six countries together
# --------------------------------------------------------------------------------
for ti, timestamp in enumerate(ds_matched['valid_time'].values):
    # 5.1 Extract t2m slice, reorder longitudes if needed, convert K→°C
    t2m_slice = ds_matched['t2m'].isel(valid_time=ti).values  # (lat, lon_original)
    if lons_full.min() >= 0:
        t2m_reordered = t2m_slice[:, order]
    else:
        t2m_reordered = t2m_slice.copy()
    t2m_celsius = t2m_reordered - 273.15

    # 5.2 Apply masks to isolate each country's temperature field
    t2m_mexico = np.where(mask_mexico == 1, t2m_celsius, np.nan)
    t2m_colombia = np.where(mask_colombia == 1, t2m_celsius, np.nan)
    t2m_chile = np.where(mask_chile == 1, t2m_celsius, np.nan)
    t2m_indonesia = np.where(mask_indonesia == 1, t2m_celsius, np.nan)
    t2m_south_africa = np.where(mask_south_africa == 1, t2m_celsius, np.nan)
    t2m_brazil = np.where(mask_brazil == 1, t2m_celsius, np.nan)

    # 5.3 Compute common vmin/vmax over all six
    all_vals = np.concatenate([
        t2m_mexico[np.isfinite(t2m_mexico)].ravel(),
        t2m_colombia[np.isfinite(t2m_colombia)].ravel(),
        t2m_chile[np.isfinite(t2m_chile)].ravel(),
        t2m_indonesia[np.isfinite(t2m_indonesia)].ravel(),
        t2m_south_africa[np.isfinite(t2m_south_africa)].ravel(),
        t2m_brazil[np.isfinite(t2m_brazil)].ravel(),
    ])
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)

    # 5.4 Begin plotting
    fig, ax = plt.subplots(figsize=(14, 8))

    # 5.4.1 Plot all world boundaries in light gray
    for geom in world_geoms:
        if isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color='lightgray', linewidth=0.5)
        elif isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            ax.plot(x, y, color='lightgray', linewidth=0.5)

    # 5.4.2 Plot t2m for each country with pcolormesh
    pcm_mexico = ax.pcolormesh(
        lon_grid, lat_grid, t2m_mexico,
        cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
    )
    pcm_colombia = ax.pcolormesh(
        lon_grid, lat_grid, t2m_colombia,
        cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
    )
    pcm_chile = ax.pcolormesh(
        lon_grid, lat_grid, t2m_chile,
        cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
    )
    pcm_indonesia = ax.pcolormesh(
        lon_grid, lat_grid, t2m_indonesia,
        cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
    )
    pcm_south_africa = ax.pcolormesh(
        lon_grid, lat_grid, t2m_south_africa,
        cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
    )
    pcm_brazil = ax.pcolormesh(
        lon_grid, lat_grid, t2m_brazil,
        cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
    )

    # 5.4.3 Overlay thick black boundaries for each of the six
    def plot_boundary(geom, color='black', lw=1.0):
        if isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color=color, linewidth=lw)
        elif isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            ax.plot(x, y, color=color, linewidth=lw)

    plot_boundary(mexico_geom, lw=1.2)
    plot_boundary(colombia_geom, lw=1.2)
    plot_boundary(chile_geom, lw=1.2)
    plot_boundary(indonesia_geom, lw=1.2)
    plot_boundary(south_africa_geom, lw=1.2)
    plot_boundary(brazil_geom, lw=1.2)

    # 5.4.4 Colorbar (with one of the colormesh objects—values are the same range)
    cbar = plt.colorbar(pcm_brazil, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('2 m Temperature (°C)')

    # 5.4.5 Set global extent & labels
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title(
        f'World Map with {np.datetime_as_string(timestamp, unit="D")} 2 m Temperature\n'
        'for Mexico, Colombia, Chile, Indonesia, South Africa, and Brazil'
    )

    plt.tight_layout()
    plt.show()


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio import features
from rasterio.transform import from_origin
import fiona
from shapely.geometry import shape, Polygon, MultiPolygon, box

# --------------------------------------------------------------------------------
# 1. Load the “matched‐times” t2m file and compute time‐average
# --------------------------------------------------------------------------------
ds_matched = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')
# Compute the mean over all valid_time steps (this yields a 2D field: latitude × longitude)
t2m_mean = ds_matched['t2m'].mean(dim='valid_time')  # still in Kelvin

# For reference, print out dimensions:
print(t2m_mean)
# e.g.:
# <xarray.DataArray 't2m' (latitude: 1801, longitude: 3600)>
# array([...], dtype=float32)
# Coordinates:
#   * latitude   (latitude) float64  90.0 89.9 89.8 ... -89.9 -90.0
#   * longitude  (longitude) float64   0.0 0.1 0.2 ... 359.7 359.8 359.9

# --------------------------------------------------------------------------------
# 2. Load country shapefile and extract geometries
# --------------------------------------------------------------------------------
shapefile_path = '/Users/marie-audepradal/Documents/WB_countries_Admin0_10m.shp'
with fiona.open(shapefile_path) as shp:
    mexico_geom = None
    colombia_geom = None
    chile_geom = None
    indonesia_geom = None
    south_africa_geom = None
    brazil_geom = None
    world_geoms = []   # to plot all country outlines faintly in the background
    for feature in shp:
        props = feature['properties']
        geom = shape(feature['geometry'])
        world_geoms.append(geom)
        name = props.get('NAME_EN') or props.get('WB_NAME')
        if name == 'Mexico':
            mexico_geom = geom
        elif name == 'Colombia':
            colombia_geom = geom
        elif name == 'Chile':
            chile_geom = geom
        elif name == 'Indonesia':
            indonesia_geom = geom
        elif name in ('South Africa', 'Republic of South Africa'):
            south_africa_geom = geom
        elif name == 'Brazil':
            brazil_geom = geom

# Verify that all six were found
for nm, geom in [
    ('Mexico', mexico_geom),
    ('Colombia', colombia_geom),
    ('Chile', chile_geom),
    ('Indonesia', indonesia_geom),
    ('South Africa', south_africa_geom),
    ('Brazil', brazil_geom)
]:
    if geom is None:
        raise ValueError(f"{nm} geometry not found in shapefile.")

# --------------------------------------------------------------------------------
# 3. Prepare (lon, lat) grid & affine transform once
# --------------------------------------------------------------------------------
lons_full = ds_matched['longitude'].values.copy()  # e.g. 0…359.9
lats = ds_matched['latitude'].values              # e.g. 90.0…-90.0

# Convert from 0–360 → –180–180 if needed
if lons_full.min() >= 0:
    lons_mod = (lons_full + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
else:
    lons = lons_full.copy()
    order = np.arange(len(lons))

# Compute horizontal resolution (assumes uniform grid spacing)
dx = float(lons[1] - lons[0])
dy = float(abs(lats[1] - lats[0]))

# Affine transform: origin in the upper‐left (west, north)
west = float(lons.min())
north = float(lats.max())
transform_grid = from_origin(west, north, dx, dy)

# Output raster shape = (#lat, #lon)
shape_out = (len(lats), len(lons))

# --------------------------------------------------------------------------------
# 4. Rasterize each country polygon into a 0/1 mask
# --------------------------------------------------------------------------------
mask_mexico = features.rasterize(
    [(mexico_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)
mask_colombia = features.rasterize(
    [(colombia_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)
mask_chile = features.rasterize(
    [(chile_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)
mask_indonesia = features.rasterize(
    [(indonesia_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)
mask_south_africa = features.rasterize(
    [(south_africa_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)
mask_brazil = features.rasterize(
    [(brazil_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)

# Build meshgrid for plotting
lon_grid, lat_grid = np.meshgrid(lons, lats)

# --------------------------------------------------------------------------------
# 5. Reorder + convert the time‐mean field from K → °C, then apply masks
# --------------------------------------------------------------------------------
# Convert DataArray → NumPy array; reorder longitudes if necessary
t2m_mean_arr = t2m_mean.values  # shape: (nlat, nlon_full)
if lons_full.min() >= 0:
    t2m_mean_ordered = t2m_mean_arr[:, order]
else:
    t2m_mean_ordered = t2m_mean_arr.copy()

# Kelvin → Celsius
t2m_mean_c = t2m_mean_ordered - 273.15  # still shape = (nlat, nlon)

# Apply each country’s mask (NaN outside)
t2m_mexico = np.where(mask_mexico == 1, t2m_mean_c, np.nan)
t2m_colombia = np.where(mask_colombia == 1, t2m_mean_c, np.nan)
t2m_chile = np.where(mask_chile == 1, t2m_mean_c, np.nan)
t2m_indonesia = np.where(mask_indonesia == 1, t2m_mean_c, np.nan)
t2m_south_africa = np.where(mask_south_africa == 1, t2m_mean_c, np.nan)
t2m_brazil = np.where(mask_brazil == 1, t2m_mean_c, np.nan)

# Compute common color scale (vmin/vmax) across all six masked regions
all_vals = np.concatenate([
    t2m_mexico[np.isfinite(t2m_mexico)].ravel(),
    t2m_colombia[np.isfinite(t2m_colombia)].ravel(),
    t2m_chile[np.isfinite(t2m_chile)].ravel(),
    t2m_indonesia[np.isfinite(t2m_indonesia)].ravel(),
    t2m_south_africa[np.isfinite(t2m_south_africa)].ravel(),
    t2m_brazil[np.isfinite(t2m_brazil)].ravel(),
])
vmin = np.nanmin(all_vals)
vmax = np.nanmax(all_vals)

# --------------------------------------------------------------------------------
# 6. Plot a single map with all six countries colored by their time‐mean t2m
# --------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))

# 6.1 Draw all world boundaries faintly in light gray
from shapely.geometry import MultiPolygon
for geom in world_geoms:
    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='lightgray', linewidth=0.5)
    else:
        x, y = geom.exterior.xy
        ax.plot(x, y, color='lightgray', linewidth=0.5)

# 6.2 Plot each country’s averaged t2m via pcolormesh (same colormap & vmin/vmax)
pcm_mexico = ax.pcolormesh(
    lon_grid, lat_grid, t2m_mexico,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)
pcm_colombia = ax.pcolormesh(
    lon_grid, lat_grid, t2m_colombia,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)
pcm_chile = ax.pcolormesh(
    lon_grid, lat_grid, t2m_chile,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)
pcm_indonesia = ax.pcolormesh(
    lon_grid, lat_grid, t2m_indonesia,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)
pcm_south_africa = ax.pcolormesh(
    lon_grid, lat_grid, t2m_south_africa,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)
pcm_brazil = ax.pcolormesh(
    lon_grid, lat_grid, t2m_brazil,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)

# 6.3 Overlay thick black boundaries for each of the six countries
def plot_boundary(geom, lw=1.2, color='black'):
    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color=color, linewidth=lw)
    else:
        x, y = geom.exterior.xy
        ax.plot(x, y, color=color, linewidth=lw)

plot_boundary(mexico_geom)
plot_boundary(colombia_geom)
plot_boundary(chile_geom)
plot_boundary(indonesia_geom)
plot_boundary(south_africa_geom)
plot_boundary(brazil_geom)

# 6.4 Add a colorbar (we can reference any of the pcolormesh handles; values share the same vmin/vmax)
cbar = plt.colorbar(pcm_brazil, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Mean 2 m Temperature (°C)')

# 6.5 Final formatting
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.set_title('Time‐Mean 2 m Temperature (°C)\n' +
             'for Mexico, Colombia, Chile, Indonesia, South Africa, and Brazil')

plt.tight_layout()
plt.show()


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio import features
from rasterio.transform import from_origin
import fiona
from shapely.geometry import shape, Polygon, MultiPolygon, box

# --------------------------------------------------------------------------------
# 1. Load the “matched‐times” t2m file
# --------------------------------------------------------------------------------
ds_matched = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')
# “t2m” is in Kelvin, dims = (valid_time, latitude, longitude)
# “valid_time” are the timestamps aligned to your Niño index.
# Print to check:
print(ds_matched)
# e.g.:
# <xarray.Dataset>
# Dimensions:     (valid_time: N, latitude: 1801, longitude: 3600)
# Coordinates:
#   * valid_time  (valid_time) datetime64[ns] 1983-01-01 1983-02-01 … etc.
#   * latitude    (latitude) float64 90.0 89.9 … -89.9 -90.0
#   * longitude   (longitude) float64 0.0 0.1 … 359.8 359.9
# Data variables:
#     t2m         (valid_time, latitude, longitude) float32 …
# We will loop over each valid_time and subtract climatology.

# --------------------------------------------------------------------------------
# 2. Load the monthly climatology (1970–2024) from the uploaded file
# --------------------------------------------------------------------------------
clim_path = '/Users/marie-audepradal/Documents/temp_climatology_1970-2024_ERA5Land_monthly.nc'
ds_clim = xr.open_dataset(clim_path)
# It has dims “month: 1–12”, “latitude: 1801”, “longitude: 3600”,
# and data variable “t2m_climatology” (in K).
print(ds_clim)
# e.g.:
# <xarray.Dataset>
# Dimensions:          (month: 12, latitude: 1801, longitude: 3600)
# Coordinates:
#     number           int64 8B …      (not used here)
#   * latitude         (latitude) float64  14kB 90.0 89.9 … -89.9 -90.0
#   * longitude        (longitude) float64  29kB 0.0 0.1 … 359.8 359.9
#   * month            (month) int64  96B 1 2 … 12
# Data variables:
#     t2m_climatology  (month, latitude, longitude) float32  311MB …

# --------------------------------------------------------------------------------
# 3. Load country shapefile & extract geometries for six countries
# --------------------------------------------------------------------------------
shapefile_path = '/Users/marie-audepradal/Documents/WB_countries_Admin0_10m.shp'
with fiona.open(shapefile_path) as shp:
    mexico_geom = None
    colombia_geom = None
    chile_geom = None
    indonesia_geom = None
    south_africa_geom = None
    brazil_geom = None
    world_geoms = []

    for feature in shp:
        props = feature['properties']
        geom = shape(feature['geometry'])
        world_geoms.append(geom)
        name = props.get('NAME_EN') or props.get('WB_NAME')

        if name == 'Mexico':
            mexico_geom = geom
        elif name == 'Colombia':
            colombia_geom = geom
        elif name == 'Chile':
            chile_geom = geom
        elif name == 'Indonesia':
            indonesia_geom = geom
        elif name in ('South Africa', 'Republic of South Africa'):
            south_africa_geom = geom
        elif name == 'Brazil':
            brazil_geom = geom

# Ensure all six were found
for nm, geom in [
    ('Mexico', mexico_geom),
    ('Colombia', colombia_geom),
    ('Chile', chile_geom),
    ('Indonesia', indonesia_geom),
    ('South Africa', south_africa_geom),
    ('Brazil', brazil_geom),
]:
    if geom is None:
        raise ValueError(f"{nm} geometry not found in shapefile.")

# --------------------------------------------------------------------------------
# 4. Prepare a common (longitude, latitude) grid & affine transform
# --------------------------------------------------------------------------------
# Extract lon/lat arrays from ds_matched
lons_full = ds_matched['longitude'].values.copy()  # 0.0 … 359.9 (likely)
lats = ds_matched['latitude'].values.copy()       #  90.0 … -90.0

# Convert 0–360 → -180–180 if needed (and keep a “order” index)
if lons_full.min() >= 0:
    lons_mod = (lons_full + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
else:
    lons = lons_full.copy()
    order = np.arange(len(lons))

# Grid resolution (assume uniform spacing)
dx = float(lons[1] - lons[0])
dy = float(abs(lats[1] - lats[0]))

# Build an affine transform for rasterio
west = float(lons.min())
north = float(lats.max())
transform_grid = from_origin(west, north, dx, dy)

# Output raster shape = (n_lat, n_lon)
shape_out = (len(lats), len(lons))

# Build meshgrid once for plotting
lon_grid, lat_grid = np.meshgrid(lons, lats)

# --------------------------------------------------------------------------------
# 5. Rasterize each country into a 0/1 mask
# --------------------------------------------------------------------------------
mask_mexico = features.rasterize(
    [(mexico_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)
mask_colombia = features.rasterize(
    [(colombia_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)
mask_chile = features.rasterize(
    [(chile_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)
mask_indonesia = features.rasterize(
    [(indonesia_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)
mask_south_africa = features.rasterize(
    [(south_africa_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)
mask_brazil = features.rasterize(
    [(brazil_geom, 1)],
    out_shape=shape_out,
    transform=transform_grid,
    fill=0, dtype=np.uint8
)

# --------------------------------------------------------------------------------
# 6. Loop over each valid_time, subtract climatology (same grid), accumulate anomaly
# --------------------------------------------------------------------------------
n_times = ds_matched.dims['valid_time']
# Initialize an array to accumulate “sum of anomaly” in Kelvin (anomaly K = T_matched_K - T_clim_K)
sum_anomaly = np.zeros((len(lats), len(lons)), dtype=np.float64)

for ti, timestamp in enumerate(ds_matched['valid_time'].values):
    # 6.1 Extract the t2m slice (in K) at this time
    t2m_slice = ds_matched['t2m'].isel(valid_time=ti).values  # shape: (n_lat, n_lon_full)

    # 6.2 Reorder from 0–360 → -180–180 if necessary
    if lons_full.min() >= 0:
        t2m_reordered = t2m_slice[:, order]
    else:
        t2m_reordered = t2m_slice.copy()

    # 6.3 Find the calendar month (1–12) of this timestamp
    #    numpy datetime64 → convert to pandas Timestamp for .month
    month = int(np.datetime64(timestamp, 'M').astype('datetime64[M]').astype('datetime64[ns]').astype(str)[5:7])
    # Alternatively: pd.to_datetime(timestamp).month

    # 6.4 Extract the climatology for that month (in K), reorder the lons in the same way
    clim_slice = ds_clim['t2m_climatology'].isel(month=month-1).values  # (n_lat, n_lon_full)
    if lons_full.min() >= 0:
        clim_reordered = clim_slice[:, order]
    else:
        clim_reordered = clim_slice.copy()

    # 6.5 Compute the anomaly (in K).  (T_matched_K – T_clim_K) ⇒ same numeric difference in °C.
    anomaly_this = t2m_reordered - clim_reordered  # shape = (n_lat, n_lon)

    # 6.6 Accumulate
    sum_anomaly += anomaly_this

# 6.7 Divide by number of time steps to get the time‐mean anomaly (in K, which = °C numerically)
mean_anomaly = sum_anomaly / float(n_times)  # shape = (n_lat, n_lon)

# --------------------------------------------------------------------------------
# 7. Apply each country’s mask to that mean anomaly & compute common color scale
# --------------------------------------------------------------------------------
anom_mexico = np.where(mask_mexico == 1, mean_anomaly, np.nan)
anom_colombia = np.where(mask_colombia == 1, mean_anomaly, np.nan)
anom_chile = np.where(mask_chile == 1, mean_anomaly, np.nan)
anom_indonesia = np.where(mask_indonesia == 1, mean_anomaly, np.nan)
anom_south_africa = np.where(mask_south_africa == 1, mean_anomaly, np.nan)
anom_brazil = np.where(mask_brazil == 1, mean_anomaly, np.nan)

# Compute a single vmin/vmax across all six countries (excluding NaNs)
all_vals = np.concatenate([
    anom_mexico[np.isfinite(anom_mexico)].ravel(),
    anom_colombia[np.isfinite(anom_colombia)].ravel(),
    anom_chile[np.isfinite(anom_chile)].ravel(),
    anom_indonesia[np.isfinite(anom_indonesia)].ravel(),
    anom_south_africa[np.isfinite(anom_south_africa)].ravel(),
    anom_brazil[np.isfinite(anom_brazil)].ravel(),
])
vmin = np.nanmin(all_vals)
vmax = np.nanmax(all_vals)

# --------------------------------------------------------------------------------
# 8. Plot a single map of the six countries’ time‐mean anomalies
# --------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))

# 8.1 Draw all world boundaries in light gray
from shapely.geometry import MultiPolygon
for geom in world_geoms:
    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='black', linewidth=1)
    else:
        x, y = geom.exterior.xy
        ax.plot(x, y, color='black', linewidth=1)

# 8.2 Plot each country’s anomaly (°C) via pcolormesh
pcm_mex = ax.pcolormesh(
    lon_grid, lat_grid, anom_mexico,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)
pcm_col = ax.pcolormesh(
    lon_grid, lat_grid, anom_colombia,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)
pcm_chl = ax.pcolormesh(
    lon_grid, lat_grid, anom_chile,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)
pcm_ind = ax.pcolormesh(
    lon_grid, lat_grid, anom_indonesia,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)
pcm_saf = ax.pcolormesh(
    lon_grid, lat_grid, anom_south_africa,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)
pcm_bra = ax.pcolormesh(
    lon_grid, lat_grid, anom_brazil,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)

# 8.3 Overlay thick black outlines for each of the six countries
def plot_boundary(geom, lw=0.1, color='black'):
    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color=color, linewidth=lw)
    else:
        x, y = geom.exterior.xy
        ax.plot(x, y, color=color, linewidth=lw)

plot_boundary(mexico_geom)
plot_boundary(colombia_geom)
plot_boundary(chile_geom)
plot_boundary(indonesia_geom)
plot_boundary(south_africa_geom)
plot_boundary(brazil_geom)

# 8.4 Add a colorbar (values are ΔT in °C)
cbar = plt.colorbar(pcm_bra, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Mean 2 m Temperature Anomaly (°C)')

# 8.5 Final axes formatting
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.set_title('Mean 2 m Temperature Anomaly (°C)\n' +
             'for Mexico, Colombia, Chile, Indonesia, South Africa, Brazil\n' +
             'relative to 1970–2024 monthly climatology')

plt.tight_layout()
plt.show()


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio import features
from rasterio.transform import from_origin
import fiona
from shapely.geometry import shape, Polygon, MultiPolygon

# --------------------------------------------------------------------------------
# 1. Load the “matched‐times” t2m file
# --------------------------------------------------------------------------------
ds_matched = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')
# “t2m” is in Kelvin, dims = (valid_time, latitude, longitude)

# --------------------------------------------------------------------------------
# 2. Load the monthly climatology (1970–2024)
# --------------------------------------------------------------------------------
clim_path = '/Users/marie-audepradal/Documents/temp_climatology_1970-2024_ERA5Land_monthly.nc'
ds_clim = xr.open_dataset(clim_path)
# dims = (month, latitude, longitude), variable = 't2m_climatology' in K

# --------------------------------------------------------------------------------
# 3. Load country shapefile & extract geometries for six countries
# --------------------------------------------------------------------------------
shapefile_path = '/Users/marie-audepradal/Documents/WB_countries_Admin0_10m.shp'
with fiona.open(shapefile_path) as shp:
    mexico_geom = None
    colombia_geom = None
    chile_geom = None
    indonesia_geom = None
    south_africa_geom = None
    brazil_geom = None
    world_geoms = []

    for feature in shp:
        props = feature['properties']
        geom = shape(feature['geometry'])
        world_geoms.append(geom)
        name = props.get('NAME_EN') or props.get('WB_NAME')

        if name == 'Mexico':
            mexico_geom = geom
        elif name == 'Colombia':
            colombia_geom = geom
        elif name == 'Chile':
            chile_geom = geom
        elif name == 'Indonesia':
            indonesia_geom = geom
        elif name in ('South Africa', 'Republic of South Africa'):
            south_africa_geom = geom
        elif name == 'Brazil':
            brazil_geom = geom

for nm, geom in [
    ('Mexico', mexico_geom),
    ('Colombia', colombia_geom),
    ('Chile', chile_geom),
    ('Indonesia', indonesia_geom),
    ('South Africa', south_africa_geom),
    ('Brazil', brazil_geom),
]:
    if geom is None:
        raise ValueError(f"{nm} geometry not found in shapefile.")

# --------------------------------------------------------------------------------
# 4. Prepare (lon, lat) grid & affine transform
# --------------------------------------------------------------------------------
lons_full = ds_matched['longitude'].values.copy()  # probably 0→359.9
lats      = ds_matched['latitude'].values.copy()   #  90→-90

# Convert 0–360 → -180–180 if needed
if lons_full.min() >= 0:
    lons_mod = (lons_full + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
else:
    lons = lons_full.copy()
    order = np.arange(len(lons))

dx = float(lons[1] - lons[0])
dy = float(abs(lats[1] - lats[0]))

west  = float(lons.min())
north = float(lats.max())
transform_grid = from_origin(west, north, dx, dy)

shape_out   = (len(lats), len(lons))
lon_grid, lat_grid = np.meshgrid(lons, lats)

# --------------------------------------------------------------------------------
# 5. Rasterize each country into a 0/1 mask
# --------------------------------------------------------------------------------
mask_mexico       = features.rasterize([(mexico_geom, 1)], out_shape=shape_out,
                                        transform=transform_grid, fill=0, dtype=np.uint8)
mask_colombia     = features.rasterize([(colombia_geom, 1)], out_shape=shape_out,
                                        transform=transform_grid, fill=0, dtype=np.uint8)
mask_chile        = features.rasterize([(chile_geom, 1)], out_shape=shape_out,
                                        transform=transform_grid, fill=0, dtype=np.uint8)
mask_indonesia    = features.rasterize([(indonesia_geom, 1)], out_shape=shape_out,
                                        transform=transform_grid, fill=0, dtype=np.uint8)
mask_south_africa = features.rasterize([(south_africa_geom, 1)], out_shape=shape_out,
                                        transform=transform_grid, fill=0, dtype=np.uint8)
mask_brazil       = features.rasterize([(brazil_geom, 1)], out_shape=shape_out,
                                        transform=transform_grid, fill=0, dtype=np.uint8)

# --------------------------------------------------------------------------------
# 6. Subtract monthly climatology at each time → accumulate anomaly
# --------------------------------------------------------------------------------
n_times = ds_matched.dims['valid_time']
sum_anomaly = np.zeros((len(lats), len(lons)), dtype=np.float64)

for ti, timestamp in enumerate(ds_matched['valid_time'].values):
    t2m_slice = ds_matched['t2m'].isel(valid_time=ti).values  # shape = (nlat, nlon_full)
    # reorder longitudes if needed
    if lons_full.min() >= 0:
        t2m_reordered = t2m_slice[:, order]
    else:
        t2m_reordered = t2m_slice.copy()

    # extract month (1–12)
    month = int(str(np.datetime64(timestamp, 'M'))[5:7])

    # climatology slice (in K)
    clim_slice = ds_clim['t2m_climatology'].isel(month=month - 1).values  # (nlat, nlon_full)
    if lons_full.min() >= 0:
        clim_reordered = clim_slice[:, order]
    else:
        clim_reordered = clim_slice.copy()

    anomaly_this = t2m_reordered - clim_reordered
    sum_anomaly += anomaly_this

mean_anomaly = sum_anomaly / float(n_times)  # shape = (nlat, nlon), in K (≡ °C numerically)

# --------------------------------------------------------------------------------
# 7. Mask each country’s anomaly & find common vmin/vmax
# --------------------------------------------------------------------------------
anom_mexico       = np.where(mask_mexico       == 1, mean_anomaly, np.nan)
anom_colombia     = np.where(mask_colombia     == 1, mean_anomaly, np.nan)
anom_chile        = np.where(mask_chile        == 1, mean_anomaly, np.nan)
anom_indonesia    = np.where(mask_indonesia    == 1, mean_anomaly, np.nan)
anom_south_africa = np.where(mask_south_africa == 1, mean_anomaly, np.nan)
anom_brazil       = np.where(mask_brazil       == 1, mean_anomaly, np.nan)

all_vals = np.concatenate([
    anom_mexico[np.isfinite(anom_mexico)].ravel(),
    anom_colombia[np.isfinite(anom_colombia)].ravel(),
    anom_chile[np.isfinite(anom_chile)].ravel(),
    anom_indonesia[np.isfinite(anom_indonesia)].ravel(),
    anom_south_africa[np.isfinite(anom_south_africa)].ravel(),
    anom_brazil[np.isfinite(anom_brazil)].ravel(),
])
vmin = np.nanmin(all_vals)
vmax = np.nanmax(all_vals)

# --------------------------------------------------------------------------------
# 8. Plot one map, zoomed on South America
# --------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))

# 8.1 Plot all world boundaries faintly
for geom in world_geoms:
    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='lightgray', linewidth=0.5)
    else:
        x, y = geom.exterior.xy
        ax.plot(x, y, color='lightgray', linewidth=0.5)

# 8.2 Plot each country’s anomaly (°C) with the common vmin/vmax
ax.pcolormesh(lon_grid, lat_grid, anom_mexico,
              cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
ax.pcolormesh(lon_grid, lat_grid, anom_colombia,
              cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
ax.pcolormesh(lon_grid, lat_grid, anom_chile,
              cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
ax.pcolormesh(lon_grid, lat_grid, anom_indonesia,
              cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
ax.pcolormesh(lon_grid, lat_grid, anom_south_africa,
              cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
ax.pcolormesh(lon_grid, lat_grid, anom_brazil,
              cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)

# 8.3 Overlay thick black boundaries for each country
def plot_boundary(geom, lw=1.2):
    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='black', linewidth=lw)
    else:
        x, y = geom.exterior.xy
        ax.plot(x, y, color='black', linewidth=lw)

plot_boundary(mexico_geom)
plot_boundary(colombia_geom)
plot_boundary(chile_geom)
plot_boundary(indonesia_geom)
plot_boundary(south_africa_geom)
plot_boundary(brazil_geom)

# 8.4 Colorbar
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax)),
                    ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Mean 2 m Temperature Anomaly (°C)')

# 8.5 ZOOM onto South America:
ax.set_xlim(-90, -30)   # approximate lon‐bounds for South America
ax.set_ylim(-60,  20)   # approximate lat‐bounds for South America

ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.set_title(
    'Mean 2 m Temperature Anomaly (°C)\n'
    'Zoomed on South America\n'
    'for Mexico, Colombia, Chile, Indonesia, South Africa, Brazil'
)

plt.tight_layout()
plt.show()


# %%
# --------------------------------------------------------------------------------
# 8. Plot a zoomed‐in map over South America and save as PDF
# --------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))

# 8.1 Draw all world boundaries in light gray (for context)
from shapely.geometry import MultiPolygon
for geom in world_geoms:
    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='lightgray', linewidth=0.5)
    else:
        x, y = geom.exterior.xy
        ax.plot(x, y, color='lightgray', linewidth=0.5)

# 8.2 Plot each South America‐country’s anomaly (°C) via pcolormesh
pcm_col = ax.pcolormesh(
    lon_grid, lat_grid, anom_colombia,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)
pcm_chl = ax.pcolormesh(
    lon_grid, lat_grid, anom_chile,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)
pcm_bra = ax.pcolormesh(
    lon_grid, lat_grid, anom_brazil,
    cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax
)

# (Optionally: if you want to include Peru, Ecuador, etc., add their masks the same way.)

# 8.3 Overlay thick black outlines for Colombia, Chile, Brazil
def plot_boundary(geom, lw=1.2, color='black'):
    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color=color, linewidth=lw)
    else:
        x, y = geom.exterior.xy
        ax.plot(x, y, color=color, linewidth=lw)

plot_boundary(colombia_geom)
plot_boundary(chile_geom)
plot_boundary(brazil_geom)

# 8.4 Add a colorbar (values are ΔT in °C)
cbar = plt.colorbar(pcm_bra, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Mean 2 m Temperature Anomaly (°C)')

# 8.5 Zoom bounds for South America:
#      longitude: approx. –82° → –34° 
#      latitude:  approx. –56° →  13°
ax.set_xlim(-82, -34)
ax.set_ylim(-56, 13)

# 8.6 Final axes formatting
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.set_title(
    'South America: Mean 2 m Temperature Anomaly (°C)\n'
    'for Colombia, Chile, and Brazil\n'
    'relative to 1970–2024 monthly climatology'
)

plt.tight_layout()

# 8.7 Save to PDF
fig.savefig('south_america_anomaly_map.pdf', format='pdf', dpi=300)

plt.show()


# %%
import numpy as np               
import pandas as pd              
import xarray as xr
import matplotlib.pyplot as plt  
import matplotlib as mpl        
import matplotlib.colors as mcolors  
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from helpers import *

%matplotlib inline                
_ = plt.xkcd()

plt.rcParams['font.family'] = 'Arial'

# Define the column names (Year + 12 months)
columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Read the file into a DataFrame
df = pd.read_csv('/Users/marie-audepradal/Documents/enso_indices.txt', delim_whitespace=True, header=None, names=columns)

# Display the first few rows
print(df.head())

# Convert to long format
df_long = df.melt(id_vars='Year', var_name='Month', value_name='ENSO_Index')

# Convert to datetime for proper x-axis
month_to_num = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
df_long['Month_Num'] = df_long['Month'].map(month_to_num)
df_long['Date'] = pd.to_datetime(dict(year=df_long['Year'], month=df_long['Month_Num'], day=1))

# Sort by date
df_long = df_long.sort_values('Date')
# Plot the continuous time series
plt.figure(figsize=(14, 6))
plt.plot(df_long['Date'], df_long['ENSO_Index'], label='ENSO Index', color='blue')
plt.axhline(0, color='gray', linestyle='--')
plt.title('ENSO Index Over Time')
plt.xlabel('Year')
plt.ylabel('ENSO Index')
plt.grid(True)
plt.tight_layout()
plt.show()

ensoLB=2
ensoUB=2.5


# Conditions for highlighting la Nina extreme events
extreme_mask_nino = (df_long['ENSO_Index'] >= ensoLB) & (df_long['ENSO_Index'] < ensoUB)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(df_long['Date'], df_long['ENSO_Index'], color='gray', label='ENSO Index')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
years = pd.date_range(start='1979', end='2025', freq='YS')
for year in years:
    plt.axvline(x=year, color='gray', linestyle='--', linewidth=0.5)

# Highlight extremes
plt.plot(df_long.loc[extreme_mask_nino, 'Date'],
         df_long.loc[extreme_mask_nino, 'ENSO_Index'],
         color='red', linestyle='none', marker='o', label='Extreme El Nino')

plt.title('ENSO Index Over Time')
plt.xlabel('Date')
plt.ylabel('ENSO Index')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Select extreme values El Nino
extreme_nino_df = df_long[extreme_mask_nino].copy()
extreme_nino_df.set_index('Date', inplace=True)


# Convert to xarray Dataset
ds_nino = xr.Dataset(
    {"enso_index": ("time", extreme_nino_df['ENSO_Index'].values)},
    coords={"time": extreme_nino_df.index}
)


# build the output path using ensoLB and ensoUB values
nc_path_enso_extr = f"/Users/marie-audepradal/Documents/{ensoLB}nino{ensoUB}.nc"
# save to NetCDF
ds_nino.to_netcdf(nc_path_enso_extr)
#ds_nino.to_netcdf("/Users/marie-audepradal/Documents/{ensoLB}_nino_{ensoUB}.nc")
# 2) save as txt file 
# Build the output path using ensoLB
txt_path = f"/Users/marie-audepradal/Documents/{ensoLB}_nino_{ensoUB}.txt"

df_nino = ds_nino.to_dataframe()
df_nino.to_csv(txt_path, index=True)

file_path = "/Users/marie-audepradal/Documents/ERA5SST.nc"
era5_ds = xr.open_dataset(file_path)
file_path = "/Users/marie-audepradal/Documents/1970-2024_tpe_ERA5Land_monthly.nc"
era5L_ds = xr.open_dataset(file_path)

#file_path=  "/Users/marie-audepradal/Documents/{ensoLB}nino{ensoLB}.nc"
#nino15_ds = xr.open_dataset(file_path)

#nc_path_nino_extr = f"/Users/marie-audepradal/Documents/{ensoLB}_nino_{ensoUB}.nc"

# Open the file
nino15_ds = xr.open_dataset(nc_path_enso_extr)

# Extract time values from both datasets
era5_times = era5_ds['valid_time']
era5L_times = era5L_ds['valid_time']
nino15_times = nino15_ds['time']

# Find common timestamps
matching_nino = np.intersect1d(era5L_times.values, nino15_times.values)

# Select the corresponding entries from the enso2023 dataset
matching_nino15_data = era5_ds.sel(valid_time=matching_nino)

# Compute the average over time of the matched nino extreme events

mean_sstKo = matching_nino15_data['sst'].mean(dim='valid_time')
mean_ssto = mean_sstKo.sel(latitude=slice(5, -5), longitude=slice(210, 270)) - 273.15


# For better visualization, focus on Nino 3 region. 
# plot the composite SST for extreme Nino 

# Plot the 2D map
plt.figure(figsize=(12, 6))
plt.pcolormesh(mean_ssto['longitude'], mean_ssto['latitude'], mean_ssto, shading='auto', cmap='viridis', vmin=22,
    vmax=30)
plt.colorbar(label='SST')
plt.title('Composite map of SST for extreme Nino Events')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.show()


# Load NetCDF data
ds_ERA5L = xr.open_dataset("/Users/marie-audepradal/Documents/1970-2024_tpe_ERA5Land_monthly.nc")  # Replace with your file path

# 2. Identify which variable holds temperature
#    (in your file it’s “t2m”, but this will print all data_vars)
print("Data variables in the file:", list(ds_ERA5L.data_vars))
# → e.g. ['t2m']

# 3. Select the temperature DataArray
temp = ds_ERA5L['t2m']
tp = ds_ERA5L['tp']

# 4. Compute the monthly climatology over valid_time
#    (group by the month of the timestamp and take the mean across all years)
#temp_clim = temp.groupby('valid_time.month').mean(dim='valid_time')
#tp_clim = tp.groupby('valid_time.month').mean(dim='valid_time')
#temp_clim.name = 't2m_climatology'
#tp_clim.name = 'precip_climatology'

# 5. Save the result to a new NetCDF
#temp_output_path = '/Users/marie-audepradal/Documents/temp_climatology_1970-2024_ERA5Land_monthly.nc'
#temp_clim.to_netcdf(temp_output_path)
#print(f"Monthly climatology written to: {temp_output_path}")
#tp_output_path = '/Users/marie-audepradal/Documents/precip_climatology_1970-2024_ERA5Land_monthly.nc'
#tp_clim.to_netcdf(tp_output_path)
#print(f"Monthly climatology written to: {tp_output_path}")
#print(temp_clim)
#print(tp_clim)

import rasterio
from rasterio import features
from rasterio.transform import from_origin
import fiona
from shapely.geometry import shape, Polygon, MultiPolygon, box
from shapely.ops import transform

# Re-open the datasets
ds_nino = xr.open_dataset(nc_path_enso_extr)
ds_t2 = xr.open_dataset('/Users/marie-audepradal/Documents/1970-2024_tpe_ERA5Land_monthly.nc')

# Identify the appropriate time variables
times_nino = ds_nino['time'].values
# ERA5-Land t2m uses 'valid_time' instead of 'time'
times_t2 = ds_t2['valid_time'].values

print("Nino dataset time coordinates:")
print(times_nino)

print("\nERA5-Land t2m dataset valid_time coordinates:")
print(times_t2)

# Find the common times between the two datasets
common_times = np.intersect1d(times_t2, times_nino)

print(f"\nCommon times ({len(common_times)} entries):")
print(common_times)

# Select t2m data for those common times
if 't2m' in ds_t2:
    t2m_var = 't2m'
else:
    # List variables if 't2m' not found
    print("\nVariables in ERA5-Land dataset:", list(ds_t2.data_vars))
    t2m_var = list(ds_t2.data_vars)[0]  # fallback

# Subset using valid_time as the coordinate
ds_t2_sel = ds_t2.sel(valid_time=common_times)[[t2m_var]]

# Save the subset to a new NetCDF file
output_path = '/Users/marie-audepradal/Documents/t2m_matched_times.nc'
ds_t2_sel.to_netcdf(output_path)


print(f"\nSubset t2m data saved to: {output_path}")
print(ds_t2_sel)

#select precipitation data
# Select tp data for those common times
if 'tp' in ds_t2:
    tp_var = 'tp'
else:
    # List variables if 'tp' not found
    print("\nVariables in ERA5-Land dataset:", list(ds_t2.data_vars))
    tp_var = list(ds_t2.data_vars)[0]  # fallback

# Subset using valid_time as the coordinate
ds_tp_sel = ds_t2.sel(valid_time=common_times)[[tp_var]]

# Save the subset to a new NetCDF file
output_path = '/Users/marie-audepradal/Documents/tp_matched_times.nc'
ds_tp_sel.to_netcdf(output_path)

#just a sanity check: times should be identical for both variables
print(f"\nSubset t2m  and tp data saved to: {output_path}")
print(ds_t2_sel)
print(ds_tp_sel)

ds_matched = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')
ds_matched
ds_tp_matched = xr.open_dataset('/Users/marie-audepradal/Documents/tp_matched_times.nc')


# --------------------------------------------------------------------------------
# 1. Load the “matched‐times” t2m file
# --------------------------------------------------------------------------------
ds_matched = xr.open_dataset('/Users/marie-audepradal/Documents/t2m_matched_times.nc')
# “t2m” is in Kelvin, dims = (valid_time, latitude, longitude)

# --------------------------------------------------------------------------------
# 2. Load the monthly climatology (1970–2024)
# --------------------------------------------------------------------------------
clim_path = '/Users/marie-audepradal/Documents/temp_climatology_1970-2024_ERA5Land_monthly.nc'
ds_clim = xr.open_dataset(clim_path)
# dims = (month, latitude, longitude), variable = 't2m_climatology' in K

# --------------------------------------------------------------------------------
# 3. Load country shapefile & extract geometries for six countries
# --------------------------------------------------------------------------------
shapefile_path = '/Users/marie-audepradal/Documents/WB_countries_Admin0_10m.shp'
with fiona.open(shapefile_path) as shp:
    mexico_geom = None
    colombia_geom = None
    chile_geom = None
    indonesia_geom = None
    south_africa_geom = None
    brazil_geom = None
    peru_geom = None
    world_geoms = []

    for feature in shp:
        props = feature['properties']
        geom = shape(feature['geometry'])
        world_geoms.append(geom)
        name = props.get('NAME_EN') or props.get('WB_NAME')

        if name == 'Mexico':
            mexico_geom = geom
        elif name == 'Colombia':
            colombia_geom = geom
        elif name == 'Chile':
            chile_geom = geom
        elif name == 'Indonesia':
            indonesia_geom = geom
        elif name in ('South Africa', 'Republic of South Africa'):
            south_africa_geom = geom
        elif name == 'Brazil':
            brazil_geom = geom
        elif name == 'Peru':
            peru_geom = geom

for nm, geom in [
    ('Mexico', mexico_geom),
    ('Colombia', colombia_geom),
    ('Chile', chile_geom),
    ('Indonesia', indonesia_geom),
    ('South Africa', south_africa_geom),
    ('Brazil', brazil_geom),
    ('Peru', peru_geom),
]:
    if geom is None:
        raise ValueError(f"{nm} geometry not found in shapefile.")

# --------------------------------------------------------------------------------
# 4. Prepare (lon, lat) grid & affine transform
# --------------------------------------------------------------------------------
lons_full = ds_matched['longitude'].values.copy()  # probably 0→359.9
lats      = ds_matched['latitude'].values.copy()   #  90→-90

# Convert 0–360 → -180–180 if needed
if lons_full.min() >= 0:
    lons_mod = (lons_full + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
else:
    lons = lons_full.copy()
    order = np.arange(len(lons))

dx = float(lons[1] - lons[0])
dy = float(abs(lats[1] - lats[0]))

west  = float(lons.min())
north = float(lats.max())
transform_grid = from_origin(west, north, dx, dy)

shape_out   = (len(lats), len(lons))
lon_grid, lat_grid = np.meshgrid(lons, lats)

# --------------------------------------------------------------------------------
# 5. Rasterize each country into a 0/1 mask
# --------------------------------------------------------------------------------
mask_mexico       = features.rasterize([(mexico_geom, 1)], out_shape=shape_out,
                                        transform=transform_grid, fill=0, dtype=np.uint8)
mask_colombia     = features.rasterize([(colombia_geom, 1)], out_shape=shape_out,
                                        transform=transform_grid, fill=0, dtype=np.uint8)
mask_chile        = features.rasterize([(chile_geom, 1)], out_shape=shape_out,
                                        transform=transform_grid, fill=0, dtype=np.uint8)
mask_indonesia    = features.rasterize([(indonesia_geom, 1)], out_shape=shape_out,
                                        transform=transform_grid, fill=0, dtype=np.uint8)
mask_south_africa = features.rasterize([(south_africa_geom, 1)], out_shape=shape_out,
                                        transform=transform_grid, fill=0, dtype=np.uint8)
mask_brazil       = features.rasterize([(brazil_geom, 1)], out_shape=shape_out,
                                        transform=transform_grid, fill=0, dtype=np.uint8)
mask_peru      = features.rasterize([(peru_geom, 1)], out_shape=shape_out,
                                        transform=transform_grid, fill=0, dtype=np.uint8)

# --------------------------------------------------------------------------------
# 6. Subtract monthly climatology at each time → accumulate anomaly
# --------------------------------------------------------------------------------
n_times = ds_matched.dims['valid_time']
sum_anomaly = np.zeros((len(lats), len(lons)), dtype=np.float64)

for ti, timestamp in enumerate(ds_matched['valid_time'].values):
    t2m_slice = ds_matched['t2m'].isel(valid_time=ti).values  # shape = (nlat, nlon_full)
    # reorder longitudes if needed
    if lons_full.min() >= 0:
        t2m_reordered = t2m_slice[:, order]
    else:
        t2m_reordered = t2m_slice.copy()

    # extract month (1–12)
    month = int(str(np.datetime64(timestamp, 'M'))[5:7])

    # climatology slice (in K)
    clim_slice = ds_clim['t2m_climatology'].isel(month=month - 1).values  # (nlat, nlon_full)
    if lons_full.min() >= 0:
        clim_reordered = clim_slice[:, order]
    else:
        clim_reordered = clim_slice.copy()

    anomaly_this = t2m_reordered - clim_reordered
    sum_anomaly += anomaly_this

mean_anomaly = sum_anomaly / float(n_times)  # shape = (nlat, nlon), in K (≡ °C numerically)

# --------------------------------------------------------------------------------
# 7. Mask each country’s anomaly & find common vmin/vmax
# --------------------------------------------------------------------------------
anom_mexico       = np.where(mask_mexico       == 1, mean_anomaly, np.nan)
anom_colombia     = np.where(mask_colombia     == 1, mean_anomaly, np.nan)
anom_chile        = np.where(mask_chile        == 1, mean_anomaly, np.nan)
anom_indonesia    = np.where(mask_indonesia    == 1, mean_anomaly, np.nan)
anom_south_africa = np.where(mask_south_africa == 1, mean_anomaly, np.nan)
anom_brazil       = np.where(mask_brazil       == 1, mean_anomaly, np.nan)
anom_peru         = np.where(mask_peru         == 1, mean_anomaly, np.nan)

all_vals = np.concatenate([
    anom_mexico[np.isfinite(anom_mexico)].ravel(),
    anom_colombia[np.isfinite(anom_colombia)].ravel(),
    anom_chile[np.isfinite(anom_chile)].ravel(),
    anom_indonesia[np.isfinite(anom_indonesia)].ravel(),
    anom_south_africa[np.isfinite(anom_south_africa)].ravel(),
    anom_brazil[np.isfinite(anom_brazil)].ravel(),
    anom_peru[np.isfinite(anom_peru)].ravel(),
])
vmin = np.nanmin(all_vals)
vmax = np.nanmax(all_vals)

# --------------------------------------------------------------------------------
# 8. Plot one map, zoomed on South America
# --------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))

# 8.1 Plot all world boundaries faintly
for geom in world_geoms:
    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='black', linewidth=0.1)
    else:
        x, y = geom.exterior.xy
        ax.plot(x, y, color='black', linewidth=0.1)

# 8.2 Plot each country’s anomaly (°C) with the common vmin/vmax
vmin=-3
vmax=3
ax.pcolormesh(lon_grid, lat_grid, anom_mexico,
              cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
ax.pcolormesh(lon_grid, lat_grid, anom_colombia,
              cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
ax.pcolormesh(lon_grid, lat_grid, anom_chile,
              cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
ax.pcolormesh(lon_grid, lat_grid, anom_indonesia,
              cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
ax.pcolormesh(lon_grid, lat_grid, anom_south_africa,
              cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
ax.pcolormesh(lon_grid, lat_grid, anom_brazil,
              cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
ax.pcolormesh(lon_grid, lat_grid, anom_peru,
              cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)


# 8.3 Overlay thick black boundaries for each country
def plot_boundary(geom, lw=1.2):
    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='black', linewidth=0.1)
    else:
        x, y = geom.exterior.xy
        ax.plot(x, y, color='black', linewidth=0.1)

plot_boundary(mexico_geom)
plot_boundary(colombia_geom)
plot_boundary(chile_geom)
plot_boundary(indonesia_geom)
plot_boundary(south_africa_geom)
plot_boundary(brazil_geom)
plot_boundary(peru_geom)

# 8.4 Colorbar
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax)),
                    ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Mean 2 m Temperature Anomaly (°C)')

# 8.5 ZOOM onto South America:
ax.set_xlim(-90, -30)   # approximate lon‐bounds for South America
ax.set_ylim(-60,  20)   # approximate lat‐bounds for South America

ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.set_title(
    'Mean 2 m Temperature Anomaly (°C)\n'
    'Zoomed on South America\n'
    'for Mexico, Colombia, Chile, Indonesia, South Africa, Brazil, Peru'
)

plt.tight_layout()
out_path = f"/Users/marie-audepradal/Documents/{ensoLB}_enso_bounded_by_{ensoUB}.png"
plt.savefig(out_path, dpi=600)
#"plt.savefig('/Users/marie-audepradal/Desktop/{ensoLB}enso{ensoUB}.png', dpi=200)
plt.show()


# --------------------------------------------------------------------------------
# 1. Load the “matched‐times” tp file
# --------------------------------------------------------------------------------
ds_matched = xr.open_dataset('/Users/marie-audepradal/Documents/tp_matched_times.nc')
# “tp” is in mm, dims = (valid_time, latitude, longitude)

# --------------------------------------------------------------------------------
# 2. Load the monthly precipitation climatology (1970–2024)
# --------------------------------------------------------------------------------
clim_path = '/Users/marie-audepradal/Documents/precip_climatology_1970-2024_ERA5Land_monthly.nc'
ds_clim = xr.open_dataset(clim_path)
# dims = (month, latitude, longitude), variable = 'precip_climatology' in mm

# --------------------------------------------------------------------------------
# 3. Load country shapefile & extract geometries for six countries
# --------------------------------------------------------------------------------
shapefile_path = '/Users/marie-audepradal/Documents/WB_countries_Admin0_10m.shp'
with fiona.open(shapefile_path) as shp:
    countries = {
        'Mexico': None, 'Colombia': None, 'Chile': None, 'Indonesia': None,
        'South Africa': None, 'Brazil': None, 'Peru': None
    }
    world_geoms = []

    for feature in shp:
        props = feature['properties']
        geom = shape(feature['geometry'])
        world_geoms.append(geom)
        name = props.get('NAME_EN') or props.get('WB_NAME')

        if name in countries:
            countries[name] = geom
        elif name == 'Republic of South Africa':
            countries['South Africa'] = geom

for name, geom in countries.items():
    if geom is None:
        raise ValueError(f"{name} geometry not found in shapefile.")

# --------------------------------------------------------------------------------
# 4. Prepare (lon, lat) grid & affine transform
# --------------------------------------------------------------------------------
lons_full = ds_matched['longitude'].values.copy()
lats = ds_matched['latitude'].values.copy()

if lons_full.min() >= 0:
    lons_mod = (lons_full + 180) % 360 - 180
    order = np.argsort(lons_mod)
    lons = lons_mod[order]
else:
    lons = lons_full.copy()
    order = np.arange(len(lons))

dx = float(lons[1] - lons[0])
dy = float(abs(lats[1] - lats[0]))
transform_grid = from_origin(float(lons.min()), float(lats.max()), dx, dy)
shape_out = (len(lats), len(lons))
lon_grid, lat_grid = np.meshgrid(lons, lats)

# --------------------------------------------------------------------------------
# 5. Rasterize each country into a 0/1 mask
# --------------------------------------------------------------------------------
masks = {
    name: features.rasterize([(geom, 1)], out_shape=shape_out,
                             transform=transform_grid, fill=0, dtype=np.uint8)
    for name, geom in countries.items()
}

# --------------------------------------------------------------------------------
# 6. Subtract monthly climatology from each time step → accumulate anomaly
# --------------------------------------------------------------------------------
n_times = ds_matched.dims['valid_time']
sum_anomaly = np.zeros((len(lats), len(lons)), dtype=np.float64)

for ti, timestamp in enumerate(ds_matched['valid_time'].values):
    tp_slice = ds_matched['tp'].isel(valid_time=ti).values
    tp_reordered = tp_slice[:, order] if lons_full.min() >= 0 else tp_slice.copy()

    month = int(str(np.datetime64(timestamp, 'M'))[5:7])
    clim_slice = ds_clim['precip_climatology'].isel(month=month - 1).values
    clim_reordered = clim_slice[:, order] if lons_full.min() >= 0 else clim_slice.copy()

    anomaly_this = tp_reordered - clim_reordered
    sum_anomaly += anomaly_this

mean_anomaly = sum_anomaly / float(n_times)  # units: mm

# --------------------------------------------------------------------------------
# 7. Mask each country’s anomaly & compute color scale bounds
# --------------------------------------------------------------------------------
anomalies = {
    name: np.where(mask == 1, mean_anomaly, np.nan)
    for name, mask in masks.items()
}

all_vals = np.concatenate([arr[np.isfinite(arr)].ravel() for arr in anomalies.values()])
vmin = np.nanmin(all_vals)
vmax = np.nanmax(all_vals)

# --------------------------------------------------------------------------------
# 8. Plot one map, zoomed on South America
# --------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))

for geom in world_geoms:
    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='black', linewidth=0.1)
    else:
        x, y = geom.exterior.xy
        ax.plot(x, y, color='black', linewidth=0.1)

# Plot anomalies
vmin, vmax = -10, 10  # Adjust for expected range in mm
for name, data in anomalies.items():
    ax.pcolormesh(lon_grid, lat_grid, data*1000, cmap='BrBG', shading='auto', vmin=vmin, vmax=vmax)

# Overlay country boundaries
def plot_boundary(geom, lw=1.2):
    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='black', linewidth=0.2)
    else:
        x, y = geom.exterior.xy
        ax.plot(x, y, color='black', linewidth=0.2)

for geom in countries.values():
    plot_boundary(geom)

# Colorbar
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='BrBG', norm=plt.Normalize(vmin=vmin, vmax=vmax)),
                    ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Mean Precipitation Anomaly (mm)')

# Zoom on South America
ax.set_xlim(-90, -30)
ax.set_ylim(-60, 20)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.set_title(
    'Mean Precipitation Anomaly (mm)\n'
    'Zoomed on South America\n'
    'for Mexico, Colombia, Chile, Indonesia, South Africa, Brazil, Peru'
)

plt.tight_layout()
out_path = "/Users/marie-audepradal/Documents/tp_anomaly_SA.png"
plt.savefig(out_path, dpi=600)
plt.show()


# Load the climatology dataset
clim_path = '/Users/marie-audepradal/Documents/precip_climatology_1970-2024_ERA5Land_monthly.nc'
ds_clim = xr.open_dataset(clim_path)

# Extract January (month index 0)
may_precip = ds_clim['precip_climatology'].isel(month=4)

# Extract coordinates
lats = ds_clim['latitude'].values
lons_full = ds_clim['longitude'].values

# Convert longitudes if needed
if lons_full.min() >= 0:
    lons = (lons_full + 180) % 360 - 180
    order = np.argsort(lons)
    may_precip = may_precip.values[:, order]*1000
    lons = lons[order]
else:
    may_precip = may_precip.values
    lons = lons_full

# Create meshgrid
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
pc = ax.pcolormesh(lon_grid, lat_grid, may_precip,
                   cmap='Blues', shading='auto', vmin=0, vmax=20)

# Colorbar
cbar = plt.colorbar(pc, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Precipitation Climatology (mm)')

# Zoom on South America
ax.set_xlim(-90, -20)
ax.set_ylim(-60, 20)

ax.set_title('May Precipitation Climatology (mm)\nSouth America')
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

# Load the climatology dataset
clim_path = '/Users/marie-audepradal/Documents/precip_climatology_1970-2024_ERA5Land_monthly.nc'
ds_clim = xr.open_dataset(clim_path)

# Extract January (month index 0)
may_precip = ds_clim['precip_climatology'].isel(month=4)

# Extract coordinates
lats = ds_clim['latitude'].values
lons_full = ds_clim['longitude'].values

# Convert longitudes if needed
if lons_full.min() >= 0:
    lons = (lons_full + 180) % 360 - 180
    order = np.argsort(lons)
    may_precip = may_precip.values[:, order]*1000
    lons = lons[order]
else:
    may_precip = may_precip.values
    lons = lons_full

# Create meshgrid
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
pc = ax.pcolormesh(lon_grid, lat_grid, may_precip,
                   cmap='Blues', shading='auto', vmin=0, vmax=20)

# Colorbar
cbar = plt.colorbar(pc, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Precipitation Climatology (mm)')

# Zoom on South America
ax.set_xlim(-90, -20)
ax.set_ylim(-60, 20)

ax.set_title('May Precipitation Climatology (mm)\nSouth America')
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()

# Load the datasets
ds_1983 = xr.open_dataset('/Users/marie-audepradal/Documents/1970-2024_tpe_ERA5Land_monthly.nc')
ds_clim = xr.open_dataset('/Users/marie-audepradal/Documents/precip_climatology_1970-2024_ERA5Land_monthly.nc')

# Select the March 1983 t2m data using the 'valid_time' coordinate
t2m_mar1983 = ds_1983['tp'].sel(valid_time='2014-11-01')
# Select the climatological t2m for March (month=3)
t2m_clim_mar = ds_clim['precip_climatology'].sel(month=11)

# Compute the anomaly
anomaly = (t2m_mar1983)*1000 - (t2m_clim_mar)*1000

# Plot the anomaly
#plt.figure(figsize=(12, 6))
#anomaly.plot()
#plt.title('Precipitation Anomaly ')
#plt.show()

# Define the custom colormap and normalization
cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_diverging",
    ["blue", "white", "brown"],
    N=256
)
norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)

# Plot with Cartopy
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())  # Use PlateCarree for regular lat/lon grids
anomaly.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, cbar_kwargs={'label': 'Anomaly (in mm)'})
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_title('Precipitation Anomaly for NOV 2014')
# Save the figure
plt.tight_layout()
out_path = "/Users/marie-audepradal/Desktop/precip_anom_NOV2014.png"
plt.savefig(out_path, dpi=600)
plt.show()

from PIL import Image
import os

# List of PNG files to combine, replace with your actual file paths
png_files = ["/Users/marie-audepradal/Desktop/precip_anom_NOV2014.png", "/Users/marie-audepradal/Desktop/precip_anom_JAN2015.png", "/Users/marie-audepradal/Desktop/precip_anom_APR2015.png", "/Users/marie-audepradal/Desktop/precip_anom_NOV2015.png", "/Users/marie-audepradal/Desktop/precip_anom_JAN2016.png","/Users/marie-audepradal/Desktop/precip_anom_APR2016.png"]

# Open all images and convert to RGB (PDFs don't support alpha channel)
images = [Image.open(f).convert("RGB") for f in png_files]

# Save as a single PDF
if images:
    images[0].save("/Users/marie-audepradal/Desktop/output.pdf", save_all=True, append_images=images[1:])

	


# %%
# load all hourly ERA5-land data for 1970-1971
import xarray as xr
import os
import dask


#define base directory
base_dir = "/Users/marie-audepradal/Documents/"

#generate list of file names from 1970_01.nc to 1971_12.nc
years = [2015]
months = [1,4]
file_paths = [os.path.join(base_dir, f"{year}_{month:02d}.nc") for year in years for month in months]

file_paths = [f for f in file_paths if os.path.exists(f)]

#load all files as a combined dataset
ds = xr.open_mfdataset(file_paths, combine='by_coords', parallel=True)

#select only t2m and tp variables
ds_selected = ds[['t2m', 'tp']]

#write to a new netcdf file
output_file = "/Users/marie-audepradal/Documents/combined_era5land2015-t2m-tp.nc"
ds_selected.to_netcdf(output_file)


# %%
import xarray as xr
import matplotlib.pyplot as plt

# Define the file path
combined_file = "/Users/marie-audepradal/Documents/combined_era5land1970-71-t2m-tp.nc"

# Open the NetCDF file
ds= xr.open_dataset(combined_file)

# View the dataset summary
print(ds)
ds
# Select data for January 2015
tp_jan2015 = ds['tp'].sel(valid_time=ds.valid_time.dt.month == 1)
tp_apr2015 = ds['tp'].sel(valid_time=ds.valid_time.dt.month == 4)
# Accumulate (sum) over the time dimension
tp_accumulated_jan2015 = tp_jan2015.sum(dim='valid_time')
tp_accumulated_apr2015 = tp_apr2015.sum(dim='valid_time')
# View the result
print(tp_accumulated_jan2015)
print(tp_accumulated_apr2015)



# %%
# Plot January
plt.figure(figsize=(10, 5))
tp_accumulated_jan2015.plot(
    vmin=0, vmax=1,
    cbar_kwargs={'label': 'Precipitation (m)'}
)
plt.title("Accumulated Total Precipitation - January")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Plot April
plt.figure(figsize=(10, 5))
tp_accumulated_apr2015.plot(
    vmin=0, vmax=1,
    cbar_kwargs={'label': 'Precipitation (m)'}
)
plt.title("Accumulated Total Precipitation - April")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()



# %%
import xarray as xr
import numpy as np

# Load dataset
file_path = "/Users/marie-audepradal/Documents/combined_era5land1970-71-t2m-tp.nc"
ds = xr.open_dataset(file_path)

# Use the correct time coordinate
time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'

# Round time to hour in case it's not exact
time_hour = ds[time_coord].dt.hour

# Extract times that are exactly 23:00 (11 PM)
ds_11pm = ds.sel({time_coord: time_hour == 23})

# Extract only the precipitation variable
ds_precip_11pm = ds_11pm[['tp']]

# Save to NetCDF file
output_path = "/Users/marie-audepradal/Documents/daily_precip.nc"
ds_precip_11pm.to_netcdf(output_path)

print(f"Precipitation at 11 PM saved to: {output_path}")


# %%
import xarray as xr
import matplotlib.pyplot as plt

# Load dataset
file_path = "/Users/marie-audepradal/Documents/combined_era5land1970-71-t2m-tp.nc"
ds = xr.open_dataset(file_path)

# Use the correct time coordinate
time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'

# Convert tp to mm (assuming meters)
tp_mm = ds['tp'] * 1000

# Compute spatial average over the whole domain
tp_avg = tp_mm.mean(dim=['latitude', 'longitude'])

# Plot time series
plt.figure(figsize=(12, 5))
tp_avg.plot()
plt.title("Spatial Average of Precipitation Over Time (1970–71)")
plt.xlabel("Time")
plt.ylabel("Precipitation (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Extract 11 PM time steps
ds_11pm = ds.sel({time_coord: ds[time_coord].dt.hour == 23})

# Select the precipitation variable
tp_11pm = ds_11pm['tp']

# Convert to millimeters (assuming unit is meters)
tp_11pm_mm = tp_11pm * 1000

# Average over spatial dimensions (assumes lat/lon are the names)
tp_11pm_avg = tp_11pm_mm.mean(dim=['latitude', 'longitude'])

# Plot the time series
plt.figure(figsize=(12, 5))
tp_11pm_avg.plot(marker='o')
plt.title("Average Precipitation at 11 PM (1970–71)")
plt.ylabel("Precipitation (mm)")
plt.xlabel("Time")
plt.grid(True)
plt.tight_layout()
plt.show()




# %%
import xarray as xr
import matplotlib.pyplot as plt

# Load dataset
file_path = "/Users/marie-audepradal/Documents/combined_era5land1970-71-t2m-tp.nc"
ds = xr.open_dataset(file_path)

# Use the correct time coordinate
time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'

# Convert tp to mm (assuming meters)
tp_mm = ds['tp'] * 1000

# Filter for January 1–15 (both 1970 and 1971 if present)
tp_jan1_15 = tp_mm.sel({time_coord: (ds[time_coord].dt.month == 1) & (ds[time_coord].dt.day <= 15)})

# Compute spatial average
tp_avg_jan = tp_jan1_15.mean(dim=['latitude', 'longitude'])

# Plot
plt.figure(figsize=(12, 5))
tp_avg_jan.plot(marker='o')
plt.title("Spatial Average of Precipitation (Jan 1–15, 1970–71)")
plt.xlabel("Time")
plt.ylabel("Precipitation (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
import xarray as xr
import matplotlib.pyplot as plt

# Load dataset
file_path = "/Users/marie-audepradal/Documents/2015_04.nc"
ds = xr.open_dataset(file_path)

# Identify the correct time coordinate
time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'

# Extract tp and convert to mm (assuming unit is meters)
tp_mm = ds['tp'] * 1000

# Compute spatial average
tp_avg = tp_mm.mean(dim=['latitude', 'longitude'])

# Plot the time series
plt.figure(figsize=(12, 5))
tp_avg.plot(marker='o')
plt.title("Spatial Average of Precipitation (April 2015)")
plt.xlabel("Time")
plt.ylabel("Precipitation (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%

# Extract 11 PM time steps
ds_11pm = ds.sel({time_coord: ds[time_coord].dt.hour == 23})

# Select the precipitation variable
tp_11pm = ds_11pm['tp']

# Convert to millimeters (assuming unit is meters)
tp_11pm_mm = tp_11pm * 1000

# Average over spatial dimensions (assumes lat/lon are the names)
tp_11pm_avg = tp_11pm_mm.mean(dim=['latitude', 'longitude'])

# Plot the time series
plt.figure(figsize=(12, 5))
tp_11pm_avg.plot(marker='o')
plt.title("Average Precipitation at 11 PM (1970–71)")
plt.ylabel("Precipitation (mm)")
plt.xlabel("Time")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
import xarray as xr

# Load dataset
file_path = "/Users/marie-audepradal/Documents/1970-2024_tpe_ERA5Land_monthly.nc"
ds = xr.open_dataset(file_path)

# Identify correct time coordinate
time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'

# Select April 2015
tp_april_2015 = ds['tp'].sel({time_coord: (ds[time_coord].dt.year == 2015) & (ds[time_coord].dt.month == 4)})

# Convert to mm (if in meters)
tp_april_2015_mm = tp_april_2015 * 1000

# Compute spatial average
tp_april_2015_avg = tp_april_2015_mm.mean(dim=['latitude', 'longitude'])

# Extract the value
tp_april_2015_value = tp_april_2015_avg.item()

# Print result
print(f"Spatial average of precipitation for April 2015: {tp_april_2015_value:.2f} mm")



