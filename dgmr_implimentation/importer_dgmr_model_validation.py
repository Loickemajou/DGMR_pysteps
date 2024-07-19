import xarray as xr
import glob
import os
import tarfile
import numpy as np
import pandas as pd
import time
import xesmf as xe
import xskillscore as xs
import sys
from wradlib.io import read_opera_hdf5

"""
# Functions and scripts to:
# - read in RMI-precitation data and transform to xarray structure
# - precipitation validation using xarray
# Author: Michiel Van Ginderachter (michiel.vanginderachter@meteo.be)
# Modified by Yann Laroche (yelaroche@gmail.com)

# This scripts assumes that:
# - pySTEPS nowcasts are in CF1.7 compliant netCDF-format, arranged by case-number in a tarball
# - Radar QPE products are stored in an HDF5-file according to OPERA specifications
# - INCA-BE nowcasts are stored in .gz compressed GRIB(I)-files
# - DGMR and LDCast nowcasts are stored as .npy files

# This scipts does:
# - Calculate deterministic (RMSE, BIAS and MAPE) and probabilistic (BRIER, CRPS and Reliabity) score
#   for a subset of the nowcasts that is part of a specific case

"""


## Uncomment the next lines if pyproj is needed for the importer.
try:
    import pyproj

    PYPROJ_IMPORTED = True
except ImportError:
    PYPROJ_IMPORTED = False

from pysteps.decorators import postprocess_import

# Open RMI QPE HDF-file and transform to xarray
# # TODO: Still some problems with the x,y coordinates in the 15th decimal

def read_hdf5_to_xarray(fname):
    
 
    # Read the content
    fcontent = read_opera_hdf5(fname)
    # Determine the quantity
    try:
        quantity = fcontent['dataset1/data1/what']['quantity'].decode()
    except:
        quantity = fcontent['dataset1/data1/what']['quantity']
    if quantity == 'RATE':
        short_name = 'precip_intensity'
        long_name = 'instantaneous precipitation rate'
        units = 'mm h-1'
    else:
        #TODO: implement other quantities
        raise Exception(f"Quantity {quantity} not yet implemented.")
    # Create the grid
    try:
        projection = fcontent["where"]["projdef"].decode()
    except:
        projection = fcontent["where"]["projdef"]
    gridspec = fcontent["dataset1/where"]
    # - X and Y coordinates
    x = np.linspace(
        gridspec['UL_x'],
        gridspec['UL_x']+gridspec['xsize']*gridspec['xscale'],
        num = gridspec['xsize'],
        endpoint=False
        )
    x += gridspec['xscale']
    y = np.linspace(
        gridspec['UL_y'],
        gridspec['UL_y']-gridspec['ysize']*gridspec['yscale'],
        num=gridspec['ysize'],
        endpoint=False
        )
    y -= gridspec['yscale']/2
    x_2d, y_2d = np.meshgrid(x, y)
    pr=pyproj.Proj(projection)
    # - Lon and Lat coordinates
    lon, lat = pr(x_2d.flatten(),y_2d.flatten(), inverse=True)
    lon = lon.reshape(gridspec['ysize'],gridspec['xsize'])
    lat = lat.reshape(gridspec['ysize'],gridspec['xsize'])
    
    # Build the xarray dataset
    ds = xr.Dataset(
        # - data
        data_vars = {
            short_name : (
                #FIXME: is it really ['x','y'] or is should it be ['y','x']
                ['x','y'],
                fcontent["dataset1/data1/data"],
                {'long_name' : long_name, 'units': units}
                )
            },
        # - coordinates
        coords = {
            'x' : (
                ['x'],
                x,
                {
                    'axis' : 'X', 
                    'standard_name': 'projection_x_coordinate',
                    'long_name': 'x-coordinate in Cartesian system',
                    'units': 'm'
                    }
                ),
            'y' : (
                ['y'], 
                y,
                {
                    'axis' : 'Y', 
                    'standard_name': 'projection_y_coordinate',
                    'long_name': 'y-coordinate in Cartesian system',
                    'units': 'm'
                    }
                ),
            'lon' : (
                ['y','x'],
                lon,
                {
                    'standard_name': 'longitude',
                    'long_name': 'longitude coordinate',
                    'units': 'degrees_east'
                    }
                ),
            'lat' : (
                ['y','x'],
                lat,
                {
                    'standard_name': 'latitude',
                    'long_name': 'latitude coordinate',
                    'units': 'degrees_north'

                    }
                )
            }
    )
    return(ds)

def create_xrDatasets(data_folder, case):
    dgmr = []
    dataset = []
    
    # The code considers that the ouputs were saved in the same folder
    # and separated into two folders (dgmr and ldcast)
    output_folder = "" # Path where the DL nowcasts are stored
    dgmr_path = os.path.join(output_folder, 'dgmr', f'DGMRpred_{case}.npy')
    dgmr_array = np.load(dgmr_path)
    
    
    fns = []
    # The slice used here complements to the slice used in generate_input.py
    # Such that the appropriate files are loaded to match 18 predictions frames
    for filename in os.listdir(data_folder)[26:62]:
        if filename.endswith('mfb.hdf'):
            fns.append(f"{data_folder}/{filename}")
    
    for i, file_name in enumerate(fns):
        # Read the content
        file_content = read_opera_hdf5(file_name)

        # Extract time information
        time_str = os.path.splitext(os.path.basename(file_name))[0].split('.', 1)[0]
        time = pd.to_datetime(time_str, format='%Y%m%d%H%M%S')

        # Extract quantity information
        try:
            quantity = file_content['dataset1/data1/what']['quantity'].decode()
        except:
            quantity = file_content['dataset1/data1/what']['quantity']

        # Set variable properties based on quantity
        if quantity == 'RATE':
            short_name = 'precip_intensity'
            long_name = 'instantaneous precipitation rate'
            units = 'mm h-1'
        else:
            raise Exception(f"Quantity {quantity} not yet implemented.")

        # Create the grid
        projection = file_content.get("where", {}).get("projdef", "")
        if type(projection) is not str:
            projection = projection.decode("UTF-8")

        gridspec = file_content.get("dataset1/where", {})

        x = np.linspace(gridspec.get('UL_x', 0),
                        gridspec.get('UL_x', 0) + gridspec.get('xsize', 0) * gridspec.get('xscale', 0),
                        num=gridspec.get('xsize', 0), endpoint=False)
        x += gridspec.get('xscale', 0)
        y = np.linspace(gridspec.get('UL_y', 0),
                        gridspec.get('UL_y', 0) - gridspec.get('ysize', 0) * gridspec.get('yscale', 0),
                        num=gridspec.get('ysize', 0), endpoint=False)
        y -= gridspec.get('yscale', 0) / 2

        x_2d, y_2d = np.meshgrid(x, y)

        pr = pyproj.Proj(projection)
        
        ens_number = np.linspace(1, 48, 48)
        lon, lat = pr(x_2d.flatten(), y_2d.flatten(), inverse=True)
        lon = lon.reshape(gridspec.get('ysize', 0), gridspec.get('xsize', 0))
        lat = lat.reshape(gridspec.get('ysize', 0), gridspec.get('xsize', 0))
        
        # crop the values to fit the 256x256 dimensions of the predictions
        low = (700//2) - (256//2)
        high = (700//2) + (256//2)
        lon = lon[low:high,low:high]
        lat = lat[low:high,low:high]
        x = x[low:high]
        y = y[low:high]
        
        sources = ['ldcast',  'radar']
        for source in sources:
        # Build the xarray dataset
            if source == 'dgmr':
                content = dgmr_array[:, i, ..., 0]
                content = np.reshape(content, (48, 256, 256))
                content = np.transpose(content, (0, 2, 1))
            else:
                content = file_content.get("dataset1/data1/data", np.nan)
                content = content[low:high,low:high]
                content = np.transpose(content, (1,0))

            if source == 'radar':
                ds = xr.Dataset(
                    data_vars={
                        short_name: (['y','x'], content,
                                    {'long_name': long_name, 'units': units})
                    },
                    coords={
                        'y': (['y'], y, {'axis': 'Y', 'standard_name': 'projection_y_coordinate',
                                        'long_name': 'y-coordinate in Cartesian system', 'units': 'm'}),
                        'x': (['x'], x, {'axis': 'X', 'standard_name': 'projection_x_coordinate',
                                        'long_name': 'x-coordinate in Cartesian system', 'units': 'm'}),
                        'lon': (['y', 'x'], lon, {'standard_name': 'longitude', 'long_name': 'longitude coordinate',
                                                'units': 'degrees_east'}),
                        'lat': (['y', 'x'], lat, {'standard_name': 'latitude', 'long_name': 'latitude coordinate',
                                                'units': 'degrees_north'})
                    }
                )
                ds['time'] = time
            else:
                ds = xr.Dataset(
                    data_vars={
                        short_name: (['ens_number','y','x'], content,
                                    {'long_name': long_name, 'units': units})
                    },
                    coords={
                        'ens_number': (['ens_number'], ens_number, {'axis': 'n_ens_memeber', 'standard_name': 'number_ens_member',
                                        'long_name': 'number of the ensemble member', 'units': 'ens_member'}),
                        'y': (['y'], y, {'axis': 'Y', 'standard_name': 'projection_y_coordinate',
                                        'long_name': 'y-coordinate in Cartesian system', 'units': 'm'}),
                        'x': (['x'], x, {'axis': 'X', 'standard_name': 'projection_x_coordinate',
                                        'long_name': 'x-coordinate in Cartesian system', 'units': 'm'}),
                        'lon': (['y', 'x'], lon, {'standard_name': 'longitude', 'long_name': 'longitude coordinate',
                                                'units': 'degrees_east'}),
                        'lat': (['y', 'x'], lat, {'standard_name': 'latitude', 'long_name': 'latitude coordinate',
                                                'units': 'degrees_north'})
                    }
                )
                ds['time'] = time

            # Append the dataset to the list
            if source == 'dgmr':
                dgmr.append(ds)
            else:
                dataset.append(ds)

    # Concatenate datasets along the time dimension
    final_dgmr = xr.concat(dgmr, dim='time')
    final_dataset = xr.concat(dataset,dim='time')


def usage():
    print("xrValidation [ncase]")
    print("    ncase: case number")
    print("Run the validation for case <ncase>.")
    sys.exit(1)



