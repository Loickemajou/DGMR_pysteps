
import tensorflow as tf
import numpy as np
import os
import numpy as np
from wradlib.io import read_opera_hdf5
import xarray as xr
import pandas as pd
from params import *
import torch

## Uncomment the next lines if pyproj is needed for the importer.
try:
    import pyproj

    PYPROJ_IMPORTED = True
except ImportError:
    PYPROJ_IMPORTED = False



"""
Generates and preprocessing input that will be used for prediction on the model


When run, this file creates two sets of inputs. One set of .nf files 
which can be used in plotting.ipynb for visualization with cartopy, 
and a set of .npy files which can be used in 
pred_on_colab.ipynb to generate the predictions.
"""





def r_to_dbz(r):
    '''
    Convert mm/h to dbz
    '''
    # Convert to dBZ
    return 10 * tf_log10(200*r**(8/5)+1)

def dbz_to_r(dbz):
    '''
    Convert dbz to mm/h
    '''
    r = ((10**(dbz/10)-1)/200)**(5/8)
    return r

def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def minmax(x, norm_method='minmax', convert_to_dbz = False, undo = False):
    '''
    Performs minmax scaling to scale the images to range of 0 to 1.
    norm_method: 'minmax' or 'minmax_tanh'. If tanh is used than scale to -1 to 1 as tanh
                is used for activation function generator, else scale values to be between 0 and 1
    '''
    assert norm_method == 'minmax' or norm_method == 'minmax_tanh'
    
    # define max intensity as 100mm
    MIN = 0
    MAX = 128
    
    if not undo:
        if convert_to_dbz:
            MAX = 55
            x = r_to_dbz(x)
        # Set values over 100mm/h to 100mm/h
        x = tf.clip_by_value(x, MIN, MAX)
        if norm_method == 'minmax_tanh':
            x = (x - MIN - MAX/2)/(MAX/2 - MIN) 
        else:
            x = (x - MIN)/(MAX- MIN)
    else:
        if convert_to_dbz:
            MAX = 55
        if norm_method == 'minmax_tanh':
            x = x*(MAX/2 - MIN) + MIN + MAX/2
        else:
            x = x*(MAX - MIN) + MIN           
    return x

def pad_along_axis( x, pad_size = 3, axis = 2):

  '''
    Pad input to be divisible by 2.
    height of 765 to 768
        '''
  if pad_size <= 0:
    return x

  npad = [(0, 0)] * x.ndim
  npad[axis] = (0, pad_size)

  return tf.pad(x, paddings=npad, constant_values=0)






def get_data_as_xarray(data_folder=DATAFILE,start_index,end_index,start_time_frame=None):
    '''Code by Simon De Kock <simon.de.kock@vub.be>
    Generate the input to be trained by the model based in the file folder
   
    
    Parameters
    ----------
    data_folder : String
        contains path to the hdf files.
    start_time_frame: string
         contains the time of the frame from which you wish to start
         e.g(202311062000500)
         that is '%Year%month%day%Hour%Minute%Second'

    
    Returns
    -------
    X_array : xr.Data_array
    '''
    fns = None
    # A slice of the files was selected to produce nowcasts with DGMR and LDCast
    # Such that those nowcast start as close as possible to the startime of the PySTEPS and INCA nowcasts
    fne=None
    for i, filename in enumerate(sorted(os.listdir(data_folder))):
     
      time_str = os.path.splitext(os.path.basename(filename))[0].split('.', 1)[0]
        
      if filename.endswith('.hdf') and start_time_frame==time_str:
          fns=[f"{data_folder}/{file_name}" for file_name in sorted(os.listdir(data_folder))[i:i+5]]
      elif start_time_frame==None   and filename.endswith('.hdf'):
        fns=[f"{data_folder}/{file_name}" for file_name in sorted(os.listdir(data_folder))[start_index:end_index]]
    
    dataset = []
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
        
        lon, lat = pr(x_2d.flatten(), y_2d.flatten(), inverse=True)
        lon = lon.reshape(gridspec.get('ysize', 0), gridspec.get('xsize', 0))
        lat = lat.reshape(gridspec.get('ysize', 0), gridspec.get('xsize', 0))
        
        # Build the xarray dataset
        ds = xr.Dataset(
            data_vars={
                short_name: (['x', 'y'], file_content.get("dataset1/data1/data", np.nan),
                            {'long_name': long_name, 'units': units})
            },
            coords={
                'x': (['x'], x, {'axis': 'X', 'standard_name': 'projection_x_coordinate',
                                'long_name': 'x-coordinate in Cartesian system', 'units': 'm'}),
                'y': (['y'], y, {'axis': 'Y', 'standard_name': 'projection_y_coordinate',
                                'long_name': 'y-coordinate in Cartesian system', 'units': 'm'}),
                'lon': (['y', 'x'], lon, {'standard_name': 'longitude', 'long_name': 'longitude coordinate',
                                        'units': 'degrees_east'}),
                'lat': (['y', 'x'], lat, {'standard_name': 'latitude', 'long_name': 'latitude coordinate',
                                        'units': 'degrees_north'})
            }
        )
        ds['time'] = time

        # Append the dataset to the list
        dataset.append(ds)
        
    # Concatenate datasets along the time dimension
    dataset = xr.concat(dataset, dim='time')
    final_dataset=dataset.t

    return final_dataset




def get_input_array(field,downscale256=True) -> np.ndarray:
    '''
    Parameters
    ----------
    field : xr.DataArray

    

    
    Returns
    -------
    tensor : np.ndarray
    
    - Crop xarray data to required dimensions (700x700 to 256x256)
    - Reshape it to:
        [B, T, C, H, W] - Batch, Time, Channel, Heigh, Width
    args:
        - field: xarray.DataArray
            The precipitation data variable from the xarray
    '''
    arrays= [np.array(path) for path in field['precip_intensity']]

   
    preprocessed_data=[]

    for array in arrays:
      array[np.isnan(array)]=0
      # Sometimes 255 or other number (244) is used for the calibration
      # for out of image values, so also check the first pixel
      mask = np.where(arrays[0] == 65535, 1, 0)
      array[array == mask] = 0
      array = (array / 100) * 12
      x = minmax(array, norm_method='minmax', convert_to_dbz = True, undo = False)
      x = np.expand_dims(x, axis=-1)
      if downscale256:
        # First make the images square size
        x = pad_along_axis(x, axis=0, pad_size=3)
        x = pad_along_axis(x, axis=1, pad_size=68)
        x =  tf.image.resize(x, (256, 256))
        preprocessed_data.append(x)
    array=np.stack(preprocessed_data)
    tensor = torch.from_numpy(array)
    return tensor















