# -*- coding: utf-8 -*-
"""
This is a deep learning model for performing nowcasting on radar images.
"""

# Import the needed libraries
import numpy as np
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

### Uncomment the next lines if pyproj is needed for the importer.
# try:
#     import pyproj
#
#     PYPROJ_IMPORTED = True
# except ImportError:
#     PYPROJ_IMPORTED = False

from pysteps.decorators import postprocess_import


# Function importer_dgmr_xxx to import XXX-format
# files from the ABC institution

# IMPORTANT: The name of the importer should follow the "importer_institution_format"
# naming convention, where "institution" is the acronym or short-name of the
# institution. The "importer_" prefix to the importer name is MANDATORY since it is
# used by the pysteps interface.
#
# Check the pysteps documentation for examples of importers names that follow this
# convention:
# https://pysteps.readthedocs.io/en/latest/pysteps_reference/io.html#available-importers
#
# The function prototype for the importer's declaration should have the following form:
#
#  @postprocess_import()
#  def import_institution_format(filename, keyword1="some_keyword", keyword2=10, **kwargs):
#
# The @postprocess_import operator
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The `pysteps.io.importers` module uses the `postprocess_import` decorator to easily
# define the default data type and default value used for the missing/invalid data.
# The allowed postprocessing operations are
#   - Data type casting using the `dtype` keyword.
#   - Set invalid or missing data to a predefined value using the `fillna` keyword.
# The @postprocess_import decorator should always be added immediately above the
# importer definition to maintain full compatibility with the pysteps library.
# Adding the decorator @add_postprocess_keywords() without any keywords will ensure that
# the precipitation data returned by the importer has double precision, and the
# invalid/missing data is set to `np.nan`.
# For more information on the postprocessing decorator, see:
# https://pysteps.readthedocs.io/en/latest/generated/pysteps.decorators.postprocess_import.html
#
# Function arguments
# ~~~~~~~~~~~~~~~~~~
#
# The function arguments should have the following form:
# (filename, keyword1="some_keyword", keyword2=10,...,keywordN="something", **kwargs)
# The `filename` and `**kwargs` arguments are mandatory to comply with the pysteps
# interface. To fine-control the behavior of the importer, additional keywords can be
# added to the function.
# For example: keyword1="some_keyword", keyword2=10, ..., keywordN="something"
# It is recommended to declare the keywords explicitly in the function to improve the
# readability.
#
#
# Return arguments
# ~~~~~~~~~~~~~~~~
#
# The importer should always return the following fields:
#
# precipitation : 2D array (ndarray or MaskedArray)
#     Precipitation field in mm/h. The dimensions are [latitude, longitude].
# quality : 2D array or None
#     If no quality information is available, set to None.
# metadata : dict
#     Associated metadata (pixel sizes, map projections, etc.).
#
#


@postprocess_import()
def importer_dgmr_load_model(filename,size,TFHUB_BASE_PATH,**kwargs):
    """
    
    Load the DGMR pre-trained model from a google storage (with tensorflow_hub).
    

    Parameters
    ----------
    size : tupple
        contains the height and the width of the model.
    path: string
        Contains the path where the model is saved

    
    Returns
    -------
    pretrained model.
    """

    ### Uncomment the next lines if pyproj is needed for the importer
    # if not PYPROJ_IMPORTED:
    #     raise MissingOptionalDependency(
    #         "pyproj package is required by importer_dgmr_xxx
    #         "but it is not installed"
    #     )
    # The hieght and the width indicates the model to be loaded.
    # This means that size of the input frames the model will take
    input_height=size[0]
    input_width=size[1]
    ####################################################################################
    hub_module = hub.load(
      os.path.join(TFHUB_BASE_PATH, f"{input_height}x{input_width}"))
  # Note this has loaded a legacy TF1 model for running under TF2 eager mode.
  # This means we need to access the module via the "signatures" attribute. See
  # https://github.com/tensorflow/hub/blob/master/docs/migration_tf2.md#using-lower-level-apis
  # for more information.
    return hub_module.signatures['default']

@postprocess_import()
def importer_dgmr_predict(filename,module, input_frames, num_samples=1,
            include_input_frames_in_result=False,**kwargs):
  """
    
    Load the DGMR pre-trained model from a google storage (with tensorflow_hub).
    

    Parameters
    ----------
    size : tupple
        contains the height and the width of the model.
    path: string
        Contains the path where the model is saved

    
    Returns
    -------
    A tensor of shape (num_samples,T_out,H,W,C), where T_out is either 18 or 22
    as described above.
  
  Make predictions from a TF-Hub snapshot of the 'Generative Method' model.

  Args:
    module: One of the raw TF-Hub modules returned by load_module above.
    input_frames: Shape (T_in,H,W,C), where T_in = 4. Input frames to condition
      the predictions on.
    num_samples: The number of different samples to draw.
    include_input_frames_in_result: If True, will return a total of 22 frames
      along the time axis, the 4 input frames followed by 18 predicted frames.
      Otherwise will only return the 18 predicted frames.


    
  """
  NUM_INPUT_FRAMES = 4
  input_frames = tf.math.maximum(input_frames, 0.)
  # Add a batch dimension and tile along it to create a copy of the input for
  # each sample:
  input_frames = tf.expand_dims(input_frames, 0)
  input_frames = tf.tile(input_frames, multiples=[num_samples, 1, 1, 1, 1])

  # Sample the latent vector z for each sample:
  _, input_signature = module.structured_input_signature
  z_size = input_signature['z'].shape[1]
  z_samples = tf.random.normal(shape=(num_samples, z_size))

  inputs = {
      "z": z_samples,
      "labels$onehot" : tf.ones(shape=(num_samples, 1)),
      "labels$cond_frames" : input_frames
  }
  samples = module(**inputs)['default']
  if not include_input_frames_in_result:
    # The module returns the input frames alongside its sampled predictions, we
    # slice out just the predictions:
    samples = samples[:, NUM_INPUT_FRAMES:, ...]

  # Take positive values of rainfall only.
  samples = tf.math.maximum(samples, 0.)
  return samples
