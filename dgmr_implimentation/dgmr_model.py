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
from params import *
from dgmr import DGMR
import os
from torch import load
import urllib
from tqdm import tqdm

### Uncomment the next lines if pyproj is needed for the importer.
# try:
#     import pyproj
#
#     PYPROJ_IMPORTED = True
# except ImportError:
#     PYPROJ_IMPORTED = False





def load_model(input_height,input_width):
    """
    Load the DGMR pre-trained model from a google storage (with tensorflow_hub).
    """
    print("--> Loading model...")
    TFHUB_BASE_PATH=r"C:\Users\user\Desktop\Intersnhip documents\tfhub_snapshots"
    hub_module = hub.load(
      os.path.join(TFHUB_BASE_PATH, f"{input_height}x{input_width}"))
  # Note this has loaded a legacy TF1 model for running under TF2 eager mode.
  # This means we need to access the module via the "signatures" attribute. See
  # https://github.com/tensorflow/hub/blob/master/docs/migration_tf2.md#using-lower-level-apis
  # for more information.
    return hub_module.signatures['default']



  
 

   


def predict(module, input_frames, num_samples=1,
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
