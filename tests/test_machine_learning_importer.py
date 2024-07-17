#!/usr/bin/env python

"""Tests for `machine_learning_importer` package."""
from pysteps_importer_dgmr.importer_dgmr_generate_input import importer_dgmr_generate_input
def test_importers_discovery():
    """It is recommended to at least test that the importers provided by the plugin are
    correctly detected by pysteps. For this, the tests should be ran on the installed
    version of the plugin (and not against the plugin sources).
    """

    from pysteps.io import interface

    new_importers = ["importer_dgmr_input"]
    for importer in new_importers:
        assert importer.replace("import_", "") in interface._importer_methods


def test_importers_with_files():
    """Additionally, you can tests that your importers correctly reads the corresponding
    some example data.
    """
    

    # Write the test here.
    
    data=r"C:\Users\user\Desktop\meteo_france_data"
    f=importer_dgmr_generate_input(data)
    print(f)

test_importers_with_files()
