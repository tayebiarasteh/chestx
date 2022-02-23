"""
Created on November 10, 2019
functions for writing/reading data to/from disk

@modified_by: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
"""
import yaml
import numpy as np
import os
import warnings
import shutil
import pdb




def read_config(config_path):
    """Reads config file in yaml format into a dictionary

    Parameters
    ----------
    config_path: str
        Path to the config file in yaml format

    Returns
    -------
    config dictionary
    """

    with open(config_path, 'rb') as yaml_file:
        return yaml.safe_load(yaml_file)


def write_config(params, cfg_path, sort_keys=False):
    with open(cfg_path, 'w') as f:
        yaml.dump(params, f)

