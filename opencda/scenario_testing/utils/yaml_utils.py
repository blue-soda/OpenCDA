# -*- coding: utf-8 -*-
"""
Used to load and write yaml files
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: MIT

import re
import yaml
from datetime import datetime


def load_yaml(file):
    """
    Load yaml file and return a dictionary.
    Parameters
    ----------
    file : string
        yaml file path.

    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """

    stream = open(file, 'r')
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    param = yaml.load(stream, Loader=loader)

    # load current time for data dumping and evaluation
    current_time = datetime.now()
    current_time = current_time.strftime("_%Y_%m_%d_%H_%M_%S")

    param['current_time'] = current_time

    return param
