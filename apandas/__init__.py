#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from .version import VERSION as __version__
from .acolumn import AFunction, ANamedFunction, AColumn
from .aframe import AFrame, ASeries

__all__ = ['AFunction', 'ANamedFunction', 'AColumn', 'AFrame', 'ASeries']
__author__ = 'Tomas Protivinsky'

_remove_from_doc = """
APandas: Lightweight wrapper to support custom analytics in Pandas
==================================================================
"""

# Get the current file's directory
_here = os.path.abspath(os.path.dirname(__file__))
# And path to README.rst
_readme_path = os.path.join(_here, '..', 'README.rst')

# Construct the path to the README file in the parent directory
with open(_readme_path, 'r') as f:
    __doc__ = f.read().replace(_remove_from_doc, '')

