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

with open(os.path.join('..', 'README.rst'), 'r') as f:
    __doc__ = f.read().replace(_remove_from_doc, '')
