#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APandas: Lightweight wrapper to support custom analytics in Pandas
"""

from .version import VERSION as __version__
from .acolumn import AFunction, AColumn
from .aframe import AFrame

__all__ = ['AFunction', 'AColumn', 'AFrame']
__author__ = 'Tomas Protivinsky'
