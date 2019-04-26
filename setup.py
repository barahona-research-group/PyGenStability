#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup

import numpy as np

setup(
        name = 'PyGenStability',
        version = '1.0',
        include_dirs = [np.get_include()], #Add Include path of numpy
        packages=['.'],
      )
