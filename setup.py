#!/usr/bin/env python3

from distutils.core import setup, Extension
import numpy.distutils.misc_util

module1 = Extension('C_array',
                    sources = ['C_array.c'])

setup (name = '_C_array',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [
       		Extension('C_array', sources = ['C_array.c'], 
       					include_dirs=[numpy.get_include()]),
       	],
)