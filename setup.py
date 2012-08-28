#!/usr/bin/env python

from distutils.core import setup, Extension

module = Extension('tinyarray',
                   language = 'c++',
                   extra_compile_args = ['--std=c++0x'],
                   sources = ['src/array.cc', 'src/functions.cc',
                              'src/arithmetic.cc'])

setup (name = 'tinyarray',
       version = '0.0',
       ext_modules = [module])
