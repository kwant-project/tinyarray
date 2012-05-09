#!/usr/bin/env python

from distutils.core import setup, Extension

module = Extension('tinyarray',
                   language = 'c++',
                   extra_compile_args = ['--std=c++0x'],
                   sources = ['src/main.cc', 'src/array.cc',
                              'src/functions.cc'])

setup (name = 'tinyarray',
       version = '0.0',
       ext_modules = [module])
