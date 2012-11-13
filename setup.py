#!/usr/bin/env python

from distutils.core import setup, Extension

module = Extension('tinyarray',
                   language='c++',
                   extra_compile_args=['--std=c++0x'],
                   sources=['src/arithmetic.cc', 'src/array.cc',
                            'src/functions.cc'],
                   depends=['src/arithmetic.hh', 'src/array.hh',
                            'src/conversion.hh', 'src/functions.hh'])

setup (name = 'tinyarray',
       version = '0.0',
       ext_modules = [module])
