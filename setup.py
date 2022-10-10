#!/usr/bin/env python
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2020 Jochen Küpper <jochen.kuepper@cfel.de>

from setuptools import setup, find_packages

copyright = 'Copyright (C) 2022 Yahya Saleh <yahya.saleh@cfel.de> and Jochen Küpper <jochen.kuepper@cfel.de>'
name = "FlowBasis"
version = "0.1.dev0"
release = version
long_description = """FlowBasis

This is the installation and general build file of the CMI FlowBasis code. The code demonstrates how to augmented basis sets and use them to compute eigenpairs of Schr�dinger equations.

Author:    Yahya Saleh <yahya.saleh@cfel.de>
Current maintainer: Yahya Saleh <yahya.saleh@cfel.de>
"""

classifiers = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: POSIX :: Linux',
    'Operating System :: Unix',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
]

with open('requirements.txt') as of:
    install_requires = of.read().splitlines()

setup(name=name,
      python_requires     = '>=3.7',
      author              = "Yahya Saleh and the CFEL-CMI group",
      author_email        = "yahya.saleh@cfel.de",
      maintainer          = "Yahya Saleh and the CFEL-CMI group",
      maintainer_email    = "yahya.saleh@cfel.de",
      url                 = "https://github.com/CFEL-CMI/CMI-Python-project-template",
      description         = "CMI FlowBasis",
      version             = version,
      long_description    = long_description,
      license             = "GPL",
      packages            = ['flowbasis', 'scripts'],
      command_options={
          'build_sphinx': {
              'project': ('setup.py', name),
              'version': ('setup.py', version),
              'release': ('setup.py', release),
              'source_dir': ('setup.py', 'doc'),
              'copyright': ('setup.py', copyright)}
      },
      )
