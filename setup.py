#!/usr/bin/env python
# -*- coding: utf-8; fill-column: 120 -*-
#
# Copyright (C) 2020 Jochen Küpper <jochen.kuepper@cfel.de>

from setuptools import setup

copyright = 'Copyright (C) 2020 Jochen Küpper <jochen.kuepper@cfel.de>'
name = "CMI Python-project template"
version = "0.2.dev0"
release = version
long_description = """CMI Python template

This is the installation and general build file of the CMI Python project template.

Original author:    Jochen Küpper <jochen.kuepper@cfel.de>
Current maintainer: Jochen Küpper <jochen.kuepper@cfel.de>
"""


setup(name=name,
      python_requires     = '>=3.6',
      author              = "Jochen Küpper and the CFEL-CMI group",
      author_email        = "jochen.kuepper@cfel.de",
      maintainer          = "Jochen Küpper and the CFEL-CMI group",
      maintainer_email    = "jochen.kuepper@cfel.de",
      url                 = "https://github.com/CFEL-CMI/CMI-Python-project-template",
      description         = "CMI Python-software template",
      version             = version,
      long_description    = long_description,
      license             = "GPL",
      packages            = ['flowbasis', 'scripts'],
      #scripts             = ['scripts/cmitemplate_calc'],
      command_options={
          'build_sphinx': {
              'project': ('setup.py', name),
              'version': ('setup.py', version),
              'release': ('setup.py', release),
              'source_dir': ('setup.py', 'doc'),
              'copyright': ('setup.py', copyright)}
      },
      )
