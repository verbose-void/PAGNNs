#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='PAGNN',
      version='0.1',
      description='Persistent Artificial Graph-based Neural Networks (PAGNNs)',
      author='Dyllan McCreary',
      author_email='dyllanmccreary@protonmail.com',
      packages=['PAGNN'],
      install_requires=requirements
      )
