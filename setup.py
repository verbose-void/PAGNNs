#!/usr/bin/env python

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='pagnn',
      version='0.1',
      description='Persistent Artificial Graph-based Neural Networks (PAGNNs)',
      author='Dyllan McCreary',
      author_email='dyllanmccreary@protonmail.com',
      packages=find_packages(),
      install_requires=requirements)
