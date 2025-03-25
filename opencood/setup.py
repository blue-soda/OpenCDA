# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from os.path import dirname, realpath
from setuptools import setup, find_packages, Distribution
from opencood.version import __version__
from Cython.Build import cythonize
from setuptools import Extension
from distutils.command.build_ext import build_ext

def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]

import numpy as np
extensions = [
    Extension(
        "opencood.utils.box_overlaps",
        sources=["opencood/utils/box_overlaps.pyx"],
        include_dirs=[np.get_include()],  # 自动获取numpy头文件路径
        extra_compile_args=[],
    )
]


setup(
    name='OpenCOOD',
    version=__version__,
    packages=find_packages(),
    url='https://github.com/ucla-mobility/OpenCDA.git',
    license='MIT',
    author='Runsheng Xu, Hao Xiang',
    author_email='rxx3386@ucla.edu',
    description='An opensource pytorch framework for autonomous driving '
                'cooperative detection',
    long_description = open("README.md", encoding="utf-8").read(),
    install_requires=_read_requirements_file(),

    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': build_ext},
)
