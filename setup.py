from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


ext_modules=[
    Extension(
        "stm.rmsd.qcp",
        sources=["stm/rmsd/qcp.pyx"],
        include_dirs=[np.get_include()]
        )
    ]
    
setup(
    name='structural-template-matching',
    version='1.0',
    description='Structural template matching',
    author='Jacob Madsen',
    author_email='jamad@fysik.dtu.dk',
    packages=find_packages(),  #same as name
    install_requires=[], #external packages as dependencies
    ext_modules=cythonize(ext_modules),
)
