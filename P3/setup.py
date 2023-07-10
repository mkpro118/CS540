from distutils.core import setup, Extension

module = Extension(
    'probability',
    sources=['probability.c']
)

setup(
    name='probability',
    author='M K',
    description='Provides functions to build Probability Tables',
    ext_modules=[module]
)
