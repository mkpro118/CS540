from distutils.core import setup, Extension

module = Extension(
    'transition_probability_table',
    sources=['transition_probability_table.c']
)

setup(
    name='transition_probability_table',
    author='M K',
    description='Provides functions to build Transition Probability Tables',
    ext_modules=[module]
)
