#from distutils.core import setup
#from Cython.Build import cythonize
try:
    from setuptools import setup
    from setuptools import Extension
    from Cython.Build import cythonize
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    
setup(
  name = 'corners',
  ext_modules = cythonize("corners.pyx"),
)