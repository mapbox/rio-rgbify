import os
import sys
from setuptools import setup, find_packages
from setuptools.extension import Extension

# Parse the version from the fiona module.
with open('rio_rgbify/__init__.py') as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            break

long_description = """"""


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='rio-rgbify',
      version=version,
      description=u"Encode arbitrary bit depth rasters in psuedo base-256 as RGB",
      long_description=long_description,
      classifiers=[],
      keywords='',
      author=u"Damon Burgett",
      author_email='damon@mapbox.com',
      url='https://github.com/mapbox/rio-rgbify',
      license='BSD',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=["click", "rasterio", "rio-mucho"],
      extras_require={
          'test': ['pytest', 'pytest-cov', 'codecov']},
      entry_points="""
      [rasterio.rio_plugins]
      toa=rio_rgbify.scripts.cli:rgbify
      """
      )
