from setuptools import setup, find_packages
from aisier.version import __version__, __author__, __license__

desc = 'aisier, Tensorflow project management made easier'

required = []
with open('requirements.txt') as fp:
    for line in fp:
        line = line.strip()
        if line != "":
            required.append(line)

setup(name='aisier',
      version=__version__,
      description=desc,
      long_description=desc,
      long_description_content_type='text/plain',
      author=__author__,
      url='http://www.github.com/pagiux/aisier',
      packages=find_packages(),
      install_requires=required,
      scripts=['bin/aisier'],
      license=__license__,
      )
