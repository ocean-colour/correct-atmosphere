
# Standard imports
import glob, os
from setuptools import setup, find_packages


# Begin setup
setup_keywords = dict()
setup_keywords['name'] = 'correct-atmosphere'
setup_keywords['description'] = 'Python package for ocean color oriented atmospheric corrections' 
setup_keywords['author'] = 'J. Xavier Prochaska'
setup_keywords['author_email'] = 'jxp@ucsc.edu'
setup_keywords['license'] = 'BSD'
setup_keywords['url'] = 'https://github.com/ocean-colour/correct-atmosphere'
setup_keywords['version'] = '0.0.dev0'
# Use README.rst as long_description.
setup_keywords['long_description'] = ''
if os.path.exists('README.md'):
    with open('README.md') as readme:
        setup_keywords['long_description'] = readme.read()
setup_keywords['python_requires'] = '>=3.11'
setup_keywords['install_requires'] = [
    'seaborn', 'smart-open[s3]', 
    'scikit-learn', 'scikit-image', 'tqdm', 
    'healpy', 'cftime', 'bokeh', 'umap-learn', 'llvmlite', 'boto3',
    'xarray', 'h5netcdf', 'emcee', 'corner',
    'importlib-metadata', 'timm==0.3.2', 'IPython',
    'scikit-learn', 'scikit-image', 'tqdm',
    'pysolar']
setup_keywords['extras_require'] = {
    'dev': ['pytest', 'pytest-runner'],
}
setup_keywords['zip_safe'] = False
setup_keywords['packages'] = find_packages()
setup_keywords['setup_requires'] = ['pytest-runner']
setup_keywords['tests_require'] = ['pytest']

if os.path.isdir('bin'):
    setup_keywords['scripts'] = [fname for fname in glob.glob(os.path.join('bin', '*'))
                                 if not os.path.basename(fname).endswith('.rst')]

setup(**setup_keywords)
