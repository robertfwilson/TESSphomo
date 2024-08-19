from setuptools import setup
from os import path

# If Python3: Add "README.md" to setup. 
# Useful for PyPI (pip install wotan). Irrelevant for users using Python2
try:
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except:
    long_description = ' '

setup(
    name='TESSphomo',
    version='0.2',
    description='TESS PHOtomoeter MOdeler',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/robertfwilson/TESSphomo',
    author='Robby Wilson',
    author_email='robert.f.wilson@nasa.gov',
    license='MIT',
    zip_safe=False,
    packages=['tessphomo'],
    install_requires=[
        'TESS_PRF',
        'lightkurve'
        ]
)
