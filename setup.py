from importlib.metadata import entry_points
from setuptools import setup, find_packages

setup(
    name='croac',
    version='0.0.1',
    description='Counting and Recognition using Omnidirectional Acoustic Capture (CROAC)',
    author='Marcel Gietzmann-Sanders',
    author_email='marcelsanders96@gmail.com',
    packages=find_packages(include=['croac', 'croac*']),
    install_requires=[
        'numpy',
        'pytest',
        'tqdm'
    ]
)