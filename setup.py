from setuptools import setup, find_packages

setup(
    name='avfet_modules',
    version='0.1',
    description="Modules for the code associated with the paper 'High-order conservative and accurately dissipative numerical integrators via auxiliary variables'",
    long_description=open('README.txt').read(),
    author='Boris D. Andrews, Patrick E. Farrell',
    packages=find_packages(),
    install_requires=[],
)

