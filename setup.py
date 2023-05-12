from setuptools import find_packages, setup

setup(
    name='streaming_indicators',
    packages=find_packages(include=['streaming_indicators']),
    version='0.1.1',
    description='A python library for computing technical analysis indicators on streaming data.',
    author='rishabg1997@gmail.com',
    license='MIT',
    # install_requires=['numpy']
)