from setuptools import setup
from ctnet import __version__

setup(
    name='ctnet',
    version=__version__,
    long_description="",
    packages=[
        "ctnet",
        "ctnet.infer",
        "ctnet.modules",
        "ctnet.trainer",
    ],
    include_package_data=True,
    url='https://github.com/JeanMaximilienCadic/ctnet',
    license='MIT',
    author='Jean Maximilien Cadic',
    python_requires='>=3.6',
    install_requires=[d.rsplit()[0] for d in open("requirements.txt").readlines()],
    author_email='support@cadic.jp',
    description='GNU Tools for python',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
    ]
)

