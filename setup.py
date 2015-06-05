import os
from setuptools import setup, find_packages
from pip.req import parse_requirements
import uuid

setup(
    name = "theano_class",
    version = "0.0.1",
    author = "Sharad Vikram",
    author_email = "sharad.vikram@gmail.com",
    description = "",
    license = "MIT",
    keywords = "",
    url = "",
    packages=find_packages(include=[
        'theano_class'
    ]),
    long_description="",
    classifiers=[
    ],
)
