from setuptools import setup, find_packages

setup(
    name = "theanify",
    version = "0.0.1",
    author = "Sharad Vikram",
    author_email = "sharad.vikram@gmail.com",
    description = "Allows you to annotate instance methods to be compiled by Theano.",
    license = "MIT",
    keywords = "theano",
    url = "https://github.com/sharadmv/theanify",
    install_requires=['theano'],
    packages=find_packages(include=[
        'theanify'
    ]),
    classifiers=[
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',

     'License :: OSI Approved :: MIT License',

    'Programming Language :: Python :: 2.7',
],
)
