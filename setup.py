from setuptools import setup

setup(
    name = "theanify",
    version = "0.1.14",
    author = "Sharad Vikram",
    author_email = "sharad.vikram@gmail.com",
    description = "Allows you to annotate instance methods to be compiled by Theano.",
    license = "MIT",
    keywords = "theano",
    url = "https://github.com/sharadmv/theanify",
    install_requires=['Theano==0.8.0beta'],
    dependency_links = [
        'git+git://github.com/Theano/Theano.git@0e6e9f5acaa2fd0deb427b3dad7b5a7611e5c8b7#egg=Theano-0.8.0beta'
    ],
    packages=[
        'theanify'
    ],
    classifiers=[
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',

     'License :: OSI Approved :: MIT License',

    'Programming Language :: Python :: 2.7',
],
)
