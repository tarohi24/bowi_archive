#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

requirements = [ ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Wataru Hirota",
    author_email='hirota@whiro.me',
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
    description="BoW Improved",
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    keywords='bowi',
    name='bowi',
    packages=find_packages(include=['bowi', 'bowi.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/tarohi24/bowi',
    version='0.1.0',
    zip_safe=False,
)
