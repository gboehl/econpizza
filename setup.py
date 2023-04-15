from setuptools import setup, find_packages
from os import path

# get version from dedicated version file
version = {}
with open("econpizza/__version__.py") as fp:
    exec(fp.read(), version)

# read the contents of the README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/gboehl/econpizza",
    name='econpizza',
    version=version['__version__'],
    author="Gregor Boehl",
    author_email="admin@gregorboehl.com",
    description="Solve nonlinear perfect foresight models with heterogeneous agents",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    include_package_data=True,
    packages=['econpizza', 'econpizza.parser',
              'econpizza.utilities', 'econpizza.solvers'],
    package_data={"econpizza": ["examples/*"]},
    extras_require={
        'linear': ['grgrlib>=0.1.22'],
    },
    install_requires=[
        "jax<0.4.6",
        "jaxlib<0.4.6",
        "grgrjax>=0.4.3",
        "pyyaml",
        "scipy",
    ],
)
