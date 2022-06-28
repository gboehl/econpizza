from setuptools import setup, find_packages
from os import path
from econpizza import __version__

# read the contents of the README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/gboehl/econpizza",
    name='econpizza',
    version=__version__,
    author="Gregor Boehl",
    author_email="admin@gregorboehl.com",
    description="Solve nonlinear perfect foresight models",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    include_package_data=True,
    packages=['econpizza', 'econpizza.parser', 'econpizza.utilities'],
    package_data={"econpizza": ["examples/*"]},
    install_requires=[
        "numpy",
        "scipy",
        "jax",
        "jaxlib",
        "grgrlib>=0.1.9",
        "pyyaml",
    ],
)
