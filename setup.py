from setuptools import setup, find_packages

setup(
        name = 'econpizza',
        version = '0.0.1',
        author='Gregor Boehl',
        author_email='admin@gregorboehl.com',
        description='Solve nonlinear perfect foresight models',
        packages = find_packages(),
        package_data={'pydsge': ["examples/*"]},
        install_requires=[
            'numba',
            'numpy',
            'scipy',
         ],
   )
