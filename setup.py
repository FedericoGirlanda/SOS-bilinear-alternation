from setuptools import setup, find_packages

setup(
    name='SosBilinearAlternation',
    author='Federico Girlanda',
    version='1.0.0',
    url="https://github.com/FedericoGirlanda/SOS-bilinear-alternation",
    packages=find_packages(),
    install_requires=[
        # general
        'numpy',
        'matplotlib',
        'scipy',
        'ipykernel',

        # optimal control
        'drake',
    ],
    classifiers=[
          'Development Status :: 5 - Stable',
          'Environment :: Console',
          'Intended Audience :: Academic Usage',
          'Programming Language :: Python',
          ],
)