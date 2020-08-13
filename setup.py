from setuptools import setup

# Get the long description from the README file
with open('README.md') as fp:
    long_description = fp.read()

setup(
    name='AutoReduce',
    version='0.1.0',
    author='Ayush Pandey',
    url='https://github.com/ayush9pandey/AutoReduce/',
    description='Python based automated model reduction tool for SBML models'
    long_description=long_description,
    packages=['autoreduce'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
    ],
    install_requires=[
          "python-libsbml",
          "sympy",
          "scipy",
          "numpy"
          ],
    extras_require = { 
        "all": [
            "matplotlib",
            "seaborn",
            ]
            },
    setup_requires=["pytest-runner"],
    python_requires='>=3.6',
    keywords="SBML Automated Model Reduction Modeling QSSA Hill functions"
    tests_require=["pytest", "pytest-cov", "nbval"],
    project_urls={
    'Documentation': 'https://readthedocs.org/projects/AutoReduce/',
    'Funding': 'http://www.cds.caltech.edu/~murray/wiki/index.php?title=Developing_Standardized_Cell-Free_Platforms_for_Rapid_Prototyping_of_Synthetic_Biology_Circuits_and_Pathways',
    'Source': 'https://github.com/ayush9pandey/AutoReduce',
    'Tracker': 'https://github.com/ayush9pandey/AutoReduce/issues',
    }, 
)
