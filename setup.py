from setuptools import setup

# Get the long description from the README file
with open('README.md') as fp:
    long_description = fp.read()

setup(
    name='auto_reduce',
    version='0.1',
    author='Ayush Pandey',
    author_email='apandey@caltech.edu',
    url='https://github.com/ayush9pandey/autoReduce/',
    description='An automated model reduction toolbox',
    long_description=long_description,
    packages=['auto_reduce'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: POSIX',
        'Operating System :: Unix'
        'Operating System :: MacOS'
    ],
    install_requires=[
          'sympy',
          'scipy',
          'numpy',
          'nose',
      ],
    test_suite='nose.collector',
    tests_require=['nose'],
)
