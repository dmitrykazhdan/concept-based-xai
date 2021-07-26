from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

version = '0.1.0'
setup(
    name='concepts_xai',
    version=version,
    packages=find_packages(),
    description='Concept Extraction Comparison',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

'''
VERSIONING:

1.2.0.dev1  # Development release
1.2.0a1     # Alpha Release
1.2.0b1     # Beta Release
1.2.0rc1    # Release Candidate
1.2.0       # Final Release
1.2.0.post1 # Post Release
15.10       # Date based release
23          # Serial release




    MAJOR version when they make incompatible API changes,

    MINOR version when they add functionality in a backwards-compatible manner, and

    MAINTENANCE version when they make backwards-compatible bug fixes.

'''
