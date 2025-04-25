from setuptools import setup, find_packages

setup(
    name="nonSphericalJeansData", 
    version="0.1.0",          
    author="Andrew Robertson",       
    author_email="arobertson@carnegiescience.edu", 
    description="Generate data from the EAGLE 50 Mpc SIDM sims (from https://arxiv.org/abs/2009.07844) that can be used to test and calibrate a non-spherical Jeans model",
    long_description=open('README.md').read(),  # Include the content of your README file
    long_description_content_type="text/markdown",
    url="https://github.com/Andrew-Robertson/nonSphericalJeansData",  # Replace with the URL of your project
    packages=find_packages(),  # Automatically find packages in the project
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify the Python version compatibility
    install_requires=[
       'numpy', 'matplotlib', 'h5py', 'astropy', 'scipy'
    ],
)