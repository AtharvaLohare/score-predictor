from setuptools import find_packages, setup

from typing import List

def get_requirements(file_path:str)->List[str]:
    """ This function will return the list of required packages."""

setup(
    name='project1',
    version='0.0.1',
    auther='Atharva Lohare',
    author_email='atharvlohare@gmail.com',
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'seaborn']
)