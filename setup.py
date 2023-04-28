from setuptools import (setup, find_packages)


setup(
    author='Mikael Brudfors',
    author_email='brudfors@gmail.com',
    description='UniRes: Unified Super-Resolution of Medical Imaging Data',    
    entry_points={'console_scripts': ['unires=unires._cli:run']},
    install_requires=[
        "nitorch[all]@git+https://github.com/balbasty/nitorch#2c7739ab856f07148be310f00d6690ee9378c8cf",
    ],
    name='unires',    
    packages=find_packages(),
    python_requires='>=3.8',
    url='https://github.com/brudfors/UniRes',
    version='0.1',        
)
