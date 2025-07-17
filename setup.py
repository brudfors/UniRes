from setuptools import (setup, find_packages)


setup(
    author='Mikael Brudfors',
    author_email='brudfors@gmail.com',
    description='UniRes: Unified Super-Resolution of Medical Imaging Data',    
    entry_points={'console_scripts': ['unires=unires._cli:run']},
    install_requires=[
        "numpy==1.26.0",
        "nitorch[all]@git+https://github.com/balbasty/nitorch#8067d60542642a39ab6c6eb5e1157373a9d3dcc3",        
    ],
    name='unires',    
    packages=find_packages(),
    python_requires='>=3.10',
    url='https://github.com/brudfors/UniRes',
    version='0.3',        
)