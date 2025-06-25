# setup.py

from setuptools import setup, find_packages

setup(
    name='G1',                     # 包名字，可自定义
    version='0.1.0',
    description='Layered control for Unitree G1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.10',
        'osqp',
        'scipy',
        'isaacgym',
    ],
    entry_points={
        'console_scripts': [
            'run-g1=G1.main:main'
        ]
    },
    python_requires='>=3.7',
)

