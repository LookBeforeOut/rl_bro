from setuptools import setup, find_packages

setup(
    name="vehicle_drift",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "mujoco>=2.3.0",
        "pyyaml>=6.0",
        "gym>=0.21.0",
        "matplotlib>=3.5.0",
        "tensorboard>=2.8.0"
    ],
) 