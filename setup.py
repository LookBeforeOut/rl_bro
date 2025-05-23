from setuptools import setup, find_packages

# setup(
#     name="vehicle_drift",
#     version="0.1.0",
#     packages=find_packages(),
#     install_requires=[
#         "numpy>=1.21.0",
#         "torch>=1.9.0",
#         "mujoco>=2.3.0",
#         "pyyaml>=6.0",
#         "gym>=0.21.0",
#         "matplotlib>=3.5.0",
#         "tensorboard>=2.8.0"
#     ],
# ) 


setup(
    name="humanoid_robot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "gymnasium",
        "mujoco",
        "stable-baselines3",
        "matplotlib",
        "pandas",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A reinforcement learning project for humanoid robot control",
    keywords="reinforcement learning, robotics, humanoid",
    python_requires=">=3.8",
) 