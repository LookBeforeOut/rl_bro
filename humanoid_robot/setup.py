from setuptools import setup, find_packages

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