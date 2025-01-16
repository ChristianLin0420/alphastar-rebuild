from setuptools import setup, find_packages

setup(
    name="alphastar",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pytest>=7.3.0",
        "pytest-cov>=4.1.0",
        "wandb>=0.15.0",
        "pyyaml>=6.0",
        "protobuf<=3.20.0",
        "pysc2>=3.0.0",
    ],
    python_requires=">=3.8",
) 