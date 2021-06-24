from setuptools import setup, find_packages

setup(
    name="configuration-interaction",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "numba",
        "scipy",
        "quantum-systems @ git+https://github.com/Schoyen/quantum-systems.git",
    ],
)
