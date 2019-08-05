Getting started
===============
This project is written in pure Python and can therefore be installed using
common package managers.
Note that we not yet released this project to PyPi, and the installation must
therefore be done via Github.

Installation
------------

The recommended way to install this project as of now is by using ``pipenv``. Run::

	pipenv install -e git+https://github.com/Schoyen/quantum-systems.git#egg=configuration-interaction

This will install the project with all dependencies.

Using pip
------------

The installation can also be done using ``pip``::

    pip install git+https://github.com/Schoyen/quantum-systems.git

Alternatively, the same task can be accomplished using three commands::

    git clone https://github.com/Schoeyn/quantum-system.git
    cd quantum-systems
    pip install .

This downloads the repository and installs directly from the ``setup.py``-file.
In order to update to the latest version use::

    pip install -U git+https://github.com/Schoyen/quantum-systems.git

or, whilst inside the cloned repo::

    pip install -U .

Conda Environment
-----------------

Due to some of the optional dependencies in ``quantum-systems``, it can be
useful to set up a conda environment.
We have included an environment specification file for this purpose::

    conda environment create -f environment.yml
    conda activate quantum-systems

Deactivating the ``conda`` environment is done with::

    conda deactivate

The environment can be updated with::

    conda env update -f environment.yml
