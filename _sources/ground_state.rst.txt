Ground state calculations
=========================

Configuration interaction super class
-------------------------------------
In our implementation the truncation level of the Slater determinant basis
uniquely decides the configuration interaction class.
We've therefore created a single abstract super-class defining all methods
needed by the CI-solvers.

.. autoclass:: configuration_interaction.ci.ConfigurationInteraction
    :members:

To create a specific CI-solver a subclass of the
``ConfigurationInteraction``-class must be created setting the field
``excitations``.

.. _truncated-ci-classes:

Creating truncated CI-classes
-----------------------------
The library includes pre-defined classes for some of the more common truncation
levels.
These are:

* ``CIS``
* ``CID``
* ``CISD``
* ``CIDT``
* ``CISDT``
* ``CIDTQ``
* ``CISDTQ``

However, these classes and others can be created by the function
``get_ci_class`` listed below.

.. autofunction:: configuration_interaction.ci.get_ci_class

As an example, we demonstrate how to create a CISDTQ56-class using this
function::

    # Create the object class
    CISDTQ56 = get_ci_class("CISDTQ56")
    # Instantiate the object, and setup the Slater determinants
    cisdtq56 = CISDTQ56(system, verbose=True)
    # Compute the ground state of the system
    cisdtq56.compute_ground_state()
