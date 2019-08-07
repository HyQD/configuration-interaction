Time-evolution computations
===========================
The time-evolution of the configuration interaction method is done by computing
the ground state at a specified truncation level, and choosing initial
conditions for the coefficient vector :math:`\mathbf{c}`.
The default choice is the ground state coefficient vector found from the ground
state computations, but other choices are valid as well as long as the
normalization of the coefficient vector is the same as for the eigenstates.

Time-dependent configuration interaction super class
----------------------------------------------------
Similarly to the ground state computations, the truncation level of the Slater
determinants decides which time-dependent configuration interaction class to
use.
However, this time the truncation level of the class decides which ground state
solver to use, whereas the time-dependent part only requires an initial state
represented as a coefficient vector :math:`\mathbf{c}`.

.. autoclass:: configuration_interaction.tdci.TimeDependentConfigurationInteraction
    :members:
    :special-members: __call__

Creating truncated TDCI-classes
-------------------------------
.. autofunction:: configuration_interaction.tdci.get_tdci_class
