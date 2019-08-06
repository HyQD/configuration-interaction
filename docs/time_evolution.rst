Time-evolution computations
===========================

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
