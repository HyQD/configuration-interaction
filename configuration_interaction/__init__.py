from configuration_interaction.ci import ConfigurationInteraction
from configuration_interaction.tdci import TimeDependentConfigurationInteraction


def excitation_string_handler(excitations):
    if isinstance(excitations, str):
        excitations = excitations.upper()

        if excitations.startswith("CI"):
            excitations = excitations[2:]

        excitations = [excitation for excitation in excitations]

    from configuration_interaction.ci_helper import ORDER

    for excitation in excitations:
        assert excitation in ORDER, f'"{excitation}" is not a supported order'

    return list(map(lambda x: x.upper(), excitations))


def get_ci_class(excitations, cl_type=ConfigurationInteraction):
    """Function constructing a truncated CI-class with the specified
    excitations.

    Parameters
    ----------
    excitations : str
        The specified excitations to use in the CI-class. For example, to
        create a CISD class both ``excitations="CISD"`` and ``excitations=["S",
        "D"]`` are valid.

    Returns
    -------
    ConfigurationInteraction
        A subclass of ``ConfigurationInteraction``.
    """
    assert cl_type in [
        ConfigurationInteraction,
        TimeDependentConfigurationInteraction,
    ]

    excitations = excitation_string_handler(excitations)

    prefix = "CI" if cl_type is ConfigurationInteraction else "TDCI"
    class_name = prefix + "".join(excitations)

    ci_class = type(class_name, (cl_type,), dict(excitations=excitations))

    return ci_class


def get_tdci_class(*args):
    return get_ci_class(*args, cl_type=TimeDependentConfigurationInteraction)


CIS = get_ci_class("CIS")
CID = get_ci_class("CID")
CISD = get_ci_class("CISD")
CIDT = get_ci_class("CIDT")
CISDT = get_ci_class("CISDT")
CIDTQ = get_ci_class("CIDTQ")
CISDTQ = get_ci_class("CISDTQ")

TDCIS = get_tdci_class("CIS")
TDCID = get_tdci_class("CID")
TDCISD = get_tdci_class("CISD")
TDCIDT = get_tdci_class("CIDT")
TDCISDT = get_tdci_class("CISDT")
TDCIDTQ = get_tdci_class("CIDTQ")
TDCISDTQ = get_tdci_class("CISDTQ")
