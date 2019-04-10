from configuration_interaction import CIS
from configuration_interaction.tdci import TimeDependentConfigurationInteraction


class TDCIS(TimeDependentConfigurationInteraction):
    def __init__(self, *args, **kwargs):
        super().__init__(CIS, *args, **kwargs)
