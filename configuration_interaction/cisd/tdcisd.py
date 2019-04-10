from configuration_interaction import CISD
from configuration_interaction.tdci import TimeDependentConfigurationInteraction


class TDCISD(TimeDependentConfigurationInteraction):
    def __init__(self, *args, **kwargs):
        super().__init__(CISD, *args, **kwargs)
