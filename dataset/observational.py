from torch.utils.data import StackDataset

from common import Iota
from dataset.column_dataset import ColumnDataset
from dataset.tmscm import TMSCM
from dataset.wrapper import *


# Observational Dataset, with observational data only
class ObservationalDictDatasetTMSCM(StackDataset):

    def __init__(
        self,
        tmscm: TMSCM,
        n_samples: int,
    ):
        self._iota = tmscm.iota
        TMSCM.check_tmscm(tmscm)

        observation = ObservationalDictDatasetTMSCM.setup_datasets(
            tmscm=tmscm,
            n_samples=n_samples,
        )
        super().__init__(**{
            'observation': observation,
        })

    @staticmethod
    def setup_datasets(
        tmscm: TMSCM,
        n_samples: int = 16384,
    ):
        exogenous_values = tmscm.sample_exogenous(n_samples=n_samples)

        observation_values = tmscm.pushforward(
            exogenous_values=exogenous_values,
            intervention_values=None,
            intervention_masks=None,
        )

        observation_datasets = StackDataset(
            **{
                endo_index: ColumnDataset(observation)
                for endo_index, observation in observation_values.items()
            })
        return observation_datasets

    @property
    def iota(self) -> Iota:
        return self._iota


# Observational Dataset, with 3 additional preprocessing wrappers
class ObservationalDatasetTMSCM(
        saved(standardized(vectorized(cls=ObservationalDictDatasetTMSCM)))):

    def __init__(
        self,
        tmscm: TMSCM,
        filepath: str,
        n_samples: int = 16384,
    ):
        self._iota = tmscm.iota

        super().__init__(
            tmscm=tmscm,
            n_samples=n_samples,
            vectorize_rule={'observation': tmscm.iota.order},
            standardize_rule={
                'observation': {
                    'mean': 'observation',
                    'std': 'observation'
                }
            },
            filepath=filepath,
        )

    @property
    def iota(self) -> Iota:
        return self._iota
