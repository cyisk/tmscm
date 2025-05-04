import os
import torch as th

from dataset.tmscm_er.common import RandomTMSCMER
from dataset.observational import ObservationalDatasetTMSCM
from dataset.counterfactual import CounterfactualDatasetTMSCM

# Default settings for generating random TM-SCM in ER-TRIL
tmscm_er_tril_kwargs = dict(
    min_variables=3,
    max_variables=8,
    min_dimensions=1,
    max_dimensions=8,
    prob_edges=0.5,
    num_components=3,
    mechanism_mode='tril',
)


# Load or initialize a TM-SCM in ER-TRIL
def tmscm_er_tril_i(scm_id: int = 1):
    assert 1 <= scm_id <= 50, "Error: Only 50 datasets has been created"
    i = scm_id - 1

    # Initialize SCM
    scm_dictpath = f'dataset/tmscm_er/tril/scm_{i + 1}.pt'
    if os.path.exists(scm_dictpath):
        scm_dict = th.load(scm_dictpath, weights_only=True)
        tmscm_er_tril_i = RandomTMSCMER.from_dict(scm_dict)
    else:
        th.manual_seed(i + 506461)
        tmscm_er_tril_i = RandomTMSCMER(**tmscm_er_tril_kwargs)
        th.save(tmscm_er_tril_i.to_dict(), scm_dictpath)

    # Initialize datasets
    tmscm_er_tril_i_train = ObservationalDatasetTMSCM(
        tmscm=tmscm_er_tril_i,
        filepath=f'dataset/tmscm_er/tril/train_{i + 1}.pt',
        n_samples=20000,
    )
    tmscm_er_tril_i_val = CounterfactualDatasetTMSCM(
        tmscm=tmscm_er_tril_i,
        filepath=f'dataset/tmscm_er/tril/val_{i + 1}.pt',
        n_samples=2000,
    )
    tmscm_er_tril_i_test = CounterfactualDatasetTMSCM(
        tmscm=tmscm_er_tril_i,
        filepath=f'dataset/tmscm_er/tril/test_{i + 1}.pt',
        n_samples=2000,
    )
    return tmscm_er_tril_i_train, tmscm_er_tril_i_val, tmscm_er_tril_i_test
