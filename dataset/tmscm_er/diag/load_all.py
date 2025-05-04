import os
import torch as th
from tqdm import tqdm

from dataset.tmscm_er.common import RandomTMSCMER
from dataset.observational import ObservationalDatasetTMSCM
from dataset.counterfactual import CounterfactualDatasetTMSCM

# Default settings for generating random TM-SCM in ER-DIAG
tmscm_er_diag_kwargs = dict(
    min_variables=3,
    max_variables=8,
    min_dimensions=1,
    max_dimensions=8,
    prob_edges=0.5,
    num_components=3,
    mechanism_mode='diag',
)

tmscm_er_diag_train = []
tmscm_er_diag_val = []
tmscm_er_diag_test = []


# Load or initialize all TM-SCMs in ER-DIAG
for i in tqdm(
    range(50),
    desc="Loading TM-SCM-ER-DIAG datasets",
    unit="dataset",
):
    # Initialize SCM
    scm_dictpath = f'dataset/tmscm_er/diag/scm_{i + 1}.pt'
    if os.path.exists(scm_dictpath):
        scm_dict = th.load(scm_dictpath, weights_only=True)
        tmscm_er_diag_i = RandomTMSCMER.from_dict(scm_dict)
    else:
        th.manual_seed(i + 239389)
        tmscm_er_diag_i = RandomTMSCMER(**tmscm_er_diag_kwargs)
        th.save(tmscm_er_diag_i.to_dict(), scm_dictpath)

    # Initialize datasets
    tmscm_er_diag_i_train = ObservationalDatasetTMSCM(
        tmscm=tmscm_er_diag_i,
        filepath=f'dataset/tmscm_er/diag/train_{i + 1}.pt',
        n_samples=20000,
    )
    tmscm_er_diag_train.append(tmscm_er_diag_i_train)
    tmscm_er_diag_i_val = CounterfactualDatasetTMSCM(
        tmscm=tmscm_er_diag_i,
        filepath=f'dataset/tmscm_er/diag/val_{i + 1}.pt',
        n_samples=2000,
    )
    tmscm_er_diag_val.append(tmscm_er_diag_i_val)
    tmscm_er_diag_i_test = CounterfactualDatasetTMSCM(
        tmscm=tmscm_er_diag_i,
        filepath=f'dataset/tmscm_er/diag/test_{i + 1}.pt',
        n_samples=2000,
    )
    tmscm_er_diag_test.append(tmscm_er_diag_i_test)
