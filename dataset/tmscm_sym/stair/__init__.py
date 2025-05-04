from dataset.tmscm_sym.stair.stair import stair_tmscm
from dataset.observational import ObservationalDatasetTMSCM
from dataset.counterfactual import CounterfactualDatasetTMSCM

stair_train = ObservationalDatasetTMSCM(
    tmscm=stair_tmscm,
    filepath='dataset/tmscm_sym/stair/train.pt',
    n_samples=20000,
)
stair_val = CounterfactualDatasetTMSCM(
    tmscm=stair_tmscm,
    filepath='dataset/tmscm_sym/stair/val.pt',
    n_samples=2000,
)
stair_test = CounterfactualDatasetTMSCM(
    tmscm=stair_tmscm,
    filepath='dataset/tmscm_sym/stair/test.pt',
    n_samples=2000,
)
