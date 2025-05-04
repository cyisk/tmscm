from dataset.tmscm_sym.barbell.barbell import barbell_tmscm
from dataset.observational import ObservationalDatasetTMSCM
from dataset.counterfactual import CounterfactualDatasetTMSCM

barbell_train = ObservationalDatasetTMSCM(
    tmscm=barbell_tmscm,
    filepath='dataset/tmscm_sym/barbell/train.pt',
    n_samples=20000,
)
barbell_val = CounterfactualDatasetTMSCM(
    tmscm=barbell_tmscm,
    filepath='dataset/tmscm_sym/barbell/val.pt',
    n_samples=2000,
)
barbell_test = CounterfactualDatasetTMSCM(
    tmscm=barbell_tmscm,
    filepath='dataset/tmscm_sym/barbell/test.pt',
    n_samples=2000,
)
