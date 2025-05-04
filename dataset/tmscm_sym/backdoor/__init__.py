from dataset.tmscm_sym.backdoor.backdoor import backdoor_tmscm
from dataset.observational import ObservationalDatasetTMSCM
from dataset.counterfactual import CounterfactualDatasetTMSCM

backdoor_train = ObservationalDatasetTMSCM(
    tmscm=backdoor_tmscm,
    filepath='dataset/tmscm_sym/backdoor/train.pt',
    n_samples=20000,
)
backdoor_val = CounterfactualDatasetTMSCM(
    tmscm=backdoor_tmscm,
    filepath='dataset/tmscm_sym/backdoor/val.pt',
    n_samples=2000,
)
backdoor_test = CounterfactualDatasetTMSCM(
    tmscm=backdoor_tmscm,
    filepath='dataset/tmscm_sym/backdoor/test.pt',
    n_samples=2000,
)
