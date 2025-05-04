from dataset.tmscm_sym.fork.fork import fork_tmscm
from dataset.observational import ObservationalDatasetTMSCM
from dataset.counterfactual import CounterfactualDatasetTMSCM

fork_train = ObservationalDatasetTMSCM(
    tmscm=fork_tmscm,
    filepath='dataset/tmscm_sym/fork/train.pt',
    n_samples=20000,
)
fork_val = CounterfactualDatasetTMSCM(
    tmscm=fork_tmscm,
    filepath='dataset/tmscm_sym/fork/val.pt',
    n_samples=2000,
)
fork_test = CounterfactualDatasetTMSCM(
    tmscm=fork_tmscm,
    filepath='dataset/tmscm_sym/fork/test.pt',
    n_samples=2000,
)
