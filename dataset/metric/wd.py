from geomloss import SamplesLoss

# Wasserstein Distance approximated by Sinkhorn Divergence
wasserstein_distance = SamplesLoss("sinkhorn", p=2)
