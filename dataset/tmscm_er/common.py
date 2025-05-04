import torch as th
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from typing import Dict, Tuple, Any
from zuko.transforms import MonotonicRQSTransform
from common import invert_graph
from dataset.tmscm import TMSCM


class RandomSegmentedSinusoidal():
    r"""Generate ramdom function like: $$f(\mathbf{x})=\underbrace{\sum_{i=1}^{m}c_i\cdot\sin(\langle\mathbf{w}_i,\mathbf{x}\rangle+\phi_i)}_{\text{sinusoidal term}}+\underbrace{s_{j^*(\mathbf{x})}\cdot\|\mathbf{x}-\boldsymbol{\mu}_{j^*(\mathbf{x})}\|}_{\text{segmented term}}$$
    where: $j^*(\mathbf{x})=\arg\min_{1\le j\le n}\|\mathbf{x}-\boldsymbol{\mu}_j\|$.
    """

    def __init__(
        self,
        dim: int,
        num_segmented: int = 3,
        num_sinusoidal: int = 3,
    ):
        self.dim = dim
        self.num_segmented = num_segmented
        self.num_sinusoidal = num_sinusoidal

        # Sinusoidal term parameters
        self.coefficients = th.randn(num_sinusoidal)
        self.frequencies = th.rand(num_sinusoidal, dim) * 1.5 + 0.5
        self.phases = th.rand(num_sinusoidal) * 2 * th.pi

        # Segmented term parameters
        self.segment_centers = th.randn(num_segmented, dim)
        self.segment_slopes = th.randn(num_segmented)

    def sinusoidals_component(
        self,
        x: th.Tensor,
    ) -> th.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        sinusoidals = th.sin(th.matmul(x, self.frequencies.T) + self.phases)
        return th.matmul(sinusoidals, self.coefficients)

    def segment_component(
        self,
        x: th.Tensor,
    ) -> th.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        distances = th.cdist(x, self.segment_centers)
        closest_segments = th.argmin(distances, dim=1)
        offsets = distances[th.arange(x.size(0)), closest_segments]
        return self.segment_slopes[closest_segments] * offsets

    def __call__(
        self,
        x: th.Tensor,
    ) -> th.Tensor:
        sinusoidals = self.sinusoidals_component(x)
        segments = self.segment_component(x)
        y = sinusoidals + segments
        return y.squeeze(0) if x.ndim == 1 else y

    @staticmethod
    def from_dict(save_dict: Dict[str, Any], ) -> "RandomSegmentedSinusoidal":
        dim = save_dict['dim']
        num_segmented = save_dict['num_segmented']
        num_sinusoidal = save_dict['num_sinusoidal']
        obj = RandomSegmentedSinusoidal(
            dim=dim,
            num_segmented=num_segmented,
            num_sinusoidal=num_sinusoidal,
        )
        obj.coefficients = save_dict['coefficients']
        obj.frequencies = save_dict['frequencies']
        obj.phases = save_dict['phases']
        obj.segment_centers = save_dict['segment_centers']
        obj.segment_slopes = save_dict['segment_slopes']
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dim': self.dim,
            'num_segmented': self.num_segmented,
            'num_sinusoidal': self.num_sinusoidal,
            'coefficients': self.coefficients,
            'frequencies': self.frequencies,
            'phases': self.phases,
            'segment_centers': self.segment_centers,
            'segment_slopes': self.segment_slopes,
        }


class RandomMonotonicRQS():

    def __init__(
        self,
        bin: int = 8,
    ):
        self.bin = bin

        # Monotonic rational-quadratic spline parameters
        self.widths = th.rand(bin)
        self.heights = th.rand(bin)
        self.derivatives = th.rand(bin)

    def __call__(
        self,
        x1: th.Tensor,
        x2: th.Tensor,
    ):
        x1_dim = [1] * x1.dim()
        x1 = x1.unsqueeze(-1).repeat(*x1_dim, self.bin)
        x1 = x1.contiguous()
        transform = MonotonicRQSTransform(
            widths=x1 * self.widths,
            heights=x1 * self.heights,
            derivatives=x1 * self.derivatives,
        )
        print(x1.shape, x2.shape)
        return transform(x2.contiguous())

    @staticmethod
    def from_dict(save_dict: Dict[str, Any]) -> "RandomMonotonicRQS":
        bin = save_dict['bin']
        obj = RandomMonotonicRQS(bin=bin)
        obj.widths = save_dict['widths']
        obj.heights = save_dict['heights']
        obj.derivatives = save_dict['derivatives']
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return {
            'bin': self.bin,
            'widths': self.widths,
            'heights': self.heights,
            'derivatives': self.derivatives,
        }


class RandomMonotonicAffine():

    def __init__(self):
        self.loc = th.rand((1, )).squeeze(0)
        self.scale1 = th.exp(th.rand((1, ))).squeeze(0)
        if th.rand((1, )).squeeze(0).item() > 0.5:
            self.scale1 = -self.scale1
        self.scale2 = th.exp(th.rand((1, ))).squeeze(0)
        if th.rand((1, )).squeeze(0).item() > 0.5:
            self.scale2 = -self.scale2

    def __call__(
        self,
        x1: th.Tensor,
        x2: th.Tensor,
    ):
        return x1 * self.scale1 + x2 * self.scale2 + self.loc * 2

    @staticmethod
    def from_dict(save_dict: Dict[str, Any]) -> "RandomMonotonicAffine":
        obj = RandomMonotonicAffine()
        obj.loc = save_dict['loc']
        obj.scale1 = save_dict['scale1']
        obj.scale2 = save_dict['scale2']
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return {
            'loc': self.loc,
            'scale1': self.scale1,
            'scale2': self.scale2,
        }


class RandomTriangularMonotonicCausalMechanism():

    def __init__(
        self,
        parents: Dict[str, int],
        exogenous: Tuple[str, int],
        mode: str = "diag",
        rqs_bin: int = 8,
        ss_seg: int = 6,
        ss_sin: int = 6,
    ):
        self.parents = parents
        self.exogenous = exogenous
        self.mode = mode
        self.rqs_bin = rqs_bin
        self.ss_seg = ss_seg
        self.ss_sin = ss_sin

        self.parents_names = list(parents.keys())
        self.parents_dim = sum(list(parents.values()))
        self.exogenous_name = exogenous[0]
        self.exogenous_dim = exogenous[1]
        self.mode = mode

        # Initialize mechanism function
        self.g = RandomSegmentedSinusoidal(
            dim=self.parents_dim,
            num_segmented=ss_seg,
            num_sinusoidal=ss_sin,
        )
        self.h = [
            RandomMonotonicAffine()
            for _ in range(self.exogenous_dim)
        ]

    def __call__(self, **kwargs):
        exogenous = kwargs[self.exogenous_name]
        if len(self.parents_names) > 0:
            parent = th.cat(
                [kwargs[v_name] for v_name in self.parents_names],
                dim=-1,
            )
        else:
            parent = th.zeros_like(exogenous[..., :0])

        if self.mode == 'diag':
            # y_i=h_i(g(pa), u_i)
            def dp(y): return y[0]
        else:
            # y_i=h_i(y_{i-1}, u_i); y_0 = pa
            def dp(y): return th.tanh(y[-1]) + th.ceil(y[-1])

        y = [self.g(parent)]
        for i in range(exogenous.size(-1)):
            u_i = exogenous[..., i]
            y_i = self.h[i](dp(y), u_i)
            y.append(y_i)
        return th.stack(y[1:], dim=-1)

    @staticmethod
    def from_dict(
        save_dict: Dict[str,
                        Any]) -> "RandomTriangularMonotonicCausalMechanism":
        parents = save_dict['parents']
        exogenous = save_dict['exogenous']
        mode = save_dict['mode']
        rqs_bin = save_dict['rqs_bin']
        ss_seg = save_dict['ss_seg']
        ss_sin = save_dict['ss_sin']
        obj = RandomTriangularMonotonicCausalMechanism(
            parents=parents,
            exogenous=exogenous,
            mode=mode,
            rqs_bin=rqs_bin,
            ss_seg=ss_seg,
            ss_sin=ss_sin,
        )
        obj.g = RandomSegmentedSinusoidal.from_dict(save_dict['g'])
        obj.h = [
            RandomMonotonicAffine.from_dict(h_dict)
            for h_dict in save_dict['h']
        ]
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return {
            'parents': self.parents,
            'exogenous': self.exogenous,
            'mode': self.mode,
            'rqs_bin': self.rqs_bin,
            'ss_seg': self.ss_seg,
            'ss_sin': self.ss_sin,
            'g': self.g.to_dict(),
            'h': [h_i.to_dict() for h_i in self.h]
        }


def random_erdos_renyi_dag(
    n: int,
    p: float,
):
    # Erdős-Rényi
    adjacency = th.rand(n, n) < p
    adjacency.fill_diagonal_(False)
    adjacency = th.triu(adjacency, diagonal=1)

    # Matrix to dict
    graph = {}
    num_nodes = adjacency.shape[0]
    for i in range(num_nodes):
        name_i = f"x{str(i + 1)}"
        graph[name_i] = []
        for j in range(num_nodes):
            name_j = f"x{str(j + 1)}"
            if adjacency[i, j]:
                graph[name_i].append(name_j)

    return graph


class RandomGaussianMixture(MixtureSameFamily):

    def __init__(
        self,
        n: int,
        k: int,
    ):
        self.n = n
        self.k = k

        # Randomized weights for mixture
        self.weights = th.rand(k)
        self.weights /= self.weights.sum()

        # Randomized parameters for multivariate normal
        self.means = th.randn(k, n)
        self.covs = th.randn(k, n, n)
        epsilon = (th.eye(n) * 1.0).unsqueeze(0).repeat(k, 1, 1)
        self.covs = th.matmul(self.covs, self.covs.transpose(
            -1, -2)) + epsilon  # positive-definite

        # Initialize the mixture distribution
        super().__init__(
            mixture_distribution=Categorical(self.weights),
            component_distribution=MultivariateNormal(self.means, self.covs),
        )

    @staticmethod
    def from_dict(save_dict: Dict[str, Any]) -> "RandomGaussianMixture":
        n = save_dict['n']
        k = save_dict['k']
        weights = save_dict['weights']
        means = save_dict['means']
        covs = save_dict['covs']
        obj = RandomGaussianMixture(n, k)
        obj.weights = weights
        obj.means = means
        obj.covs = covs
        super(RandomGaussianMixture, obj).__init__(
            mixture_distribution=Categorical(weights),
            component_distribution=MultivariateNormal(means, covs),
        )
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return {
            'n': self.n,
            'k': self.k,
            'weights': self.weights,
            'means': self.means,
            'covs': self.covs,
        }


class RandomTMSCMER(TMSCM):

    def __init__(
        self,
        min_variables: int,
        max_variables: int,
        min_dimensions: int,
        max_dimensions: int,
        prob_edges: float,
        num_components: int,
        mechanism_mode: bool = 'diag',
        rqs_bin: int = 8,
        ss_seg: int = 6,
        ss_sin: int = 6,
    ):
        self.min_variables = min_variables
        self.max_variables = max_variables
        self.min_dimensions = min_dimensions
        self.max_dimensions = max_dimensions
        self.prob_edges = prob_edges
        self.num_components = num_components
        self.mechanism_mode = mechanism_mode
        self.rqs_bin = rqs_bin
        self.ss_seg = ss_seg
        self.ss_sin = ss_sin

        # Variables and dimensions
        num_variables = th.randint(min_variables, max_variables + 1, tuple())
        num_variables = num_variables.item()
        variables = [f"x{str(i + 1)}" for i in range(num_variables)]
        dimensions = {}
        for variable in variables:
            dimension = th.randint(min_dimensions, max_dimensions + 1, tuple())
            dimensions[variable] = dimension.item()

        # Generate random DAG
        graph = random_erdos_renyi_dag(n=num_variables, p=prob_edges)
        while sum([len(graph[v])
                   for v in graph]) == 0:  # Ensure at least 1 edge
            graph = random_erdos_renyi_dag(n=num_variables, p=prob_edges)
        dependencies = invert_graph(graph)

        # Generate random exogenous distribution
        exogenous_distributions = {}
        for i, variable in enumerate(variables):
            distribution = RandomGaussianMixture(
                n=dimensions[variable],
                k=num_components,
            )
            exogenous_distributions[f"u{str(i + 1)}"] = distribution
        endo_exo_index_pairs = [
            (f"x{str(i + 1)}", f"u{str(i + 1)}")
            for i in range(num_variables)
        ]

        # Generate mechanisms
        mechanisms = {}
        for i, variable in enumerate(variables):
            parents = {
                parent: dimensions[parent]
                for parent in dependencies[variable]
            }
            exogenous = (
                endo_exo_index_pairs[i][1],
                dimensions[variables[i]],
            )
            mechanisms[variable] = RandomTriangularMonotonicCausalMechanism(
                parents=parents,
                exogenous=exogenous,
                mode=mechanism_mode,
                rqs_bin=rqs_bin,
                ss_seg=ss_seg,
                ss_sin=ss_sin,
            )

        self.dependencies = dependencies
        self.mechanisms = mechanisms
        self.exogenous_distributions = exogenous_distributions
        self.endo_exo_index_pairs = endo_exo_index_pairs

        super().__init__(
            dependencies=dependencies,
            mechanisms=mechanisms,
            exogenous_distributions=exogenous_distributions,
            endo_exo_index_pairs=endo_exo_index_pairs,
        )

    @staticmethod
    def from_dict(save_dict: Dict[str, Any]) \
            -> RandomTriangularMonotonicCausalMechanism:
        min_variables = save_dict['min_variables']
        max_variables = save_dict['max_variables']
        min_dimensions = save_dict['min_dimensions']
        max_dimensions = save_dict['max_dimensions']
        prob_edges = save_dict['prob_edges']
        num_components = save_dict['num_components']
        mechanism_mode = save_dict['mechanism_mode']
        rqs_bin = save_dict['rqs_bin']
        ss_seg = save_dict['ss_seg']
        ss_sin = save_dict['ss_sin']
        obj = RandomTMSCMER(
            min_variables=min_variables,
            max_variables=max_variables,
            min_dimensions=min_dimensions,
            max_dimensions=max_dimensions,
            prob_edges=prob_edges,
            num_components=num_components,
            mechanism_mode=mechanism_mode,
            rqs_bin=rqs_bin,
            ss_seg=ss_seg,
            ss_sin=ss_sin,
        )
        obj.dependencies = save_dict['dependencies']
        obj.mechanisms = {
            key: RandomTriangularMonotonicCausalMechanism.from_dict(value)
            for key, value in save_dict['mechanisms'].items()
        }
        obj.exogenous_distributions = {
            key: RandomGaussianMixture.from_dict(value)
            for key, value in save_dict['exogenous_distributions'].items()
        }
        obj.endo_exo_index_pairs = save_dict['endo_exo_index_pairs']
        super(RandomTMSCMER, obj).__init__(
            dependencies=obj.dependencies,
            mechanisms=obj.mechanisms,
            exogenous_distributions=obj.exogenous_distributions,
            endo_exo_index_pairs=obj.endo_exo_index_pairs,
        )
        return obj

    def to_dict(self):
        return {
            'min_variables': self.min_variables,
            'max_variables': self.max_variables,
            'min_dimensions': self.min_dimensions,
            'max_dimensions': self.max_dimensions,
            'prob_edges': self.prob_edges,
            'num_components': self.num_components,
            'mechanism_mode': self.mechanism_mode,
            'rqs_bin': self.rqs_bin,
            'ss_seg': self.ss_seg,
            'ss_sin': self.ss_sin,
            'dependencies': self.dependencies,
            'mechanisms': {
                key: value.to_dict()
                for key, value in self.mechanisms.items()
            },
            'exogenous_distributions': {
                key: value.to_dict()
                for key, value in self.exogenous_distributions.items()
            },
            'endo_exo_index_pairs': self.endo_exo_index_pairs,
        }
