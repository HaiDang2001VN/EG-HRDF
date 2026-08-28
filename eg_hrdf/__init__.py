"""EG-HRDF: Entropy-Guided Hierarchical Rectified Density Flow."""

from .octree import Octree, build_octree, normalize_points, denormalize_points, normalized_entropy
from .network import FiLMDensityFlowNet
from .flow import SimplexDensityFlowMatcher, uniform_p0
from .scheduler import AdaptiveDensityScheduler, SchedulerConfig, GenerationStats
from .hier_latent import HierarchicalLatentGenerator
from .hash_context import SpatialHashContext
from .residual_decoder import LocalResidualDecoder
from .metrics import density_aware_chamfer, chamfer_distance

__all__ = [
    "Octree",
    "build_octree",
    "normalize_points",
    "denormalize_points",
    "normalized_entropy",
    "FiLMDensityFlowNet",
    "SimplexDensityFlowMatcher",
    "uniform_p0",
    "AdaptiveDensityScheduler",
    "SchedulerConfig",
    "GenerationStats",
    "HierarchicalLatentGenerator",
    "SpatialHashContext",
    "LocalResidualDecoder",
    "density_aware_chamfer",
    "chamfer_distance",
]
