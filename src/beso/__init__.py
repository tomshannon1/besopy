"""Three-dimensional BESO topology optimization."""

from .fea import FiniteElementSolver, StaticResult
from .mesh import StructuredMesh
from .optimizer import BESOOptimizer, IterationRecord, OptimizationResult
from .problems import LoadCase, bridge_load_case, cantilever_load_case, make_problem

__all__ = [
    "BESOOptimizer",
    "FiniteElementSolver",
    "IterationRecord",
    "LoadCase",
    "OptimizationResult",
    "StaticResult",
    "StructuredMesh",
    "bridge_load_case",
    "cantilever_load_case",
    "make_problem",
]

__version__ = "0.1.0"
