"""Three-dimensional BESO topology optimization."""

from .fea import FiniteElementSolver, StaticResult
from .mesh import StructuredMesh
from .optimizer import BESOOptimizer, IterationRecord, OptimizationResult
from .problems import LoadCase, cantilever_load_case

__all__ = [
    "BESOOptimizer",
    "FiniteElementSolver",
    "IterationRecord",
    "LoadCase",
    "OptimizationResult",
    "StaticResult",
    "StructuredMesh",
    "cantilever_load_case",
]

__version__ = "0.1.0"
