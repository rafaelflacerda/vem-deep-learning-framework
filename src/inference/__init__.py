"""
Módulo de inferência para refinamento adaptativo.

Este módulo fornece as ferramentas para:
- Construir grafos para novas vigas (beam_builder)
- Fazer predições com incerteza (predictor)
- Calcular indicadores de refinamento (refinement)
- Verificar critérios de parada (stopping)
"""

from src.inference.beam_builder import (
    build_beam_graph,
    compute_node_features,
    create_initial_mesh,
    insert_node_midpoint,
    insert_nodes_at_intervals,
)
from src.inference.predictor import BeamPredictor
from src.inference.refinement import (
    compute_curvature,
    compute_interval_curvature,
    compute_interval_uncertainty,
    compute_refinement_indicator,
    select_intervals_to_refine,
)
from src.inference.stopping import (
    CompositeCriterion,
    MaxIterationsCriterion,
    MaxNodesCriterion,
    MaxUncertaintyCriterion,
    StoppingCriterion,
    create_stopping_criterion,
)

__all__ = [
    # beam_builder
    "build_beam_graph",
    "compute_node_features",
    "create_initial_mesh",
    "insert_node_midpoint",
    "insert_nodes_at_intervals",
    # predictor
    "BeamPredictor",
    # refinement
    "compute_curvature",
    "compute_interval_curvature",
    "compute_interval_uncertainty",
    "compute_refinement_indicator",
    "select_intervals_to_refine",
    # stopping
    "CompositeCriterion",
    "MaxIterationsCriterion",
    "MaxNodesCriterion",
    "MaxUncertaintyCriterion",
    "StoppingCriterion",
    "create_stopping_criterion",
]