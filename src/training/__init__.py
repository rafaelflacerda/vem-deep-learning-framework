"""
Módulo de treinamento para GNN em problemas de vigas 1D.

Este pacote contém a lógica completa de treinamento: a classe trainer que
coordena todo o processo, funções para calcular métricas de avaliação, e
funções para dividir dados em estratégias de validação diferentes.
"""

from .calibration import (
    calibrate_uncertainty,
    compute_calibration_metrics,
    find_optimal_sigma_scale,
)
from .metrics import compute_metrics, compute_r2
from .trainer import BeamGNNTrainer
from .validation import get_kfold_splits, split_dataset

__all__ = [
    "BeamGNNTrainer",
    "compute_r2",
    "compute_metrics",
    "compute_calibration_metrics",
    "calibrate_uncertainty",
    "find_optimal_sigma_scale",
    "split_dataset",
    "get_kfold_splits",
]