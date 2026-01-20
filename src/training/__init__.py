"""
Módulo de treinamento para GNN em problemas de vigas 1D.

Este pacote contém a lógica completa de treinamento: a classe trainer que
coordena todo o processo, funções para calcular métricas de avaliação, e
funções para dividir dados em estratégias de validação diferentes.
"""

from .trainer import BeamGNNTrainer
from .metrics import compute_r2, compute_metrics
from .validation import split_dataset, get_kfold_splits

__all__ = [
    "BeamGNNTrainer",
    "compute_r2",
    "compute_metrics",
    "split_dataset",
    "get_kfold_splits",
]