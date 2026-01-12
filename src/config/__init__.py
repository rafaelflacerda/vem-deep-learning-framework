"""
Módulo de configuração para experimentos de treinamento.

Este pacote centraliza toda a definição de configurações para experimentos,
usando Pydantic para validação automática e type checking.
"""

from .experiment_config import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)

__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "ExperimentConfig",
]
