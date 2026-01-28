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
from .refinement_config import (
    AdaptiveRefinementConfig,
    BeamConfig,
    ModelConfig as RefinementModelConfig,
    RefinementConfig,
    VisualizationConfig,
)

__all__ = [
    # Configurações de experimento (treinamento)
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    # Configurações de refinamento adaptativo
    "AdaptiveRefinementConfig",
    "BeamConfig",
    "RefinementModelConfig",
    "RefinementConfig",
    "VisualizationConfig",
]