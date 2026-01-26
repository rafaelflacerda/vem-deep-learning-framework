from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configurações de dados e dataset."""

    sampling_method: Literal["Sobol", "LHS"] = Field(default="Sobol")
    dataset_size: int = Field(default=25000, ge=10)
    val_split: float = Field(default=0.30, ge=0.0, le=1.0)
    scaler_type: Literal["standard", "minmax", "robust", "none"] = Field(default="minmax")  # <-- ADICIONAR "robust"
    rho_folder: str = Field(
        default="teste", description="Valor máximo de deslocamento normalizado"
    )


class ModelConfig(BaseModel):
    """Configurações da rede neural GNN."""

    input_dim: int = Field(
        default=10,
        ge=1,
        description="Número de features de entrada. Deve corresponder ao dataset.",
    )
    hidden_dim: int = Field(default=64, ge=16)
    output_dim: int = Field(default=1)
    num_layers: int = Field(default=6, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    
    activation: str = Field(default="relu", pattern="^(relu|silu|gelu|tanh|selu)$")


class TrainingConfig(BaseModel):
    """Configurações de treinamento."""
    epochs: int = Field(default=400, ge=1)
    batch_size: int = Field(default=256, ge=1)
    learning_rate: float = Field(default=5e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    
    loss_type: Literal["mse", "huber"] = Field(default="huber")
    huber_delta: float = Field(default=1.0, gt=0.0)
    
    optimizer_type: str = Field(default="adamw", pattern="^(adam|adamw|sgd)$")
    scheduler_type: str = Field(default="reduce_lr_on_plateau", pattern="^(reduce_lr_on_plateau|cosine_annealing|step_lr|linear)$")
    
    # Parâmetros do scheduler
    scheduler_patience: int = Field(default=20, ge=1)
    scheduler_factor: float = Field(default=0.75, ge=0.1, le=0.99)
    scheduler_t_max: int = Field(default=100, ge=1)
    scheduler_step_size: int = Field(default=10, ge=1)
    
    # NOVOS: Parâmetros específicos do AdamW
    adamw_beta1: float = Field(default=0.9, ge=0.0, le=0.99)
    adamw_beta2: float = Field(default=0.999, ge=0.0, le=0.9999)
    adamw_epsilon: float = Field(default=1e-8, gt=0.0)
    
    # Parâmetros do SGD (se você quiser variar depois)
    sgd_momentum: float = Field(default=0.9, ge=0.0, le=1.0)
    sgd_nesterov: bool = Field(default=True)

class EvaluationConfig(BaseModel):
    """Configurações de avaliação e visualização."""

    mc_samples: int = Field(default=50, ge=1)
    n_profile_samples: int = Field(default=6, ge=1)


class ExperimentConfig(BaseModel):
    """
    Configuração completa de um experimento de treinamento.

    Esta classe define e valida toda a configuração necessária para rodar
    um experimento de treinamento de GNN para predição em vigas 1D.

    Attributes:
        data: Configurações de dados e dataset.
        model: Configurações da arquitetura GNN.
        training: Configurações de otimização e treinamento.
        evaluation: Configurações de avaliação e métricas.
        cv_mode: Modo de validação ('fixed' para split fixo, 'kfold' para cross-validation).
        n_folds: Número de folds se cv_mode='kfold'.

    Example:
        >>> config = ExperimentConfig.from_yaml("configs/default.yaml")
        >>> print(config.model.hidden_dim)
        64
    """

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    cv_mode: Literal["fixed", "kfold"] = Field(default="fixed")
    n_folds: int = Field(default=5, ge=2)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Carrega configuração de um arquivo YAML."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Erro ao parsear YAML em {path}: {e}")

        if data is None:
            data = {}

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Salva a configuração em um arquivo YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
