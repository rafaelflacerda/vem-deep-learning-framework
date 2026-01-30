"""
Configuração para refinamento adaptativo de malha.

Este módulo define as classes de configuração Pydantic para o pipeline
de refinamento adaptativo guiado por incerteza.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class BeamConfig(BaseModel):
    """Parâmetros físicos da viga a ser analisada."""

    I: float = Field(..., gt=0, description="Momento de inércia [m^4]")
    L: float = Field(..., gt=0, description="Comprimento [m]")
    q: float = Field(default=5000, gt=0, description="Carregamento distribuído [N/m]")
    E: float = Field(default=200.0e9, gt=0, description="Módulo de elasticidade [Pa]")


class ModelConfig(BaseModel):
    """Configuração do modelo treinado a ser usado."""

    experiment_dir: str = Field(
        ...,
        description="Caminho para o diretório do experimento (ou 'latest' para o mais recente)",
    )
    mc_samples: int = Field(default=50, ge=1, description="Número de forward passes para MC Dropout")
    use_calibration: bool = Field(
        default=True,
        description="Se True, aplica o fator de calibração sigma_scale do calibration.pt",
    )


class RefinementConfig(BaseModel):
    """Configuração do algoritmo de refinamento."""

    initial_nodes: int = Field(default=5, ge=3, description="Número de nós na malha inicial")
    lambda_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Peso do indicador: 0 = incerteza pura, 1 = curvatura pura",
    )
    uncertainty_threshold: float = Field(
        default=1e-3,
        gt=0,
        description="Threshold de incerteza máxima para parada [m]",
    )
    max_nodes: int = Field(default=50, ge=3, description="Número máximo de nós permitido")
    max_iterations: int = Field(default=20, ge=1, description="Número máximo de iterações")
    nodes_per_iteration: int = Field(default=1, ge=1, description="Quantos nós adicionar por iteração")
    
    # Estratégia de posicionamento de novos nós
    node_placement: Literal["midpoint", "max_indicator"] = Field(
        default="midpoint",
        description="Estratégia para posicionar novos nós: 'midpoint' ou 'max_indicator'",
    )

    zz_threshold: float = Field(default=1e-6, description="Threshold de erro global ZZ para parada")
    zz_strategy: str = Field(default="threshold", description="Estratégia: threshold, top_n, fraction")
    zz_threshold_factor: float = Field(default=1.5, description="Fator para threshold adaptativo")


class VisualizationConfig(BaseModel):
    """Configuração de visualização."""

    save_figures: bool = Field(default=True, description="Salvar figuras em disco")
    save_each_iteration: bool = Field(default=True, description="Salvar figura a cada iteração")
    output_dir: str | None = Field(default=None, description="Diretório de saída (None = automático)")


class AdaptiveRefinementConfig(BaseModel):
    """
    Configuração completa para refinamento adaptativo.

    Esta classe agrega todas as configurações necessárias para executar
    o pipeline de refinamento adaptativo guiado por incerteza.
    """

    beam: BeamConfig
    model: ModelConfig
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AdaptiveRefinementConfig":
        """Carrega configuração de um arquivo YAML."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Salva a configuração em um arquivo YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)