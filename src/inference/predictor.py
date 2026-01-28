"""
Preditor para inferência com quantificação de incerteza.

Este módulo carrega um modelo treinado e seus scalers, e faz predições
com MC Dropout para obter estimativas de incerteza.
"""

from pathlib import Path

import torch
from torch_geometric.data import Data

from src.config.experiment_config import ExperimentConfig
from src.data.scalers import get_scaler
from src.modeling.gnn import BeamGNN
from src.paths import get_latest_experiment


class BeamPredictor:
    """
    Preditor para vigas usando modelo GNN treinado.

    Carrega um modelo de um diretório de experimento e fornece métodos
    para fazer predições com quantificação de incerteza via MC Dropout.

    Attributes:
        model: Modelo GNN carregado.
        feature_scaler: Scaler para normalizar features de entrada.
        target_scaler: Scaler para desnormalizar predições.
        sigma_scale: Fator de calibração da incerteza (se disponível).
        device: Device onde o modelo está carregado.
    """

    def __init__(
        self,
        experiment_dir: str | Path,
        device: torch.device | None = None,
        use_calibration: bool = True,
    ):
        """
        Inicializa o preditor carregando modelo e scalers.

        Args:
            experiment_dir: Caminho para o diretório do experimento,
                ou "latest" para usar o experimento mais recente.
            device: Device a usar (None = detecta automaticamente).
            use_calibration: Se True, carrega e usa o fator de calibração.
        """
        # Resolver diretório do experimento
        if experiment_dir == "latest":
            exp_dir = get_latest_experiment()
            if exp_dir is None:
                raise FileNotFoundError("Nenhum experimento encontrado em results/experiments/")
        else:
            exp_dir = Path(experiment_dir)

        if not exp_dir.exists():
            raise FileNotFoundError(f"Diretório do experimento não encontrado: {exp_dir}")

        self.experiment_dir = exp_dir

        # Detectar device
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        self.device = device

        # Carregar configuração
        config_path = exp_dir / "config_used.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuração não encontrada: {config_path}")

        self.config = ExperimentConfig.from_yaml(config_path)

        # Carregar modelo
        self._load_model()

        # Carregar scalers
        self._load_scalers()

        # Carregar calibração (se disponível e solicitado)
        self.sigma_scale = 1.0
        if use_calibration:
            self._load_calibration()

    def _load_model(self) -> None:
        """Carrega o modelo treinado."""
        model_path = self.experiment_dir / "best_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        self.model = BeamGNN(
            input_dim=self.config.model.input_dim,
            hidden_dim=self.config.model.hidden_dim,
            output_dim=self.config.model.output_dim,
            num_layers=self.config.model.num_layers,
            dropout=self.config.model.dropout,
            activation=self.config.model.activation,
            use_layer_norm=getattr(self.config.model, "use_layer_norm", True)
        )

        checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]
        # Remove prefixo _orig_mod. se existir (modelo salvo com torch.compile)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)

    def _load_scalers(self) -> None:
        """Carrega os scalers de features e targets."""
        scalers_path = self.experiment_dir / "scalers.pt"
        if not scalers_path.exists():
            raise FileNotFoundError(f"Scalers não encontrados: {scalers_path}")

        scalers_data = torch.load(scalers_path, weights_only=False)

        # Reconstruir feature scaler
        scaler_type = scalers_data.get("scaler_type", "minmax")
        self.feature_scaler = get_scaler(scaler_type)
        self.feature_scaler.load_state_dict(scalers_data["feature_scaler"])

        # Reconstruir target scaler
        self.target_scaler = get_scaler(scaler_type)
        self.target_scaler.load_state_dict(scalers_data["target_scaler"])

    def _load_calibration(self) -> None:
        """Carrega o fator de calibração da incerteza, se disponível."""
        calibration_path = self.experiment_dir / "calibration.pt"
        if calibration_path.exists():
            calibration_data = torch.load(calibration_path, weights_only=False)
            self.sigma_scale = calibration_data.get("sigma_scale", 1.0)

    def predict(
        self,
        graph: Data,
        mc_samples: int = 50,
        apply_calibration: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Faz predição com quantificação de incerteza via MC Dropout.

        Args:
            graph: Grafo PyTorch Geometric (já com features normalizadas).
            mc_samples: Número de forward passes para MC Dropout.
            apply_calibration: Se True, aplica o fator sigma_scale à incerteza.

        Returns:
            Tupla (y_mean, y_std) com predições e incertezas em valores físicos.
            Ambos têm shape (n_nodes,).
        """
        self.model.train()  # Ativa dropout

        graph = graph.to(self.device)

        predictions = []
        with torch.no_grad():
            for _ in range(mc_samples):
                out = self.model(graph.x, graph.edge_index)
                predictions.append(out.squeeze())

        predictions = torch.stack(predictions)  # (mc_samples, n_nodes)

        # Média e desvio padrão no espaço normalizado
        y_mean_scaled = predictions.mean(dim=0)
        y_std_scaled = predictions.std(dim=0)

        # Converter para valores físicos
        y_mean = self._inverse_transform_targets(y_mean_scaled)

        # Escalar desvio padrão para valores físicos
        scale_factor = self._get_target_scale_factor()
        y_std = y_std_scaled.cpu() * scale_factor

        # Aplicar calibração
        if apply_calibration:
            y_std = y_std * self.sigma_scale

        return y_mean, y_std

    def _inverse_transform_targets(self, targets_scaled: torch.Tensor) -> torch.Tensor:
        """Converte targets normalizados para valores físicos."""
        targets_scaled = targets_scaled.cpu()
        if targets_scaled.dim() == 1:
            targets_scaled = targets_scaled.unsqueeze(-1)
            return self.target_scaler.inverse_transform(targets_scaled).squeeze(-1)
        return self.target_scaler.inverse_transform(targets_scaled)

    def _get_target_scale_factor(self) -> float:
        """Retorna o fator de escala para converter std normalizado para físico."""
        if hasattr(self.target_scaler, "std") and self.target_scaler.std is not None:
            return self.target_scaler.std.item() + 1e-8
        elif hasattr(self.target_scaler, "max") and self.target_scaler.max is not None:
            return (self.target_scaler.max.item() - self.target_scaler.min.item()) + 1e-8
        else:
            return 1.0

    def get_feature_scaler(self):
        """Retorna o feature scaler para uso na construção de grafos."""
        return self.feature_scaler