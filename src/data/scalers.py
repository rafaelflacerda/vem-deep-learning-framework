"""
Scalers para normalização de features e targets.

Implementação modular que permite trocar o método de normalização facilmente.
Os scalers seguem a interface do scikit-learn (fit, transform, inverse_transform).
"""

from abc import ABC, abstractmethod

import torch


class BaseScaler(ABC):
    """Interface base para scalers."""

    @abstractmethod
    def fit(self, data: torch.Tensor) -> "BaseScaler":
        """Calcula estatísticas do scaler a partir dos dados."""
        pass

    @abstractmethod
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Aplica a transformação nos dados."""
        pass

    @abstractmethod
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Reverte a transformação."""
        pass

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Fit e transform em uma única chamada."""
        return self.fit(data).transform(data)

    @abstractmethod
    def state_dict(self) -> dict:
        """Retorna estado do scaler para serialização."""
        pass

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        """Carrega estado do scaler."""
        pass


class StandardScaler(BaseScaler):
    """
    Normaliza para média 0 e desvio padrão 1.

    z = (x - mean) / std

    Robusto a outliers se usar std (não é tão sensível quanto MinMax).
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, data: torch.Tensor) -> "StandardScaler":
        """
        Calcula média e desvio padrão.

        Args:
            data: Tensor de shape (n_samples, n_features) ou (n_samples, n_nodes, n_features)
        """
        if data.dim() == 3:
            # (n_samples, n_nodes, n_features) -> flatten para (n_samples * n_nodes, n_features)
            flat = data.reshape(-1, data.shape[-1])
        else:
            flat = data

        self.mean = flat.mean(dim=0)
        self.std = flat.std(dim=0)
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler não foi fitado. Chame fit() primeiro.")
        return (data - self.mean) / (self.std + self.eps)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler não foi fitado. Chame fit() primeiro.")
        return data * (self.std + self.eps) + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "eps": self.eps}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"]
        self.std = state["std"]
        self.eps = state.get("eps", 1e-8)


class MinMaxScaler(BaseScaler):
    """
    Normaliza para o intervalo [0, 1].

    z = (x - min) / (max - min)

    Útil quando você quer valores bounded, mas sensível a outliers.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.min: torch.Tensor | None = None
        self.max: torch.Tensor | None = None

    def fit(self, data: torch.Tensor) -> "MinMaxScaler":
        if data.dim() == 3:
            flat = data.reshape(-1, data.shape[-1])
        else:
            flat = data

        self.min = flat.min(dim=0).values
        self.max = flat.max(dim=0).values
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.min is None or self.max is None:
            raise RuntimeError("Scaler não foi fitado. Chame fit() primeiro.")
        return (data - self.min) / (self.max - self.min + self.eps)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.min is None or self.max is None:
            raise RuntimeError("Scaler não foi fitado. Chame fit() primeiro.")
        return data * (self.max - self.min + self.eps) + self.min

    def state_dict(self) -> dict:
        return {"min": self.min, "max": self.max, "eps": self.eps}

    def load_state_dict(self, state: dict) -> None:
        self.min = state["min"]
        self.max = state["max"]
        self.eps = state.get("eps", 1e-8)


class NoScaler(BaseScaler):
    """Scaler identidade — não faz nada. Útil para testes ou quando não quer scaling."""

    def fit(self, data: torch.Tensor) -> "NoScaler":
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        return data

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        return data

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


class RobustScaler(BaseScaler):
    """
    Normaliza usando mediana e IQR (intervalo interquartil).

    z = (x - median) / IQR

    Muito mais robusto a outliers que StandardScaler.
    IQR = Q3 - Q1 (percentil 75 - percentil 25)
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.median: torch.Tensor | None = None
        self.iqr: torch.Tensor | None = None

    def fit(self, data: torch.Tensor) -> "RobustScaler":
        """
        Calcula mediana e IQR.

        Args:
            data: Tensor de shape (n_samples, n_features) ou (n_samples, n_nodes, n_features)
        """
        if data.dim() == 3:
            # (n_samples, n_nodes, n_features) -> flatten
            flat = data.reshape(-1, data.shape[-1])
        else:
            flat = data

        self.median = torch.median(flat, dim=0).values
        q25 = torch.quantile(flat, 0.25, dim=0)
        q75 = torch.quantile(flat, 0.75, dim=0)
        self.iqr = q75 - q25
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.median is None or self.iqr is None:
            raise RuntimeError("Scaler não foi fitado. Chame fit() primeiro.")
        return (data - self.median) / (self.iqr + self.eps)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.median is None or self.iqr is None:
            raise RuntimeError("Scaler não foi fitado. Chame fit() primeiro.")
        return data * (self.iqr + self.eps) + self.median

    def state_dict(self) -> dict:
        return {"median": self.median, "iqr": self.iqr, "eps": self.eps}

    def load_state_dict(self, state: dict) -> None:
        self.median = state["median"]
        self.iqr = state["iqr"]
        self.eps = state.get("eps", 1e-8)


def get_scaler(name: str, **kwargs) -> BaseScaler:
    """
    Factory function para criar scalers pelo nome.

    Args:
        name: Nome do scaler ('standard', 'minmax', 'robust', 'none')
        **kwargs: Argumentos extras para o scaler (ex: eps)

    Returns:
        Instância do scaler.
    """
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,  # <-- ADICIONAR ESTA LINHA
        "none": NoScaler,
    }
    if name not in scalers:
        raise ValueError(
            f"Scaler '{name}' não reconhecido. Opções: {list(scalers.keys())}"
        )
    return scalers[name](**kwargs)
