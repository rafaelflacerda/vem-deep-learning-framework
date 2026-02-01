"""
Dataset para GNN que carrega dados preprocessados e constrói grafos.

Suporta grafos de tamanhos variados (diferentes números de elementos por caso).

Cada amostra (viga) é um grafo onde:
- Nós: pontos da discretização da viga
- Arestas: conectividade linear (cada nó conectado aos vizinhos)
- Node features: as 10 features físicas
- Target: deslocamento vertical por nó
"""

import torch
from torch_geometric.data import Data, Dataset

from src.data.scalers import BaseScaler, get_scaler


class BeamGraphDataset(Dataset):
    """
    Dataset de grafos para vigas 1D com tamanhos variados.

    Carrega dados preprocessados (.pt) contendo lista de grafos Data.
    Aplica scaling nas features e targets.

    Args:
        pt_path: Caminho para o arquivo .pt com dados preprocessados.
        feature_scaler: Scaler para features (ou nome: 'standard', 'minmax', 'none').
        target_scaler: Scaler para targets (ou nome).
        fit_scalers: Se True, fita os scalers nos dados. Se False, assume que já estão fitados.
    """

    def __init__(
        self,
        pt_path: str,
        feature_scaler: BaseScaler | str = "standard",
        target_scaler: BaseScaler | str = "standard",
        fit_scalers: bool = True,
    ):
        super().__init__()

        # Carregar dados
        data = torch.load(pt_path, weights_only=False)
        self.data_list: list[Data] = data["data_list"]
        self.metadata = data["metadata"]

        self.n_samples = len(self.data_list)
        self.n_features = self.data_list[0].x.shape[1]  # todas têm mesma quantidade de features

        # Configurar scalers
        if isinstance(feature_scaler, str):
            self.feature_scaler = get_scaler(feature_scaler)
        else:
            self.feature_scaler = feature_scaler

        if isinstance(target_scaler, str):
            self.target_scaler = get_scaler(target_scaler)
        else:
            self.target_scaler = target_scaler

        # Fitar scalers se necessário
        if fit_scalers:
            self._fit_scalers()

        # Aplicar scaling e armazenar grafos escalados
        self.data_list_scaled = self._apply_scaling()
        del self.data_list

    def _fit_scalers(self) -> None:
        """
        Fita os scalers concatenando features/targets de todos os grafos.
        """
        # Concatenar todas as features: (total_nodes, n_features)
        all_features = torch.cat([data.x for data in self.data_list], dim=0)

        # Concatenar todos os targets: (total_nodes,)
        all_targets = torch.cat([data.y for data in self.data_list], dim=0)

        # Fitar scalers
        # features já tem shape (total_nodes, n_features), adequado para o scaler
        self.feature_scaler.fit(all_features.unsqueeze(0))  # adiciona dim de batch

        # targets precisa de dim extra para o scaler
        self.target_scaler.fit(all_targets.unsqueeze(0).unsqueeze(-1))

    def _apply_scaling(self) -> list[Data]:
        """
        Aplica scaling em cada grafo e retorna lista de grafos escalados.
        """
        scaled_list = []

        for data in self.data_list:
            # Escalar features: (n_nodes, n_features)
            x_scaled = self.feature_scaler.transform(data.x.unsqueeze(0)).squeeze(0)

            # Escalar targets: (n_nodes,)
            y_scaled = self.target_scaler.transform(
                data.y.unsqueeze(0).unsqueeze(-1)
            ).squeeze(0).squeeze(-1)

            # Criar novo Data com valores escalados
            scaled_data = Data(
                x=x_scaled,
                edge_index=data.edge_index,
                y=y_scaled,
                n_elements=data.n_elements,
            )
            scaled_list.append(scaled_data)

        return scaled_list

    def len(self) -> int:
        return self.n_samples

    def get(self, idx: int) -> Data:
        """
        Retorna o grafo escalado para a amostra idx.

        Returns:
            Data object com:
                - x: node features (n_nodes, n_features)
                - edge_index: conectividade (2, n_edges)
                - y: targets (n_nodes,)
                - n_elements: número de elementos deste grafo
        """
        return self.data_list_scaled[idx]

    def inverse_transform_targets(self, targets_scaled: torch.Tensor) -> torch.Tensor:
        """
        Reverte o scaling dos targets para valores físicos.

        Args:
            targets_scaled: Tensor de targets escalados.

        Returns:
            Targets em unidades físicas originais.
        """
        if targets_scaled.dim() == 1:
            targets_scaled = targets_scaled.unsqueeze(0).unsqueeze(-1)
            return self.target_scaler.inverse_transform(targets_scaled).squeeze(0).squeeze(-1)

        if targets_scaled.dim() == 2:
            targets_scaled = targets_scaled.unsqueeze(-1)
            return self.target_scaler.inverse_transform(targets_scaled).squeeze(-1)

        return self.target_scaler.inverse_transform(targets_scaled)