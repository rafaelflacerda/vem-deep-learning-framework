"""
Dataset para GNN que carrega dados preprocessados e constrói grafos.

Cada amostra (viga) vira um grafo onde:
- Nós: pontos da discretização da viga
- Arestas: conectividade linear (cada nó conectado aos vizinhos)
- Node features: as 12 features físicas
- Target: deslocamento vertical por nó
"""

import torch
from torch_geometric.data import Data, Dataset

from src.data.scalers import BaseScaler, get_scaler


def build_edge_index(n_nodes: int) -> torch.Tensor:
    """
    Constrói edge_index para grafo linear (cadeia 1D).

    Para uma viga com n nós, criamos arestas bidirecionais:
    0 -- 1 -- 2 -- ... -- (n-1)

    Args:
        n_nodes: Número de nós na viga.

    Returns:
        Tensor de shape (2, 2*(n_nodes-1)) com as arestas.
        Cada coluna é uma aresta [source, target].
    """
    # Arestas para frente: 0->1, 1->2, ..., (n-2)->(n-1)
    forward_src = torch.arange(n_nodes - 1)
    forward_dst = torch.arange(1, n_nodes)

    # Arestas para trás: 1->0, 2->1, ..., (n-1)->(n-2)
    backward_src = forward_dst
    backward_dst = forward_src

    # Concatenar para grafo bidirecional
    edge_index = torch.stack(
        [
            torch.cat([forward_src, backward_src]),
            torch.cat([forward_dst, backward_dst]),
        ]
    )

    return edge_index


class BeamGraphDataset(Dataset):
    """
    Dataset de grafos para vigas 1D.

    Carrega dados preprocessados (.pt) e converte cada amostra em um grafo
    PyTorch Geometric. Aplica scaling nas features e targets.

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
        self.features = data["features"]  # (n_samples, n_nodes, n_features)
        self.targets = data["targets"]  # (n_samples, n_nodes)
        self.metadata = data["metadata"]

        self.n_samples = self.features.shape[0]
        self.n_nodes = self.features.shape[1]
        self.n_features = self.features.shape[2]

        # Edge index é o mesmo para todas as amostras (topologia fixa)
        # self.edge_index = build_edge_index(self.n_nodes)
        self.edge_index = data["edge_index"]

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
            self.feature_scaler.fit(self.features)
            # Targets tem shape (n_samples, n_nodes), precisamos adicionar dim para o scaler
            self.target_scaler.fit(self.targets.unsqueeze(-1))

        # Aplicar scaling
        self.features_scaled = self.feature_scaler.transform(self.features)
        self.targets_scaled = self.target_scaler.transform(
            self.targets.unsqueeze(-1)
        ).squeeze(-1)

    def len(self) -> int:
        return self.n_samples

    def get(self, idx: int) -> Data:
        """
        Retorna um grafo PyTorch Geometric para a amostra idx.

        Returns:
            Data object com:
                - x: node features (n_nodes, n_features)
                - edge_index: conectividade (2, n_edges)
                - y: targets (n_nodes,)
        """
        return Data(
            x=self.features_scaled[idx],
            edge_index=self.edge_index,
            y=self.targets_scaled[idx],
        )

    def inverse_transform_targets(self, targets_scaled: torch.Tensor) -> torch.Tensor:
        """
        Reverte o scaling dos targets para valores físicos.

        Args:
            targets_scaled: Tensor de targets escalados.

        Returns:
            Targets em unidades físicas originais.
        """
        if targets_scaled.dim() == 1:
            targets_scaled = targets_scaled.unsqueeze(-1)
            return self.target_scaler.inverse_transform(targets_scaled).squeeze(-1)
        return self.target_scaler.inverse_transform(targets_scaled)
