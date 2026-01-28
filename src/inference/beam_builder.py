"""
Construção de grafos para inferência em novas vigas.

Este módulo cria grafos PyTorch Geometric a partir dos parâmetros físicos
de uma viga e de um conjunto de posições de nós. As features são calculadas
exatamente como no script de preparação do dataset (prepare_sobol_dataset.py).
"""

import numpy as np
import torch
from torch_geometric.data import Data


def compute_node_features(
    I: float,
    L: float,
    q: float,
    E: float,
    positions: np.ndarray,
) -> np.ndarray:
    """
    Calcula as features para cada nó da viga.

    As features seguem exatamente a mesma ordem e cálculo do dataset de treinamento:
    [I, L, q, q_scale, q_over_EI, x, x_normalized, x_normalized², x_normalized³, x_normalized⁴]

    Args:
        I: Momento de inércia [m^4].
        L: Comprimento da viga [m].
        q: Carregamento distribuído [N/m].
        E: Módulo de elasticidade [Pa].
        positions: Array com as posições x dos nós [m].

    Returns:
        Array de shape (n_nodes, 10) com as features de cada nó.
    """
    n_nodes = len(positions)

    # Features derivadas (constantes para todos os nós)
    q_scale = (q * L**4) / (E * I)
    q_over_EI = q / (E * I)

    # Features posicionais
    x = positions
    x_normalized = positions / L
    x_normalized_2 = x_normalized**2
    x_normalized_3 = x_normalized**3
    x_normalized_4 = x_normalized**4

    # Montar array de features
    features = np.zeros((n_nodes, 9), dtype=np.float32)

    # Features globais (replicadas para todos os nós)
    features[:, 0] = I
    features[:, 1] = L

    # Features derivadas (replicadas para todos os nós)
    features[:, 2] = q_scale
    features[:, 3] = q_over_EI

    # Features posicionais (variam por nó)
    features[:, 4] = x
    features[:, 5] = x_normalized
    features[:, 6] = x_normalized_2
    features[:, 7] = x_normalized_3
    features[:, 8] = x_normalized_4

    return features


def build_edge_index(n_nodes: int) -> np.ndarray:
    """
    Constrói edge_index para grafo linear bidirecional.

    Conecta cada nó ao seu vizinho imediato, criando arestas em ambas
    as direções (grafo não-direcionado).

    Args:
        n_nodes: Número de nós no grafo.

    Returns:
        Array de shape (2, 2*(n_nodes-1)) com [source_nodes, target_nodes].
    """
    edges = []

    for i in range(n_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])

    return np.array(edges, dtype=np.int64).T


def build_beam_graph(
    I: float,
    L: float,
    q: float,
    E: float,
    positions: np.ndarray,
    feature_scaler=None,
) -> Data:
    """
    Constrói um grafo PyTorch Geometric para uma viga.

    Esta função monta o grafo completo pronto para ser passado pela GNN,
    incluindo scaling das features se um scaler for fornecido.

    Args:
        I: Momento de inércia [m^4].
        L: Comprimento da viga [m].
        q: Carregamento distribuído [N/m].
        E: Módulo de elasticidade [Pa].
        positions: Array com as posições x dos nós [m].
        feature_scaler: Scaler para normalizar as features (opcional).

    Returns:
        Objeto Data do PyTorch Geometric com x e edge_index.
    """
    # Calcular features
    features = compute_node_features(I, L, q, E, positions)
    features_tensor = torch.from_numpy(features).float()

    # Aplicar scaling se fornecido
    if feature_scaler is not None:
        features_tensor = feature_scaler.transform(features_tensor)

    # Construir edge_index
    edge_index = build_edge_index(len(positions))
    edge_index_tensor = torch.from_numpy(edge_index).long()

    return Data(x=features_tensor, edge_index=edge_index_tensor)


def create_initial_mesh(L: float, n_nodes: int) -> np.ndarray:
    """
    Cria uma malha inicial com nós igualmente espaçados.

    Args:
        L: Comprimento da viga [m].
        n_nodes: Número de nós desejado.

    Returns:
        Array com as posições dos nós.
    """
    return np.linspace(0, L, n_nodes)


def insert_node_midpoint(positions: np.ndarray, interval_index: int) -> np.ndarray:
    """
    Insere um novo nó no ponto médio de um intervalo.

    Args:
        positions: Array atual de posições dos nós (ordenado).
        interval_index: Índice do intervalo onde inserir (0 = entre nó 0 e 1).

    Returns:
        Novo array de posições com o nó inserido.
    """
    x_left = positions[interval_index]
    x_right = positions[interval_index + 1]
    x_new = (x_left + x_right) / 2

    return np.sort(np.append(positions, x_new))


def insert_nodes_at_intervals(
    positions: np.ndarray,
    interval_indices: list[int],
) -> np.ndarray:
    """
    Insere novos nós nos pontos médios de múltiplos intervalos.

    Args:
        positions: Array atual de posições dos nós (ordenado).
        interval_indices: Lista de índices dos intervalos onde inserir.

    Returns:
        Novo array de posições com os nós inseridos.
    """
    new_positions = positions.copy()

    for idx in interval_indices:
        x_left = positions[idx]
        x_right = positions[idx + 1]
        x_new = (x_left + x_right) / 2
        new_positions = np.append(new_positions, x_new)

    return np.sort(new_positions)