"""
Indicadores de refinamento para malha adaptativa.

Este módulo calcula indicadores que determinam onde a malha deve ser
refinada. Os indicadores podem ser baseados em incerteza, curvatura,
ou uma combinação de ambos.
"""

import numpy as np


def compute_interval_uncertainty(y_std: np.ndarray) -> np.ndarray:
    """
    Calcula a incerteza média em cada intervalo entre nós.

    A incerteza de um intervalo é a média das incertezas nos dois nós
    que o delimitam.

    Args:
        y_std: Array de incertezas nos nós, shape (n_nodes,).

    Returns:
        Array de incertezas por intervalo, shape (n_nodes - 1,).
    """
    return (y_std[:-1] + y_std[1:]) / 2


def compute_curvature(y_pred: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Calcula a curvatura aproximada da solução predita.

    A curvatura é aproximada por diferenças finitas centrais:
    κ_i ≈ (y_{i-1} - 2*y_i + y_{i+1}) / h²

    Nos extremos, usamos diferenças laterais.

    Args:
        y_pred: Array de predições nos nós, shape (n_nodes,).
        positions: Array de posições dos nós, shape (n_nodes,).

    Returns:
        Array de curvaturas nos nós, shape (n_nodes,).
    """
    n = len(y_pred)
    curvature = np.zeros(n)

    for i in range(1, n - 1):
        h_left = positions[i] - positions[i - 1]
        h_right = positions[i + 1] - positions[i]
        h_avg = (h_left + h_right) / 2

        # Diferença finita central
        curvature[i] = (y_pred[i - 1] - 2 * y_pred[i] + y_pred[i + 1]) / (h_avg**2)

    # Extremos: copiar do vizinho mais próximo
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]

    return np.abs(curvature)


def compute_interval_curvature(y_pred: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Calcula a curvatura média em cada intervalo.

    Args:
        y_pred: Array de predições nos nós, shape (n_nodes,).
        positions: Array de posições dos nós, shape (n_nodes,).

    Returns:
        Array de curvaturas por intervalo, shape (n_nodes - 1,).
    """
    curvature = compute_curvature(y_pred, positions)
    return (curvature[:-1] + curvature[1:]) / 2


def normalize_indicator(values: np.ndarray) -> np.ndarray:
    """
    Normaliza um indicador para o intervalo [0, 1].

    Args:
        values: Array de valores do indicador.

    Returns:
        Array normalizado entre 0 e 1.
    """
    v_min = values.min()
    v_max = values.max()

    if v_max - v_min < 1e-12:
        return np.zeros_like(values)

    return (values - v_min) / (v_max - v_min)


def compute_refinement_indicator(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    positions: np.ndarray,
    lambda_weight: float = 0.0,
) -> np.ndarray:
    """
    Calcula o indicador combinado de refinamento por intervalo.

    O indicador é uma combinação linear de incerteza e curvatura:
    η = (1 - λ) * σ̃ + λ * κ̃

    Onde σ̃ e κ̃ são as versões normalizadas (entre 0 e 1) da incerteza
    e curvatura, respectivamente.

    Args:
        y_pred: Array de predições nos nós, shape (n_nodes,).
        y_std: Array de incertezas nos nós, shape (n_nodes,).
        positions: Array de posições dos nós, shape (n_nodes,).
        lambda_weight: Peso da curvatura (0 = incerteza pura, 1 = curvatura pura).

    Returns:
        Array de indicadores por intervalo, shape (n_nodes - 1,).
    """
    # Incerteza por intervalo
    uncertainty = compute_interval_uncertainty(y_std)

    if lambda_weight == 0.0:
        # Caso especial: incerteza pura (evita cálculo de curvatura)
        return uncertainty

    # Curvatura por intervalo
    curvature = compute_interval_curvature(y_pred, positions)

    if lambda_weight == 1.0:
        # Caso especial: curvatura pura
        return curvature

    # Normalizar ambos para [0, 1]
    uncertainty_norm = normalize_indicator(uncertainty)
    curvature_norm = normalize_indicator(curvature)

    # Combinação linear
    return (1 - lambda_weight) * uncertainty_norm + lambda_weight * curvature_norm


def select_intervals_to_refine(
    indicator: np.ndarray,
    n_intervals: int = 1,
) -> list[int]:
    """
    Seleciona os intervalos com maior indicador para refinamento.

    Args:
        indicator: Array de indicadores por intervalo.
        n_intervals: Número de intervalos a selecionar.

    Returns:
        Lista de índices dos intervalos selecionados (ordenados por indicador).
    """
    # Ordenar por indicador decrescente
    sorted_indices = np.argsort(indicator)[::-1]

    # Selecionar os top n_intervals
    return sorted_indices[:n_intervals].tolist()