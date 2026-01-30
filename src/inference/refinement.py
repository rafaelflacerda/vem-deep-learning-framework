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

def compute_moment_from_displacement(
    y_pred: np.ndarray,
    positions: np.ndarray,
    E: float,
    I: float,
) -> np.ndarray:
    """
    Calcula o momento fletor a partir dos deslocamentos usando diferenças finitas.
    
    M = EI * κ = EI * d²w/dx²
    
    Para nós internos, usa diferenças finitas centrais.
    Para nós extremos, usa diferenças finitas laterais.
    
    Args:
        y_pred: Array de deslocamentos nos nós, shape (n_nodes,).
        positions: Array de posições dos nós, shape (n_nodes,).
        E: Módulo de elasticidade [Pa].
        I: Momento de inércia [m^4].
        
    Returns:
        Array de momentos fletores nos nós, shape (n_nodes,).
    """
    n = len(y_pred)
    M = np.zeros(n)
    
    # Nós internos: diferença finita central
    for i in range(1, n - 1):
        h_left = positions[i] - positions[i - 1]
        h_right = positions[i + 1] - positions[i]
        
        # Fórmula para malha não-uniforme
        curvature = 2 * (
            y_pred[i - 1] / (h_left * (h_left + h_right))
            - y_pred[i] / (h_left * h_right)
            + y_pred[i + 1] / (h_right * (h_left + h_right))
        )
        M[i] = E * I * curvature
    
    # Nó esquerdo (x=0): diferença finita forward
    if n >= 3:
        h1 = positions[1] - positions[0]
        h2 = positions[2] - positions[1]
        curvature_0 = 2 * (
            y_pred[0] / (h1 * (h1 + h2))
            - y_pred[1] / (h1 * h2)
            + y_pred[2] / (h2 * (h1 + h2))
        )
        M[0] = E * I * curvature_0
    else:
        M[0] = M[1] if n > 1 else 0.0
    
    # Nó direito (x=L): diferença finita backward
    if n >= 3:
        h1 = positions[-2] - positions[-3]
        h2 = positions[-1] - positions[-2]
        curvature_n = 2 * (
            y_pred[-3] / (h1 * (h1 + h2))
            - y_pred[-2] / (h1 * h2)
            + y_pred[-1] / (h2 * (h1 + h2))
        )
        M[-1] = E * I * curvature_n
    else:
        M[-1] = M[-2] if n > 1 else 0.0
    
    return M


def recover_moment_zz(
    M_h: np.ndarray,
    positions: np.ndarray,
    patch_size: int = 3,
) -> np.ndarray:
    """
    Recupera o campo de momento suavizado usando o método ZZ (patch recovery).
    
    Para cada nó, ajusta um polinômio de grau 2 aos momentos dos nós vizinhos
    (o "patch") por mínimos quadrados, e avalia no nó central.
    
    Args:
        M_h: Array de momentos "brutos" nos nós, shape (n_nodes,).
        positions: Array de posições dos nós, shape (n_nodes,).
        patch_size: Número de nós no patch (deve ser ímpar, mínimo 3).
        
    Returns:
        Array de momentos recuperados (suavizados), shape (n_nodes,).
    """
    n = len(M_h)
    M_star = np.zeros(n)
    
    half_patch = patch_size // 2
    
    for i in range(n):
        # Definir limites do patch
        left = max(0, i - half_patch)
        right = min(n, i + half_patch + 1)
        
        # Garantir tamanho mínimo do patch
        if right - left < 3:
            if left == 0:
                right = min(n, 3)
            else:
                left = max(0, n - 3)
        
        # Extrair dados do patch
        x_patch = positions[left:right]
        M_patch = M_h[left:right]
        
        # Centralizar coordenadas para estabilidade numérica
        x_center = positions[i]
        x_local = x_patch - x_center
        
        # Ajustar polinômio de grau 2: M*(x) = a0 + a1*x + a2*x²
        # Por mínimos quadrados: minimizar ||Va - M||²
        if len(x_local) >= 3:
            V = np.vstack([np.ones_like(x_local), x_local, x_local**2]).T
            coeffs, _, _, _ = np.linalg.lstsq(V, M_patch, rcond=None)
            # Avaliar no ponto central (x_local = 0)
            M_star[i] = coeffs[0]
        else:
            # Fallback: média simples
            M_star[i] = np.mean(M_patch)
    
    return M_star


def compute_zz_error_indicator(
    y_pred: np.ndarray,
    positions: np.ndarray,
    E: float,
    I: float,
    patch_size: int = 3,
) -> np.ndarray:
    """
    Calcula o indicador de erro de Zienkiewicz-Zhu por elemento.
    
    O erro em cada elemento é estimado como a integral da diferença
    quadrática entre o momento recuperado e o momento bruto:
    
    η_e² = ∫_e (M* - M_h)² / (EI) dx
    
    Aproximamos a integral pela regra do trapézio.
    
    Args:
        y_pred: Array de deslocamentos nos nós, shape (n_nodes,).
        positions: Array de posições dos nós, shape (n_nodes,).
        E: Módulo de elasticidade [Pa].
        I: Momento de inércia [m^4].
        patch_size: Tamanho do patch para recuperação ZZ.
        
    Returns:
        Array de indicadores de erro por elemento, shape (n_elements,).
    """
    # Calcular momento bruto
    M_h = compute_moment_from_displacement(y_pred, positions, E, I)
    
    # Recuperar momento suavizado
    M_star = recover_moment_zz(M_h, positions, patch_size)
    
    # Diferença entre momento recuperado e bruto
    delta_M = M_star - M_h
    
    # Calcular erro por elemento (regra do trapézio)
    n_elements = len(positions) - 1
    error = np.zeros(n_elements)
    
    for e in range(n_elements):
        h_e = positions[e + 1] - positions[e]
        
        # Valores nos nós do elemento
        delta_left = delta_M[e]
        delta_right = delta_M[e + 1]
        
        # Integral por regra do trapézio: ∫(ΔM)²dx ≈ h/2 * (ΔM_left² + ΔM_right²)
        integral = h_e / 2 * (delta_left**2 + delta_right**2)
        
        # Normalizar pela rigidez flexional
        error[e] = np.sqrt(integral / (E * I))
    
    return error


def compute_zz_global_error(
    element_errors: np.ndarray,
) -> float:
    """
    Calcula o erro global a partir dos erros por elemento.
    
    η_global = √(Σ η_e²)
    
    Args:
        element_errors: Array de erros por elemento.
        
    Returns:
        Erro global estimado.
    """
    return np.sqrt(np.sum(element_errors**2))


def select_intervals_zz(
    element_errors: np.ndarray,
    strategy: str = "threshold",
    threshold_factor: float = 1.5,
    n_intervals: int = 1,
) -> list[int]:
    """
    Seleciona elementos para refinamento baseado no erro ZZ.
    
    Args:
        element_errors: Array de erros por elemento.
        strategy: Estratégia de seleção:
            - "threshold": refina elementos com erro > threshold_factor * média
            - "top_n": refina os n_intervals elementos com maior erro
            - "fraction": refina a fração top (n_intervals como percentual)
        threshold_factor: Fator multiplicador da média para threshold.
        n_intervals: Número de intervalos (para "top_n") ou percentual (para "fraction").
        
    Returns:
        Lista de índices dos elementos a refinar.
    """
    if strategy == "threshold":
        mean_error = np.mean(element_errors)
        threshold = threshold_factor * mean_error
        indices = np.where(element_errors > threshold)[0].tolist()
        # Se nenhum elemento passar do threshold, pegar o pior
        if not indices:
            indices = [int(np.argmax(element_errors))]
        return indices
    
    elif strategy == "top_n":
        sorted_indices = np.argsort(element_errors)[::-1]
        return sorted_indices[:n_intervals].tolist()
    
    elif strategy == "fraction":
        n_to_refine = max(1, int(len(element_errors) * n_intervals / 100))
        sorted_indices = np.argsort(element_errors)[::-1]
        return sorted_indices[:n_to_refine].tolist()
    
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}")
