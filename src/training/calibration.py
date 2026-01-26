"""
Métricas de calibração para avaliação da qualidade da incerteza estimada.

Este módulo fornece funções para verificar se as estimativas de incerteza
produzidas pelo MC Dropout estão bem calibradas, e para calibrá-las caso
não estejam.

Uma incerteza bem calibrada significa que os intervalos de confiança têm
a cobertura esperada: se o modelo diz "95% de confiança", então 95% dos
valores reais devem cair dentro do intervalo.

Exemplo de uso:
    y_true = ...  # valores reais (numpy array)
    y_pred = ...  # predições médias do MC Dropout (numpy array)
    y_std = ...   # desvios padrão do MC Dropout (numpy array)
    
    # Apenas calcular métricas (sem calibrar)
    metrics = compute_calibration_metrics(y_true, y_pred, y_std)
    
    # Calibrar e obter métricas antes/depois
    result = calibrate_uncertainty(y_true, y_pred, y_std)
    sigma_scale = result["sigma_scale"]
    y_std_calibrated = y_std * sigma_scale
"""

import numpy as np
from numpy.typing import NDArray


def compute_coverage(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    y_std: NDArray[np.floating],
) -> dict[str, float]:
    """
    Calcula a cobertura empírica para diferentes níveis de confiança.
    
    Para uma distribuição Gaussiana, sabemos que uma fração específica dos
    dados deve cair dentro de μ ± k*σ para diferentes valores de k:
    
        k = 0.674 → 50% nominal
        k = 1.000 → 68% nominal  
        k = 1.645 → 90% nominal
        k = 1.960 → 95% nominal
    
    Esta função calcula a fração REAL de pontos que caiu em cada intervalo
    e compara com o valor nominal esperado.
    
    Args:
        y_true: Valores reais (ground truth).
        y_pred: Predições (média do MC Dropout).
        y_std: Desvio padrão estimado (do MC Dropout).
    
    Returns:
        Dicionário com cobertura empírica para cada nível nominal.
        Exemplo: {'coverage_50': 0.48, 'coverage_68': 0.66, ...}
    """
    levels = {
        "coverage_50": 0.674,
        "coverage_68": 1.000,
        "coverage_90": 1.645,
        "coverage_95": 1.960,
    }
    
    results = {}
    
    for name, k in levels.items():
        lower = y_pred - k * y_std
        upper = y_pred + k * y_std
        inside = (y_true >= lower) & (y_true <= upper)
        results[name] = float(np.mean(inside))
    
    return results


def compute_nll(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    y_std: NDArray[np.floating],
) -> float:
    """
    Calcula o Negative Log-Likelihood (NLL) assumindo distribuição Gaussiana.
    
    O NLL para uma Gaussiana com média μ e desvio padrão σ é:
    
        NLL = 0.5 * log(2π) + log(σ) + (y - μ)² / (2σ²)
    
    Esta métrica penaliza tanto erros grandes quando o modelo está confiante
    quanto incerteza alta quando o modelo acerta. Quanto menor o NLL, melhor.
    
    Args:
        y_true: Valores reais (ground truth).
        y_pred: Predições (média do MC Dropout).
        y_std: Desvio padrão estimado (do MC Dropout).
    
    Returns:
        NLL médio sobre todos os pontos.
    """
    eps = 1e-8
    y_std_safe = y_std + eps
    
    var = y_std_safe ** 2
    const = 0.5 * np.log(2 * np.pi)
    
    nll_per_point = const + np.log(y_std_safe) + ((y_true - y_pred) ** 2) / (2 * var)
    
    return float(np.mean(nll_per_point))


def compute_nll_with_scaling(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    y_std: NDArray[np.floating],
    scale_factor: float,
) -> float:
    """
    Calcula o NLL aplicando um fator de escala ao desvio padrão.
    
    Esta função é útil para avaliar diferentes valores do fator de escala T,
    seja durante a otimização ou para gerar gráficos de NLL vs T.
    
    Args:
        y_true: Valores reais (ground truth).
        y_pred: Predições (média do MC Dropout).
        y_std: Desvio padrão original (do MC Dropout).
        scale_factor: Fator T pelo qual multiplicar y_std.
    
    Returns:
        NLL médio com o desvio padrão escalado.
    """
    eps = 1e-8
    scaled_std = y_std * scale_factor + eps
    
    const = 0.5 * np.log(2 * np.pi)
    nll_per_point = (
        const 
        + np.log(scaled_std) 
        + ((y_true - y_pred) ** 2) / (2 * scaled_std ** 2)
    )
    
    return float(np.mean(nll_per_point))


def compute_zscore_stats(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    y_std: NDArray[np.floating],
) -> dict[str, float]:
    """
    Calcula estatísticas dos resíduos padronizados (z-scores).
    
    O z-score é definido como: z = (y_true - y_pred) / σ
    
    Se a incerteza estiver bem calibrada e os erros forem Gaussianos,
    z deveria seguir uma distribuição Normal(0, 1):
        - Média de z ≈ 0 (modelo não tem viés sistemático)
        - Desvio padrão de z ≈ 1 (incerteza está na escala correta)
    
    Se std(z) > 1: modelo está SUPERCONFIANTE (σ muito pequeno)
    Se std(z) < 1: modelo está SUBCONFIANTE (σ muito grande)
    
    Args:
        y_true: Valores reais (ground truth).
        y_pred: Predições (média do MC Dropout).
        y_std: Desvio padrão estimado (do MC Dropout).
    
    Returns:
        Dicionário com média e desvio padrão dos z-scores.
    """
    eps = 1e-8
    y_std_safe = y_std + eps
    
    z = (y_true - y_pred) / y_std_safe
    
    return {
        "zscore_mean": float(np.mean(z)),
        "zscore_std": float(np.std(z)),
    }


def compute_calibration_metrics(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    y_std: NDArray[np.floating],
) -> dict[str, float]:
    """
    Calcula todas as métricas de calibração de incerteza.
    
    Esta é a função principal que agrega todas as métricas em um único
    dicionário. Use esta função no script de treinamento após fazer
    predict_with_uncertainty.
    
    Args:
        y_true: Valores reais (ground truth). Shape (n_samples,).
        y_pred: Predições (média do MC Dropout). Shape (n_samples,).
        y_std: Desvio padrão estimado (do MC Dropout). Shape (n_samples,).
    
    Returns:
        Dicionário com todas as métricas:
        - coverage_50, coverage_68, coverage_90, coverage_95: cobertura empírica
        - nll: Negative Log-Likelihood médio
        - zscore_mean, zscore_std: estatísticas dos resíduos padronizados
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_std = np.asarray(y_std).flatten()
    
    if not (len(y_true) == len(y_pred) == len(y_std)):
        raise ValueError(
            f"Arrays devem ter o mesmo tamanho. "
            f"Recebido: y_true={len(y_true)}, y_pred={len(y_pred)}, y_std={len(y_std)}"
        )
    
    if len(y_true) == 0:
        raise ValueError("Arrays não podem estar vazios.")
    
    metrics = {}
    
    coverage = compute_coverage(y_true, y_pred, y_std)
    metrics.update(coverage)
    
    metrics["nll"] = compute_nll(y_true, y_pred, y_std)
    
    zscore_stats = compute_zscore_stats(y_true, y_pred, y_std)
    metrics.update(zscore_stats)
    
    return metrics


def find_optimal_sigma_scale(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    y_std: NDArray[np.floating],
    bounds: tuple[float, float] = (0.1, 20.0),
) -> dict[str, float]:
    """
    Encontra o fator de escala ótimo que minimiza o NLL.
    
    Esta função busca o valor T que, quando aplicado como σ_calibrado = T × σ_original,
    produz a melhor calibração da incerteza medida pelo Negative Log-Likelihood.
    
    O processo de otimização é rápido (milissegundos) porque estamos otimizando
    apenas um único parâmetro escalar, não os pesos de uma rede neural.
    
    Args:
        y_true: Valores reais (ground truth).
        y_pred: Predições (média do MC Dropout).
        y_std: Desvio padrão original (do MC Dropout).
        bounds: Intervalo de busca para T. O padrão (0.1, 20.0) cobre a maioria
            dos casos práticos. Se T < 1, o modelo era subconfiante (raro).
            Se T > 1, o modelo era superconfiante (comum com MC Dropout).
    
    Returns:
        Dicionário contendo:
        - 'sigma_scale': O valor ótimo de T encontrado.
        - 'nll_before': NLL antes da calibração (com T=1).
        - 'nll_after': NLL após a calibração (com T ótimo).
    """
    from scipy.optimize import minimize_scalar
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_std = np.asarray(y_std).flatten()
    
    nll_before = compute_nll_with_scaling(y_true, y_pred, y_std, scale_factor=1.0)
    
    def objective(t):
        return compute_nll_with_scaling(y_true, y_pred, y_std, scale_factor=t)
    
    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    
    optimal_t = result.x
    nll_after = result.fun
    
    return {
        "sigma_scale": float(optimal_t),
        "nll_before": float(nll_before),
        "nll_after": float(nll_after),
    }


def calibrate_uncertainty(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    y_std: NDArray[np.floating],
) -> dict:
    """
    Calibra a incerteza e retorna métricas completas antes e depois.
    
    Esta função encontra o fator de escala ótimo T, aplica a correção
    σ_calibrado = T × σ_original, e calcula todas as métricas de calibração
    (coverage, NLL, z-scores) tanto antes quanto depois da correção.
    
    Args:
        y_true: Valores reais (ground truth).
        y_pred: Predições (média do MC Dropout).
        y_std: Desvio padrão estimado (do MC Dropout, antes da calibração).
    
    Returns:
        Dicionário contendo:
        - 'sigma_scale': o fator T ótimo encontrado
        - 'nll_before': NLL antes da calibração
        - 'nll_after': NLL após a calibração
        - 'metrics_before': dict com todas as métricas antes da correção
        - 'metrics_after': dict com todas as métricas após aplicar T
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_std = np.asarray(y_std).flatten()
    
    metrics_before = compute_calibration_metrics(y_true, y_pred, y_std)
    
    scale_result = find_optimal_sigma_scale(y_true, y_pred, y_std)
    sigma_scale = scale_result["sigma_scale"]
    
    y_std_calibrated = y_std * sigma_scale
    
    metrics_after = compute_calibration_metrics(y_true, y_pred, y_std_calibrated)
    
    return {
        "sigma_scale": sigma_scale,
        "nll_before": scale_result["nll_before"],
        "nll_after": scale_result["nll_after"],
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    }