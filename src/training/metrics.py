"""
Funções para cálculo de métricas de avaliação.

Este módulo contém funções independentes para calcular diversas métricas
que avaliam a qualidade das predições de um modelo.
"""

import torch
import numpy as np


def compute_r2(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> float:
    """
    Calcula coeficiente R² (coeficiente de determinação).
    
    R² mede qual fração da variância nos dados é explicada pelo modelo.
    Varia de 0 (modelo péssimo) a 1 (modelo perfeito). Valores negativos
    indicam que o modelo é pior do que sempre prever a média.
    
    Args:
        y_true: Valores verdadeiros (tensor ou numpy array).
        y_pred: Predições do modelo (tensor ou numpy array).
        
    Returns:
        Valor de R² como float entre -∞ e 1.
    """
    y_true_np = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    
    ss_res = ((y_true_np - y_pred_np) ** 2).sum()
    ss_tot = ((y_true_np - y_true_np.mean()) ** 2).sum()
    
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> dict[str, float]:
    """
    Calcula múltiplas métricas de avaliação de uma só vez.
    
    Esta é uma função de conveniência que calcula as quatro métricas mais
    comuns para problemas de regressão. Use esta quando você quer um
    resumo completo da performance.
    
    Args:
        y_true: Valores verdadeiros (tensor ou numpy array).
        y_pred: Predições do modelo (tensor ou numpy array).
        
    Returns:
        Dicionário contendo:
            - 'mse': Mean Squared Error (erro quadrático médio)
            - 'rmse': Root Mean Squared Error (raiz do erro quadrático médio)
            - 'mae': Mean Absolute Error (erro absoluto médio)
            - 'r2': Coeficiente R²
    """
    y_true_np = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    
    # MSE é a média dos erros ao quadrado
    mse = float(((y_true_np - y_pred_np) ** 2).mean())
    
    # RMSE é a raiz quadrada do MSE (volta para a escala original)
    rmse = float(mse ** 0.5)
    
    # MAE é a média dos valores absolutos dos erros (mais robusto a outliers)
    mae = float(np.abs(y_true_np - y_pred_np).mean())
    
    # R² é calculado usando a função separada
    r2 = compute_r2(y_true, y_pred)
    
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}