"""
Módulo de visualização para avaliação de modelos.

Fornece funções padronizadas para gerar gráficos de:
- Métricas de erro (scatter, histogramas, erro por posição)
- Perfis de deslocamento (predito vs real)
- Análise de incerteza (correlação, calibração, bandas de confiança)
- Curvas de treinamento (loss)

Uso básico:
    from src.utils.visualization import plot_prediction_scatter, plot_displacement_profile

    fig = plot_prediction_scatter(y_true, y_pred)
    fig.savefig("scatter.png")

Customização:
    Todas as funções aceitam **kwargs que são passados para as funções do matplotlib.
    Além disso, há parâmetros específicos para controlar aparência (cores, fontes, etc).
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# =============================================================================
# CONFIGURAÇÃO GLOBAL DE ESTILO
# =============================================================================


def set_style(
    style: str = "whitegrid",
    context: str = "paper",
    font_scale: float = 1.2,
    palette: str = "deep",
    font_family: str = "sans-serif",
    font_list: list[str] | None = None,
) -> None:
    """
    Configura o estilo global dos gráficos.

    Chame esta função uma vez no início do script para definir o estilo
    de todos os gráficos subsequentes.

    Args:
        style: Estilo seaborn ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks').
        context: Contexto ('paper', 'notebook', 'talk', 'poster').
                 'paper' é menor, 'poster' é maior.
        font_scale: Multiplicador do tamanho de fonte.
        palette: Paleta de cores ('deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind').
        font_family: Família de fonte ('serif', 'sans-serif', 'monospace').
        font_list: Lista de fontes específicas em ordem de preferência.
                   Se None, usa defaults para a família escolhida.
                   Exemplo: ["Helvetica", "Arial", "DejaVu Sans"]
    """
    sns.set_theme(style=style, context=context, font_scale=font_scale, palette=palette)

    plt.rcParams["font.family"] = font_family

    if font_list is not None:
        if font_family == "sans-serif":
            plt.rcParams["font.sans-serif"] = font_list
        elif font_family == "serif":
            plt.rcParams["font.serif"] = font_list
        elif font_family == "monospace":
            plt.rcParams["font.monospace"] = font_list
    else:
        # Defaults com fallbacks seguros
        plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
        plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Georgia"]

    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Converte tensor para numpy array se necessário."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


# =============================================================================
# GRÁFICOS DE MÉTRICAS DE ERRO
# =============================================================================


def plot_prediction_scatter(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    figsize: tuple[float, float] = (6, 6),
    color: str = "steelblue",
    alpha: float = 0.5,
    marker_size: float = 10,
    show_diagonal: bool = True,
    diagonal_color: str = "red",
    diagonal_style: str = "--",
    xlabel: str = "Valor Real",
    ylabel: str = "Valor Predito",
    title: str = "Predito vs Real",
    show_metrics: bool = True,
) -> plt.Figure:
    """
    Scatter plot de valores preditos vs reais.

    Pontos na diagonal indicam predição perfeita. Desvios mostram erros.
    Útil para identificar viés sistemático (modelo sempre super/subestima).

    Args:
        y_true: Valores reais (qualquer shape, será achatado).
        y_pred: Valores preditos (mesmo shape que y_true).
        figsize: Tamanho da figura em polegadas (largura, altura).
        color: Cor dos pontos (nome, hex, ou RGB).
        alpha: Transparência dos pontos (0 a 1).
        marker_size: Tamanho dos marcadores.
        show_diagonal: Se True, mostra linha diagonal de referência.
        diagonal_color: Cor da linha diagonal.
        diagonal_style: Estilo da linha diagonal ('--', '-', ':', '-.').
        xlabel: Rótulo do eixo X.
        ylabel: Rótulo do eixo Y.
        title: Título do gráfico.
        show_metrics: Se True, mostra R² e RMSE no gráfico.

    Returns:
        Figura matplotlib.
    """
    y_true = _to_numpy(y_true).flatten()
    y_pred = _to_numpy(y_pred).flatten()

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_true, y_pred, c=color, alpha=alpha, s=marker_size, edgecolors="none")

    if show_diagonal:
        lims = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max()),
        ]
        ax.plot(
            lims,
            lims,
            diagonal_style,
            color=diagonal_color,
            linewidth=1.5,
            label="Ideal",
        )
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    if show_metrics:
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics_text = f"R² = {r2:.4f}\nRMSE = {rmse:.2e}"
        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    return fig


def plot_error_histogram(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    figsize: tuple[float, float] = (8, 5),
    bins: int = 50,
    color: str = "steelblue",
    edgecolor: str = "white",
    xlabel: str = "Erro (Predito - Real)",
    ylabel: str = "Frequência",
    title: str = "Distribuição dos Erros",
    show_stats: bool = True,
    show_zero_line: bool = True,
) -> plt.Figure:
    """
    Histograma da distribuição dos erros.

    Mostra se os erros são simétricos em torno de zero (modelo sem viés)
    ou se há tendência de super/subestimação. Também mostra a dispersão.

    Args:
        y_true: Valores reais.
        y_pred: Valores preditos.
        figsize: Tamanho da figura.
        bins: Número de bins do histograma.
        color: Cor das barras.
        edgecolor: Cor das bordas das barras.
        xlabel: Rótulo do eixo X.
        ylabel: Rótulo do eixo Y.
        title: Título do gráfico.
        show_stats: Se True, mostra média e desvio padrão.
        show_zero_line: Se True, mostra linha vertical em zero.

    Returns:
        Figura matplotlib.
    """
    y_true = _to_numpy(y_true).flatten()
    y_pred = _to_numpy(y_pred).flatten()
    errors = y_pred - y_true

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(errors, bins=bins, color=color, edgecolor=edgecolor, alpha=0.7)

    if show_zero_line:
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero")

    if show_stats:
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        stats_text = f"Média = {mean_err:.2e}\nStd = {std_err:.2e}"
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout()
    return fig


def plot_error_by_position(
    positions: torch.Tensor | np.ndarray,
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    figsize: tuple[float, float] = (10, 5),
    color: str = "steelblue",
    show_std: bool = True,
    std_alpha: float = 0.3,
    xlabel: str = "Posição ao longo da viga",
    ylabel: str = "Erro absoluto médio",
    title: str = "Erro por Posição na Viga",
) -> plt.Figure:
    """
    Erro médio em função da posição ao longo da viga.

    Revela se há regiões sistemáticas onde a rede tem mais dificuldade.
    Por exemplo, se a rede erra mais perto do engaste ou da ponta livre.

    Args:
        positions: Posições dos nós (n_nodes,).
        y_true: Valores reais de shape (n_samples, n_nodes).
        y_pred: Valores preditos de shape (n_samples, n_nodes).
        figsize: Tamanho da figura.
        color: Cor da linha.
        show_std: Se True, mostra banda de ±1 desvio padrão.
        std_alpha: Transparência da banda de desvio padrão.
        xlabel: Rótulo do eixo X.
        ylabel: Rótulo do eixo Y.
        title: Título do gráfico.

    Returns:
        Figura matplotlib.
    """
    positions = _to_numpy(positions).flatten()
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_pred = y_pred.reshape(1, -1)

    errors = np.abs(y_pred - y_true)
    mean_error = errors.mean(axis=0)
    std_error = errors.std(axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(positions, mean_error, color=color, linewidth=2, label="Erro médio")

    if show_std:
        ax.fill_between(
            positions,
            mean_error - std_error,
            mean_error + std_error,
            color=color,
            alpha=std_alpha,
            label="±1 std",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return fig


# =============================================================================
# GRÁFICOS DE PERFIL DE DESLOCAMENTO
# =============================================================================


def plot_displacement_profile(
    positions: torch.Tensor | np.ndarray,
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    figsize: tuple[float, float] = (10, 5),
    true_color: str = "black",
    pred_color: str = "steelblue",
    true_style: str = "-",
    pred_style: str = "--",
    true_linewidth: float = 2.0,
    pred_linewidth: float = 2.0,
    xlabel: str = "Posição ao longo da viga",
    ylabel: str = "Deslocamento vertical",
    title: str = "Perfil de Deslocamento",
    true_label: str = "VEM (referência)",
    pred_label: str = "GNN (predição)",
) -> plt.Figure:
    """
    Perfil de deslocamento ao longo da viga para uma amostra.

    Compara visualmente a curva de deslocamento predita pela GNN
    com a curva de referência do VEM.

    Args:
        positions: Posições dos nós (n_nodes,).
        y_true: Deslocamentos reais (n_nodes,).
        y_pred: Deslocamentos preditos (n_nodes,).
        figsize: Tamanho da figura.
        true_color: Cor da curva real.
        pred_color: Cor da curva predita.
        true_style: Estilo de linha da curva real.
        pred_style: Estilo de linha da curva predita.
        true_linewidth: Espessura da linha real.
        pred_linewidth: Espessura da linha predita.
        xlabel: Rótulo do eixo X.
        ylabel: Rótulo do eixo Y.
        title: Título do gráfico.
        true_label: Legenda da curva real.
        pred_label: Legenda da curva predita.

    Returns:
        Figura matplotlib.
    """
    positions = _to_numpy(positions).flatten()
    y_true = _to_numpy(y_true).flatten()
    y_pred = _to_numpy(y_pred).flatten()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        positions,
        y_true,
        color=true_color,
        linestyle=true_style,
        linewidth=true_linewidth,
        label=true_label,
    )
    ax.plot(
        positions,
        y_pred,
        color=pred_color,
        linestyle=pred_style,
        linewidth=pred_linewidth,
        label=pred_label,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_multiple_profiles(
    positions: torch.Tensor | np.ndarray,
    y_true_list: Sequence[torch.Tensor | np.ndarray],
    y_pred_list: Sequence[torch.Tensor | np.ndarray],
    figsize: tuple[float, float] = (12, 8),
    n_cols: int = 2,
    true_color: str = "black",
    pred_color: str = "steelblue",
    xlabel: str = "Posição",
    ylabel: str = "Deslocamento",
) -> plt.Figure:
    """
    Múltiplos perfis de deslocamento em subplots.

    Útil para comparar várias amostras lado a lado.

    Args:
        positions: Posições dos nós.
        y_true_list: Lista de arrays com deslocamentos reais.
        y_pred_list: Lista de arrays com deslocamentos preditos.
        figsize: Tamanho da figura.
        n_cols: Número de colunas de subplots.
        true_color: Cor das curvas reais.
        pred_color: Cor das curvas preditas.
        xlabel: Rótulo do eixo X.
        ylabel: Rótulo do eixo Y.

    Returns:
        Figura matplotlib.
    """
    positions = _to_numpy(positions).flatten()
    n_samples = len(y_true_list)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list, strict=False)):
        ax = axes[i]
        y_true = _to_numpy(y_true).flatten()
        y_pred = _to_numpy(y_pred).flatten()

        ax.plot(positions, y_true, color=true_color, linewidth=1.5, label="Real")
        ax.plot(
            positions,
            y_pred,
            color=pred_color,
            linestyle="--",
            linewidth=1.5,
            label="Pred",
        )
        ax.set_title(f"Amostra {i + 1}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if i == 0:
            ax.legend(fontsize=8)

    for i in range(n_samples, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    return fig


# =============================================================================
# GRÁFICOS DE ANÁLISE DE INCERTEZA
# =============================================================================


def plot_uncertainty_vs_error(
    uncertainty: torch.Tensor | np.ndarray,
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    figsize: tuple[float, float] = (6, 6),
    color: str = "steelblue",
    alpha: float = 0.5,
    marker_size: float = 10,
    xlabel: str = "Incerteza (std)",
    ylabel: str = "Erro Absoluto",
    title: str = "Incerteza vs Erro",
    show_correlation: bool = True,
    show_trend: bool = True,
    trend_color: str = "red",
) -> plt.Figure:
    """
    Scatter de incerteza estimada vs erro absoluto real.

    Se a incerteza estiver bem calibrada, deve haver correlação positiva:
    onde a incerteza é alta, o erro também deve ser alto (em média).

    Args:
        uncertainty: Incerteza estimada (std do MC Dropout).
        y_true: Valores reais.
        y_pred: Valores preditos (média do MC Dropout).
        figsize: Tamanho da figura.
        color: Cor dos pontos.
        alpha: Transparência dos pontos.
        marker_size: Tamanho dos marcadores.
        xlabel: Rótulo do eixo X.
        ylabel: Rótulo do eixo Y.
        title: Título do gráfico.
        show_correlation: Se True, mostra coeficiente de correlação.
        show_trend: Se True, mostra linha de tendência.
        trend_color: Cor da linha de tendência.

    Returns:
        Figura matplotlib.
    """
    uncertainty = _to_numpy(uncertainty).flatten()
    y_true = _to_numpy(y_true).flatten()
    y_pred = _to_numpy(y_pred).flatten()
    error = np.abs(y_true - y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        uncertainty, error, c=color, alpha=alpha, s=marker_size, edgecolors="none"
    )

    if show_trend:
        z = np.polyfit(uncertainty, error, 1)
        p = np.poly1d(z)
        x_line = np.linspace(uncertainty.min(), uncertainty.max(), 100)
        ax.plot(
            x_line,
            p(x_line),
            color=trend_color,
            linestyle="--",
            linewidth=1.5,
            label="Tendência",
        )

    if show_correlation:
        corr = np.corrcoef(uncertainty, error)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"Correlação = {corr:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show_trend:
        ax.legend()

    fig.tight_layout()
    return fig


def plot_calibration(
    uncertainty: torch.Tensor | np.ndarray,
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    n_bins: int = 10,
    figsize: tuple[float, float] = (7, 5),
    bar_color: str = "steelblue",
    line_color: str = "red",
    xlabel: str = "Bin (ordenado por incerteza)",
    ylabel: str = "Valor normalizado [0, 1]",
    title: str = "Calibração da Incerteza",
    normalize: bool = True,
) -> plt.Figure:
    """
    Gráfico de calibração da incerteza.

    Divide as predições em bins por nível de incerteza e calcula o erro
    médio em cada bin. Se a incerteza estiver calibrada, bins com maior
    incerteza devem ter maior erro médio.

    Args:
        uncertainty: Incerteza estimada.
        y_true: Valores reais.
        y_pred: Valores preditos.
        n_bins: Número de bins.
        figsize: Tamanho da figura.
        bar_color: Cor das barras (erro).
        line_color: Cor da linha (incerteza).
        xlabel: Rótulo do eixo X.
        ylabel: Rótulo do eixo Y.
        title: Título do gráfico.
        normalize: Se True, normaliza erro e incerteza para [0, 1].
                   Se False, usa dois eixos Y com escalas diferentes.

    Returns:
        Figura matplotlib.
    """
    uncertainty = _to_numpy(uncertainty).flatten()
    y_true = _to_numpy(y_true).flatten()
    y_pred = _to_numpy(y_pred).flatten()
    error = np.abs(y_true - y_pred)

    # Dividir em bins por percentil de incerteza
    bin_edges = np.percentile(uncertainty, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(uncertainty, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    mean_uncertainties = []
    mean_errors = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_uncertainties.append(uncertainty[mask].mean())
            mean_errors.append(error[mask].mean())

    mean_uncertainties = np.array(mean_uncertainties)
    mean_errors = np.array(mean_errors)

    fig, ax = plt.subplots(figsize=figsize)

    x_pos = np.arange(len(mean_uncertainties))
    bar_width = 0.35

    if normalize:
        # Normalizar para [0, 1]
        if mean_errors.max() > mean_errors.min():
            errors_norm = (mean_errors - mean_errors.min()) / (
                mean_errors.max() - mean_errors.min()
            )
        else:
            errors_norm = np.zeros_like(mean_errors)

        if mean_uncertainties.max() > mean_uncertainties.min():
            unc_norm = (mean_uncertainties - mean_uncertainties.min()) / (
                mean_uncertainties.max() - mean_uncertainties.min()
            )
        else:
            unc_norm = np.zeros_like(mean_uncertainties)

        # Barras lado a lado
        ax.bar(
            x_pos - bar_width / 2,
            errors_norm,
            bar_width,
            color=bar_color,
            alpha=0.7,
            label="Erro absoluto médio (norm.)",
        )
        ax.bar(
            x_pos + bar_width / 2,
            unc_norm,
            bar_width,
            color=line_color,
            alpha=0.7,
            label="Incerteza média (norm.)",
        )

        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper left")

    else:
        # Dois eixos Y (comportamento original)
        ax.bar(x_pos, mean_errors, color=bar_color, alpha=0.7, edgecolor="white")
        ax.set_ylabel("Erro absoluto médio", color=bar_color)
        ax.tick_params(axis="y", labelcolor=bar_color)

        ax2 = ax.twinx()
        ax2.plot(
            x_pos,
            mean_uncertainties,
            color=line_color,
            marker="o",
            linewidth=2,
            label="Incerteza média",
        )
        ax2.set_ylabel("Incerteza média", color=line_color)
        ax2.tick_params(axis="y", labelcolor=line_color)

    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{i + 1}" for i in range(len(mean_uncertainties))])

    fig.tight_layout()
    return fig


def plot_coverage(
    uncertainty: torch.Tensor | np.ndarray,
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    confidence_levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95, 0.99),
    figsize: tuple[float, float] = (7, 5),
    bar_color: str = "steelblue",
    ideal_color: str = "red",
    xlabel: str = "Nível de confiança esperado",
    ylabel: str = "Cobertura observada",
    title: str = "Análise de Cobertura",
) -> plt.Figure:
    """
    Gráfico de cobertura dos intervalos de confiança.

    Verifica se os intervalos de confiança baseados na incerteza
    contêm a proporção esperada de valores reais.
    Ex: intervalo de 95% deveria conter ~95% dos valores reais.

    Args:
        uncertainty: Incerteza estimada (std).
        y_true: Valores reais.
        y_pred: Valores preditos (média).
        confidence_levels: Níveis de confiança a testar.
        figsize: Tamanho da figura.
        bar_color: Cor das barras.
        ideal_color: Cor da linha ideal.
        xlabel: Rótulo do eixo X.
        ylabel: Rótulo do eixo Y.
        title: Título do gráfico.

    Returns:
        Figura matplotlib.
    """
    from scipy import stats

    uncertainty = _to_numpy(uncertainty).flatten()
    y_true = _to_numpy(y_true).flatten()
    y_pred = _to_numpy(y_pred).flatten()

    observed_coverages = []

    for conf in confidence_levels:
        z = stats.norm.ppf((1 + conf) / 2)
        lower = y_pred - z * uncertainty
        upper = y_pred + z * uncertainty
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        observed_coverages.append(coverage)

    observed_coverages = np.array(observed_coverages)
    confidence_levels = np.array(confidence_levels)

    fig, ax = plt.subplots(figsize=figsize)

    x_pos = np.arange(len(confidence_levels))
    ax.bar(x_pos, observed_coverages, color=bar_color, alpha=0.7, edgecolor="white")
    ax.plot(
        x_pos,
        confidence_levels,
        color=ideal_color,
        marker="o",
        linewidth=2,
        label="Ideal",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{int(c * 100)}%" for c in confidence_levels])
    ax.set_ylim(0, 1.05)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_profile_with_uncertainty(
    positions: torch.Tensor | np.ndarray,
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    uncertainty: torch.Tensor | np.ndarray,
    figsize: tuple[float, float] = (10, 5),
    true_color: str = "black",
    pred_color: str = "steelblue",
    uncertainty_alpha: float = 0.3,
    n_std: float = 2.0,
    xlabel: str = "Posição ao longo da viga",
    ylabel: str = "Deslocamento vertical",
    title: str = "Perfil com Banda de Incerteza",
    true_label: str = "VEM (referência)",
    pred_label: str = "GNN (predição)",
    uncertainty_label: str = "±2σ (95%)",
) -> plt.Figure:
    """
    Perfil de deslocamento com banda de incerteza.

    Mostra a predição média e uma banda representando o intervalo
    de confiança baseado na incerteza do MC Dropout.

    Args:
        positions: Posições dos nós.
        y_true: Deslocamentos reais.
        y_pred: Deslocamentos preditos (média).
        uncertainty: Incerteza (std do MC Dropout).
        figsize: Tamanho da figura.
        true_color: Cor da curva real.
        pred_color: Cor da curva predita e banda.
        uncertainty_alpha: Transparência da banda.
        n_std: Número de desvios padrão para a banda (2 ≈ 95%).
        xlabel: Rótulo do eixo X.
        ylabel: Rótulo do eixo Y.
        title: Título do gráfico.
        true_label: Legenda da curva real.
        pred_label: Legenda da curva predita.
        uncertainty_label: Legenda da banda de incerteza.

    Returns:
        Figura matplotlib.
    """
    positions = _to_numpy(positions).flatten()
    y_true = _to_numpy(y_true).flatten()
    y_pred = _to_numpy(y_pred).flatten()
    uncertainty = _to_numpy(uncertainty).flatten()

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(
        positions,
        y_pred - n_std * uncertainty,
        y_pred + n_std * uncertainty,
        color=pred_color,
        alpha=uncertainty_alpha,
        label=uncertainty_label,
    )

    ax.plot(positions, y_true, color=true_color, linewidth=2, label=true_label)
    ax.plot(
        positions,
        y_pred,
        color=pred_color,
        linewidth=2,
        linestyle="--",
        label=pred_label,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return fig


# =============================================================================
# GRÁFICOS DE TREINAMENTO
# =============================================================================


def plot_loss_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float] | None = None,
    figsize: tuple[float, float] = (8, 5),
    train_color: str = "#037A68",
    val_color: str = "#E39774",
    xlabel: str = "Epochs",
    ylabel=r"MSE (Mean Squared Error) - [$m^2$]",
    title: str = "Loss Curves",
    log_scale: bool = False,
    train_label: str = "Training",
    val_label: str = "Validation",
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    yticks_step: float | None = None,
    show_minor_grid: bool = False,
    minor_grid_subdivisions: int = 2,
) -> plt.Figure:
    """
    Curvas de loss de treinamento e validação.

    Args:
        train_losses: Lista de losses de treinamento por época.
        val_losses: Lista de losses de validação (opcional).
        figsize: Tamanho da figura.
        train_color: Cor da curva de treino.
        val_color: Cor da curva de validação.
        xlabel: Rótulo do eixo X.
        ylabel: Rótulo do eixo Y.
        title: Título do gráfico.
        log_scale: Se True, usa escala logarítmica no eixo Y.
        train_label: Legenda da curva de treino.
        val_label: Legenda da curva de validação.
        ylim: Limites do eixo Y como tupla (min, max).
        xlim: Limites do eixo X como tupla (min, max).
        yticks_step: Incremento entre ticks do eixo Y (ex: 0.1).
        show_minor_grid: Se True, mostra grid intermediário.
        minor_grid_subdivisions: Número de divisões entre cada tick principal.

    Returns:
        Figura matplotlib.
    """
    from matplotlib.ticker import AutoMinorLocator

    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color=train_color, linewidth=2, label=train_label)

    if val_losses is not None:
        ax.plot(epochs, val_losses, color=val_color, linewidth=2, label=val_label)

    if log_scale:
        ax.set_yscale("log")
        # Adicionar minor ticks em escala logarítmica
        from matplotlib.ticker import LogLocator

        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=10))
        ax.grid(which="minor", linestyle=":", linewidth=0.6, alpha=0.65)

    # Definir limites dos eixos
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, len(train_losses))

    if ylim is not None:
        ax.set_ylim(ylim)

    # Definir ticks do eixo Y com incremento específico
    if yticks_step is not None and not log_scale:
        y_min, y_max = ax.get_ylim()
        yticks = np.arange(0, y_max + yticks_step, yticks_step)
        ax.set_yticks(yticks)

    # Adicionar grid intermediário (minor grid)
    if show_minor_grid and not log_scale:
        ax.yaxis.set_minor_locator(AutoMinorLocator(minor_grid_subdivisions))
        ax.xaxis.set_minor_locator(AutoMinorLocator(minor_grid_subdivisions))
        ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return fig


# =============================================================================
# FUNÇÃO DE CONVENIÊNCIA PARA SALVAR
# =============================================================================


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    formats: Sequence[str] = ("png",),
    close: bool = True,
) -> None:
    """
    Salva figura em um ou mais formatos.

    Args:
        fig: Figura matplotlib.
        path: Caminho base (sem extensão).
        formats: Formatos para salvar ('png', 'pdf', 'svg').
        close: Se True, fecha a figura após salvar.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"))

    if close:
        plt.close(fig)


def plot_r2_curves(
    train_r2: Sequence[float],
    val_r2: Sequence[float] | None = None,
    figsize: tuple[float, float] = (8, 5),
    train_color: str = "#037A68",
    val_color: str = "#E39774",
    xlabel: str = "Época",
    ylabel: str = "R²",
    title: str = "Curvas de R²",
    train_label: str = "Treino",
    val_label: str = "Validação",
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Curvas de R² (coeficiente de determinação) ao longo do treinamento.

    Equivalente às "accuracy curves" para problemas de regressão.
    R² = 1 significa predição perfeita, R² = 0 significa que o modelo
    é tão bom quanto prever a média.

    Args:
        train_r2: Lista de R² no conjunto de treino por época.
        val_r2: Lista de R² no conjunto de validação (opcional).
        figsize: Tamanho da figura.
        train_color: Cor da curva de treino.
        val_color: Cor da curva de validação.
        xlabel: Rótulo do eixo X.
        ylabel: Rótulo do eixo Y.
        title: Título do gráfico.
        train_label: Legenda da curva de treino.
        val_label: Legenda da curva de validação.
        ylim: Limites do eixo Y.
        xlim: Limites do eixo X.

    Returns:
        Figura matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_r2) + 1)
    ax.plot(epochs, train_r2, color=train_color, linewidth=2, label=train_label)

    if val_r2 is not None:
        ax.plot(epochs, val_r2, color=val_color, linewidth=2, label=val_label)

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, len(train_r2))

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_beam_deformation_comparison(
    positions: np.ndarray,
    y_undeformed: np.ndarray | None = None,
    y_vem: np.ndarray = None,
    y_nn: np.ndarray = None,
    y_nn_std: np.ndarray | None = None,
    figsize: tuple[float, float] = (10, 6),
    scale_factor: float = 1.0,
    undeformed_color: str = "black",
    vem_color: str = "#037A68",
    nn_color: str = "#326273",
    uncertainty_color: str = "red",
    uncertainty_alpha: float = 0.25,
    xlabel: str = "Posição ao longo da viga (m)",
    ylabel: str = "Deslocamento vertical (m)",
    title: str = "Comparação de Deformações",
    show_undeformed: bool = True,
    show_uncertainty: bool = True,
    n_sigma: float = 2.0,
) -> plt.Figure:
    """
    Compara a deformação da viga: indeformada, VEM (referência) e NN (predição).

    Args:
        positions: Posições dos nós ao longo da viga.
        y_undeformed: Deslocamentos da viga indeformada (zeros ou None).
        y_vem: Deslocamentos calculados pelo VEM (referência).
        y_nn: Deslocamentos preditos pela NN.
        y_nn_std: Desvio padrão da predição (incerteza do MC Dropout).
        figsize: Tamanho da figura.
        scale_factor: Fator de escala para exagerar deformações (para visualização).
        undeformed_color: Cor da viga indeformada.
        vem_color: Cor da solução VEM.
        nn_color: Cor da predição NN.
        uncertainty_color: Cor da banda de incerteza.
        uncertainty_alpha: Transparência da banda de incerteza.
        xlabel: Rótulo do eixo X.
        ylabel: Rótulo do eixo Y.
        title: Título do gráfico.
        show_undeformed: Se True, mostra a viga indeformada.
        show_uncertainty: Se True, mostra a banda de incerteza.
        n_sigma: Número de desvios padrão para a banda de incerteza.

    Returns:
        Figura matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Viga indeformada (linha reta em y=0)
    if show_undeformed:
        y_undef = np.zeros_like(positions) if y_undeformed is None else y_undeformed
        ax.plot(
            positions,
            y_undef * scale_factor,
            color=undeformed_color,
            linewidth=2,
            linestyle="-",
            label="Indeformada",
        )

    # Solução VEM (referência)
    if y_vem is not None:
        ax.plot(
            positions,
            y_vem * scale_factor,
            color=vem_color,
            linewidth=2,
            linestyle="-",
            label="VEM (Referência)",
        )

    # Predição NN
    if y_nn is not None:
        ax.plot(
            positions,
            y_nn * scale_factor,
            color=nn_color,
            linewidth=2,
            linestyle="--",
            label="GNN (Predição)",
        )

        # Banda de incerteza
        if show_uncertainty and y_nn_std is not None:
            lower = (y_nn - n_sigma * y_nn_std) * scale_factor
            upper = (y_nn + n_sigma * y_nn_std) * scale_factor
            ax.fill_between(
                positions,
                lower,
                upper,
                color=uncertainty_color,
                alpha=uncertainty_alpha,
                label=f"Incerteza (±{n_sigma}σ)",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Inverter eixo Y se deslocamentos são negativos (convenção de engenharia)
    if y_vem is not None and y_vem.min() < 0:
        ax.invert_yaxis()

    fig.tight_layout()
    return fig


def plot_beam_cases_comparison(
    positions: np.ndarray,
    cases_data: list[dict],
    figsize: tuple[float, float] = (15, 5),
    scale_factor: float = 1.0,
    scale_y: float = 1000.0,
    undeformed_color: str = "black",
    vem_color: str = "#037A68",
    nn_color: str = "#326273",
    uncertainty_color: str = "red",
    uncertainty_alpha: float = 0.25,
    n_sigma: float = 2.0,
) -> plt.Figure:
    """
    Plota três casos lado a lado: melhor, mediano e pior.

    Args:
        positions: Posições dos nós ao longo da viga.
        cases_data: Lista de 3 dicionários, cada um com:
            - 'y_vem': deslocamentos VEM
            - 'y_nn': deslocamentos NN
            - 'y_nn_std': incerteza NN
            - 'title': título do subplot
            - 'error': valor do erro (para anotação)
        figsize: Tamanho da figura.
        scale_factor: Fator de escala para deformações.
        undeformed_color: Cor da viga indeformada.
        vem_color: Cor da solução VEM.
        nn_color: Cor da predição NN.
        uncertainty_color: Cor da banda de incerteza.
        uncertainty_alpha: Transparência da banda.
        n_sigma: Número de desvios padrão para banda.

    Returns:
        Figura matplotlib.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for ax, case in zip(axes, cases_data, strict=False):
        y_vem = case["y_vem"]
        y_nn = case["y_nn"]
        y_nn_std = case.get("y_nn_std")
        title = case.get("title", "")
        error = case.get("error")

        # Viga indeformada
        ax.plot(
            positions,
            np.zeros_like(positions) * scale_y,
            color=undeformed_color,
            linewidth=1.5,
            linestyle="-",
            label="Indeformada",
        )

        # Solução VEM
        ax.plot(
            positions,
            y_vem * scale_factor * scale_y,
            color=vem_color,
            linewidth=2,
            linestyle="-",
            label="VEM",
        )

        # Predição NN
        ax.plot(
            positions,
            y_nn * scale_factor * scale_y,
            color=nn_color,
            linewidth=2,
            linestyle="--",
            label="GNN",
        )

        # Banda de incerteza
        if y_nn_std is not None:
            lower = (y_nn - n_sigma * y_nn_std) * scale_factor * scale_y
            upper = (y_nn + n_sigma * y_nn_std) * scale_factor * scale_y
            ax.fill_between(
                positions,
                lower,
                upper,
                color=uncertainty_color,
                alpha=uncertainty_alpha,
                label=f"±{n_sigma}σ",
            )

        ax.set_xlabel("Posição (m)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Anotação do erro
        if error is not None:
            ax.annotate(
                f"MSE: {error:.2e}",
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    # Ylabel apenas no primeiro subplot
    axes[0].set_ylabel("Deslocamento (mm)")

    # Legenda compartilhada
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))

    # # Inverter eixo Y se necessário
    # if cases_data[0]['y_vem'].min() < 0:
    #     axes[0].invert_yaxis()

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)  # Espaço para legenda

    return fig
