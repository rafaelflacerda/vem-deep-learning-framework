"""
Script de refinamento adaptativo de malha.

Suporta duas estratégias de refinamento:
- Guiado por incerteza (MC Dropout)
- Guiado por estimador de erro Zienkiewicz-Zhu (ZZ)

Uso:
    python scripts/adaptive_refinement.py --config scripts/configs/refinement_example.yaml
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.refinement_config import AdaptiveRefinementConfig
from src.inference import (
    BeamPredictor,
    build_beam_graph,
    compute_refinement_indicator,
    create_initial_mesh,
    create_stopping_criterion,
    insert_nodes_at_intervals,
    select_intervals_to_refine,
    # Funções ZZ
    compute_zz_error_indicator,
    compute_zz_global_error,
    select_intervals_zz,
)
from src.paths import ensure_dir, paths
from src.utils.visualization import set_style


def parse_arguments() -> argparse.Namespace:
    """Parseia argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Refinamento adaptativo de malha.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Caminho para arquivo de configuração YAML.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["uncertainty", "zz"],
        default="uncertainty",
        help="Método de refinamento: 'uncertainty' (MC Dropout) ou 'zz' (Zienkiewicz-Zhu).",
    )
    return parser.parse_args()


def plot_iteration(
    positions: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray | None,
    iteration: int,
    output_path: Path,
    n_sigma: float = 2.0,
    zz_errors: np.ndarray | None = None,
) -> None:
    """
    Gera e salva o gráfico de uma iteração.

    Args:
        positions: Posições dos nós.
        y_pred: Predições de deslocamento.
        y_std: Incertezas (pode ser None se usando ZZ).
        iteration: Número da iteração.
        output_path: Caminho para salvar a figura.
        n_sigma: Número de desvios padrão para a banda de incerteza.
        zz_errors: Erros ZZ por elemento (opcional, para visualização).
    """
    if zz_errors is not None:
        # Layout com 2 subplots para ZZ
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
    else:
        fig, ax1 = plt.subplots(figsize=(10, 5))

    # Converter para mm para melhor visualização
    y_pred_mm = y_pred * 1000

    # Viga indeformada
    ax1.plot(positions, np.zeros_like(positions), "k-", linewidth=1.5, label="Indeformada")

    # Predição com banda de incerteza (se disponível)
    if y_std is not None:
        y_std_mm = y_std * 1000
        ax1.fill_between(
            positions,
            y_pred_mm - n_sigma * y_std_mm,
            y_pred_mm + n_sigma * y_std_mm,
            color="red",
            alpha=0.25,
            label=f"±{n_sigma}σ",
        )

    ax1.plot(positions, y_pred_mm, "b--", linewidth=2, label="GNN")

    # Marcar posições dos nós
    ax1.scatter(positions, y_pred_mm, c="blue", s=30, zorder=5)

    ax1.set_xlabel("Posição ao longo da viga (m)")
    ax1.set_ylabel("Deslocamento (mm)")
    ax1.set_title(f"Iteração {iteration} - {len(positions)} nós")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot dos erros ZZ por elemento (se disponível)
    if zz_errors is not None:
        # Posição central de cada elemento
        element_centers = (positions[:-1] + positions[1:]) / 2
        ax2.bar(element_centers, zz_errors, width=np.diff(positions) * 0.8, alpha=0.7, color="orange")
        ax2.set_xlabel("Posição ao longo da viga (m)")
        ax2.set_ylabel("Erro ZZ")
        ax2.set_title("Indicador de Erro ZZ por Elemento")
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_convergence(
    history: list[dict],
    output_path: Path,
    method: str = "uncertainty",
) -> None:
    """
    Gera gráfico de convergência.

    Args:
        history: Lista de dicionários com dados de cada iteração.
        output_path: Caminho para salvar a figura.
        method: Método usado ('uncertainty' ou 'zz').
    """
    iterations = [h["iteration"] for h in history]
    n_nodes_list = [h["n_nodes"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico 1: Métrica de erro vs iteração
    if method == "uncertainty":
        error_metric = [h["max_uncertainty"] * 1000 for h in history]
        ylabel = "Incerteza máxima (mm)"
        title = "Convergência da Incerteza"
    else:
        error_metric = [h["zz_global_error"] for h in history]
        ylabel = "Erro global ZZ"
        title = "Convergência do Erro ZZ"

    axes[0].plot(iterations, error_metric, "o-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Iteração")
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    # Gráfico 2: Número de nós vs iteração
    axes[1].plot(iterations, n_nodes_list, "s-", linewidth=2, markersize=8, color="green")
    axes[1].set_xlabel("Iteração")
    axes[1].set_ylabel("Número de nós")
    axes[1].set_title("Evolução da Malha")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_uncertainty_refinement(
    config: AdaptiveRefinementConfig,
    predictor: BeamPredictor,
    output_dir: Path,
    figures_dir: Path,
) -> list[dict]:
    """
    Executa refinamento guiado por incerteza (MC Dropout).
    
    Returns:
        Histórico das iterações.
    """
    # Criar critério de parada
    stopping_criterion = create_stopping_criterion(
        uncertainty_threshold=config.refinement.uncertainty_threshold,
        max_nodes=config.refinement.max_nodes,
        max_iterations=config.refinement.max_iterations,
    )

    # Criar malha inicial
    positions = create_initial_mesh(config.beam.L, config.refinement.initial_nodes)
    logger.info("Malha inicial: {} nós", len(positions))

    # Loop de refinamento
    history = []
    iteration = 0

    while True:
        iteration += 1
        logger.info("-" * 50)
        logger.info("Iteração {}", iteration)

        # Construir grafo
        graph = build_beam_graph(
            I=config.beam.I,
            L=config.beam.L,
            q=config.beam.q,
            E=config.beam.E,
            positions=positions,
            feature_scaler=predictor.get_feature_scaler(),
        )

        # Fazer predição
        y_pred, y_std = predictor.predict(
            graph,
            mc_samples=config.model.mc_samples,
            apply_calibration=config.model.use_calibration,
        )

        y_pred_np = y_pred.numpy()
        y_std_np = y_std.numpy()

        # Registrar métricas
        max_uncertainty = y_std_np.max()
        mean_uncertainty = y_std_np.mean()

        history.append({
            "iteration": iteration,
            "n_nodes": len(positions),
            "max_uncertainty": max_uncertainty,
            "mean_uncertainty": mean_uncertainty,
            "positions": positions.copy(),
            "y_pred": y_pred_np.copy(),
            "y_std": y_std_np.copy(),
        })

        logger.info("  Nós: {}", len(positions))
        logger.info("  Incerteza máxima: {:.4e} m ({:.4f} mm)", max_uncertainty, max_uncertainty * 1000)
        logger.info("  Incerteza média: {:.4e} m ({:.4f} mm)", mean_uncertainty, mean_uncertainty * 1000)

        # Salvar figura da iteração
        if config.visualization.save_figures and config.visualization.save_each_iteration:
            fig_path = figures_dir / f"iteration_{iteration:03d}.png"
            plot_iteration(positions, y_pred_np, y_std_np, iteration, fig_path)
            logger.info("  Figura salva: {}", fig_path.name)

        # Verificar critério de parada
        should_stop, reason = stopping_criterion.should_stop(
            iteration=iteration,
            n_nodes=len(positions),
            y_std=y_std_np,
            y_pred=y_pred_np,
            positions=positions,
        )

        if should_stop:
            logger.info("Parando: {}", reason)
            break

        # Calcular indicador de refinamento
        indicator = compute_refinement_indicator(
            y_pred=y_pred_np,
            y_std=y_std_np,
            positions=positions,
            lambda_weight=config.refinement.lambda_weight,
        )

        # Selecionar intervalos para refinar
        intervals_to_refine = select_intervals_to_refine(
            indicator,
            n_intervals=config.refinement.nodes_per_iteration,
        )

        logger.info("  Refinando intervalos: {}", intervals_to_refine)

        # Inserir novos nós
        positions = insert_nodes_at_intervals(positions, intervals_to_refine)

    return history


def run_zz_refinement(
    config: AdaptiveRefinementConfig,
    predictor: BeamPredictor,
    output_dir: Path,
    figures_dir: Path,
) -> list[dict]:
    """
    Executa refinamento guiado pelo estimador de erro Zienkiewicz-Zhu.
    
    Returns:
        Histórico das iterações.
    """
    # Parâmetros ZZ (podem ser adicionados à config no futuro)
    zz_threshold = getattr(config.refinement, "zz_threshold", 1e-6)
    zz_strategy = getattr(config.refinement, "zz_strategy", "threshold")
    zz_threshold_factor = getattr(config.refinement, "zz_threshold_factor", 1.5)

    # Criar malha inicial
    positions = create_initial_mesh(config.beam.L, config.refinement.initial_nodes)
    logger.info("Malha inicial: {} nós", len(positions))

    # Loop de refinamento
    history = []
    iteration = 0

    while True:
        iteration += 1
        logger.info("-" * 50)
        logger.info("Iteração {}", iteration)

        # Construir grafo
        graph = build_beam_graph(
            I=config.beam.I,
            L=config.beam.L,
            q=config.beam.q,
            E=config.beam.E,
            positions=positions,
            feature_scaler=predictor.get_feature_scaler(),
        )

        # Fazer predição (sem incerteza para ZZ, mas podemos calcular para comparação)
        y_pred, y_std = predictor.predict(
            graph,
            mc_samples=config.model.mc_samples,
            apply_calibration=config.model.use_calibration,
        )

        y_pred_np = y_pred.numpy()
        y_std_np = y_std.numpy()

        # Calcular erro ZZ
        zz_errors = compute_zz_error_indicator(
            y_pred=y_pred_np,
            positions=positions,
            E=config.beam.E,
            I=config.beam.I,
        )
        zz_global = compute_zz_global_error(zz_errors)
        zz_max = zz_errors.max()

        # Registrar métricas
        history.append({
            "iteration": iteration,
            "n_nodes": len(positions),
            "zz_global_error": zz_global,
            "zz_max_error": zz_max,
            "zz_errors": zz_errors.copy(),
            "max_uncertainty": y_std_np.max(),  # Para comparação
            "positions": positions.copy(),
            "y_pred": y_pred_np.copy(),
            "y_std": y_std_np.copy(),
        })

        logger.info("  Nós: {}", len(positions))
        logger.info("  Erro ZZ global: {:.4e}", zz_global)
        logger.info("  Erro ZZ máximo: {:.4e}", zz_max)
        logger.info("  (Incerteza máxima: {:.4e} m para comparação)", y_std_np.max())

        # Salvar figura da iteração
        if config.visualization.save_figures and config.visualization.save_each_iteration:
            fig_path = figures_dir / f"iteration_{iteration:03d}.png"
            plot_iteration(
                positions, y_pred_np, y_std_np, iteration, fig_path,
                zz_errors=zz_errors
            )
            logger.info("  Figura salva: {}", fig_path.name)

        # Critérios de parada para ZZ
        if zz_global <= zz_threshold:
            logger.info("Parando: Erro ZZ global ({:.2e}) <= threshold ({:.2e})", zz_global, zz_threshold)
            break

        if len(positions) >= config.refinement.max_nodes:
            logger.info("Parando: Número máximo de nós atingido ({} >= {})", len(positions), config.refinement.max_nodes)
            break

        if iteration >= config.refinement.max_iterations:
            logger.info("Parando: Número máximo de iterações atingido ({} >= {})", iteration, config.refinement.max_iterations)
            break

        # Selecionar elementos para refinar
        intervals_to_refine = select_intervals_zz(
            element_errors=zz_errors,
            strategy=zz_strategy,
            threshold_factor=zz_threshold_factor,
            n_intervals=config.refinement.nodes_per_iteration,
        )

        logger.info("  Refinando {} elementos: {}", len(intervals_to_refine), intervals_to_refine)

        # Inserir novos nós
        positions = insert_nodes_at_intervals(positions, intervals_to_refine)

    return history


def main():
    """Executa o refinamento adaptativo."""
    args = parse_arguments()

    # Carregar configuração
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuração não encontrada: {config_path}")

    config = AdaptiveRefinementConfig.from_yaml(config_path)

    # Determinar método
    method = args.method
    logger.info("Método de refinamento: {}", method.upper())

    # Criar diretório de saída
    if config.visualization.output_dir:
        output_dir = Path(config.visualization.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = paths.results.experiments / f"{timestamp}_adaptive_refinement_{method}"

    ensure_dir(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")

    # Configurar logger
    logger.add(output_dir / "refinement.log", level="DEBUG")

    logger.info("=" * 70)
    logger.info("REFINAMENTO ADAPTATIVO DE MALHA")
    logger.info("Método: {}", "Incerteza (MC Dropout)" if method == "uncertainty" else "Zienkiewicz-Zhu")
    logger.info("=" * 70)
    logger.info("Configuração carregada de: {}", config_path)
    logger.info("Saída em: {}", output_dir)

    # Salvar configuração usada
    config.to_yaml(output_dir / "config_used.yaml")

    # Log dos parâmetros da viga
    logger.info("Parâmetros da viga:")
    logger.info("  I = {:.4e} m^4", config.beam.I)
    logger.info("  L = {:.2f} m", config.beam.L)
    logger.info("  q = {:.2f} N/m", config.beam.q)
    logger.info("  E = {:.2e} Pa", config.beam.E)

    # Carregar preditor
    logger.info("Carregando modelo de: {}", config.model.experiment_dir)
    predictor = BeamPredictor(
        experiment_dir=config.model.experiment_dir,
        use_calibration=config.model.use_calibration,
    )
    logger.info("Modelo carregado. Sigma scale = {:.4f}", predictor.sigma_scale)

    # Configurar estilo dos gráficos
    set_style(context="paper", font_family="sans-serif")

    # Executar refinamento
    if method == "uncertainty":
        history = run_uncertainty_refinement(config, predictor, output_dir, figures_dir)
    else:
        history = run_zz_refinement(config, predictor, output_dir, figures_dir)

    # Salvar gráfico de convergência
    if config.visualization.save_figures:
        convergence_path = figures_dir / "convergence.png"
        plot_convergence(history, convergence_path, method=method)
        logger.info("Gráfico de convergência salvo: {}", convergence_path)

        # Salvar figura final
        final_path = figures_dir / "final_result.png"
        zz_errors = history[-1].get("zz_errors") if method == "zz" else None
        plot_iteration(
            history[-1]["positions"],
            history[-1]["y_pred"],
            history[-1]["y_std"],
            len(history),
            final_path,
            zz_errors=zz_errors,
        )
        logger.info("Resultado final salvo: {}", final_path)

    # Salvar histórico
    import torch
    torch.save(history, output_dir / "refinement_history.pt")
    logger.info("Histórico salvo: refinement_history.pt")

    # Resumo final
    logger.info("=" * 70)
    logger.info("REFINAMENTO CONCLUÍDO")
    logger.info("=" * 70)
    logger.info("Iterações: {}", len(history))
    logger.info("Nós finais: {}", history[-1]["n_nodes"])

    if method == "uncertainty":
        logger.info("Incerteza máxima final: {:.4e} m ({:.4f} mm)",
                    history[-1]["max_uncertainty"],
                    history[-1]["max_uncertainty"] * 1000)
    else:
        logger.info("Erro ZZ global final: {:.4e}", history[-1]["zz_global_error"])
        logger.info("Erro ZZ máximo final: {:.4e}", history[-1]["zz_max_error"])

    logger.info("Diretório de saída: {}", output_dir)


if __name__ == "__main__":
    main()