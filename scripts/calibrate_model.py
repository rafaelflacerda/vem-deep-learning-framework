"""
Script para calibrar a incerteza de um modelo já treinado.

Este script carrega um experimento finalizado, encontra o fator de escala
ótimo T para calibrar a incerteza do MC Dropout (σ_calibrado = T × σ_original),
e salva os resultados no diretório do experimento.

O modelo treinado NÃO é modificado. Apenas o fator de escala T é calculado
e salvo para ser aplicado durante a inferência.

Uso:
    # Calibrar usando caminho do experimento
    python scripts/calibrate_model.py --exp-dir results/experiments/2026-01-28_013220_gnn_beam
    
    /Users/rafaelflacerda/00-projects/vem-deep-learning-framework/results/experiments/2026-01-28_013220_gnn_beam
    
    # Calibrar o experimento mais recente
    python scripts/calibrate_model.py --latest
"""

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.experiment_config import ExperimentConfig
from src.data.dataset import BeamGraphDataset
from src.modeling.gnn import BeamGNN
from src.paths import get_latest_experiment, paths
from src.training.calibration import calibrate_uncertainty
from src.training.validation import split_dataset
from src.utils.visualization import (
    plot_calibration_curve,
    plot_zscore_histogram,
    save_figure,
    set_style,
)
from torch_geometric.loader import DataLoader


def parse_arguments() -> argparse.Namespace:
    """
    Parseia argumentos de linha de comando.
    
    Returns:
        Namespace com os argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Calibra a incerteza de um modelo já treinado.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Calibrar experimento específico
  python scripts/calibrate_model.py --exp-dir results/experiments/2026-01-20_235324_gnn_beam
  
  # Calibrar o experimento mais recente
  python scripts/calibrate_model.py --latest
        """,
    )
    
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Caminho para o diretório do experimento (absoluto ou relativo à raiz do projeto).",
    )
    
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Usa o experimento mais recente em results/experiments/.",
    )
    
    return parser.parse_args()


def get_device() -> torch.device:
    """
    Retorna o melhor device disponível (MPS, CUDA, ou CPU).
    
    Returns:
        Device PyTorch apropriado para o sistema.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Usando device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Usando device: CUDA")
    else:
        device = torch.device("cpu")
        logger.info("Usando device: CPU")
    
    return device


def main():
    """
    Executa a calibração de incerteza para um experimento.
    
    O fluxo é:
    1. Carregar configuração e modelo do experimento
    2. Carregar dataset e criar split de validação
    3. Fazer predições com MC Dropout
    4. Encontrar fator de escala T ótimo
    5. Salvar T e métricas antes/depois da calibração
    """
    args = parse_arguments()
    
    # Determinar diretório do experimento
    if args.latest:
        exp_dir = get_latest_experiment()
        if exp_dir is None:
            logger.error("Nenhum experimento encontrado em results/experiments/")
            sys.exit(1)
    elif args.exp_dir:
        exp_dir = Path(args.exp_dir)
        # Se o caminho não for absoluto, assume que é relativo à raiz do projeto
        if not exp_dir.is_absolute():
            exp_dir = paths.root / exp_dir
    else:
        logger.error("Especifique --exp-dir <caminho> ou --latest")
        sys.exit(1)
    
    # Verificar se o diretório existe
    if not exp_dir.exists():
        logger.error("Diretório não encontrado: {}", exp_dir)
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("CALIBRAÇÃO DE INCERTEZA")
    logger.info("=" * 70)
    logger.info("Experimento: {}", exp_dir.name)
    
    # Verificar arquivos necessários
    config_path = exp_dir / "config_used.yaml"
    model_path = exp_dir / "best_model.pt"
    scalers_path = exp_dir / "scalers.pt"
    
    missing_files = []
    if not config_path.exists():
        missing_files.append(("config_used.yaml", "configuração"))
    if not model_path.exists():
        missing_files.append(("best_model.pt", "modelo"))
    if not scalers_path.exists():
        missing_files.append(("scalers.pt", "scalers"))
    
    if missing_files:
        for filename, description in missing_files:
            logger.error("Arquivo de {} não encontrado: {}", description, exp_dir / filename)
        sys.exit(1)
    
    # Carregar configuração
    logger.info("Carregando configuração...")
    config = ExperimentConfig.from_yaml(config_path)
    
    # Setup device
    device = get_device()
    
    # Carregar dataset
    logger.info("Carregando dataset...")
    dataset_path = (
        paths.data.processed
        / config.data.sampling_method
        / config.data.rho_folder
        / f"dataset_{config.data.dataset_size}.pt"
    )
    
    if not dataset_path.exists():
        logger.error("Dataset não encontrado: {}", dataset_path)
        sys.exit(1)
    
    dataset = BeamGraphDataset(
        pt_path=str(dataset_path),
        feature_scaler=config.data.scaler_type,
        target_scaler=config.data.scaler_type,
        fit_scalers=True,
    )
    
    logger.info("Dataset carregado: {} amostras, {} nós, {} features",
                dataset.n_samples, dataset.n_nodes, dataset.n_features)
    
    # Criar split de validação (mesmo split usado no treinamento)
    train_indices, val_indices = split_dataset(dataset, config.data.val_split)
    logger.info("Split: {} treino, {} validação", len(train_indices), len(val_indices))
    
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
    )
    
    # Carregar modelo treinado
    logger.info("Carregando modelo...")
    model = BeamGNN(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        output_dim=config.model.output_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        activation=config.model.activation,
    )
    
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    # Remove prefixo _orig_mod. se existir (modelo foi salvo com torch.compile)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    
    best_epoch = checkpoint.get("epoch", "?")
    logger.info("Modelo carregado (melhor época: {})", best_epoch)
    
    # Fazer predições com MC Dropout
    logger.info("Executando MC Dropout com {} amostras...", config.evaluation.mc_samples)
    
    model.train()  # Ativa dropout para MC Dropout
    
    all_means = []
    all_stds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            
            # Múltiplos forward passes para MC Dropout
            predictions = []
            for _ in range(config.evaluation.mc_samples):
                out = model(batch.x, batch.edge_index)
                predictions.append(out.squeeze())
            
            predictions = torch.stack(predictions)
            mean = predictions.mean(dim=0)
            std = predictions.std(dim=0)
            
            all_means.append(mean.cpu())
            all_stds.append(std.cpu())
            all_targets.append(batch.y.cpu())
    
    y_mean_scaled = torch.cat(all_means)
    y_std_scaled = torch.cat(all_stds)
    y_true_scaled = torch.cat(all_targets)
    
    # Converter para valores físicos (desfazer normalização)
    y_mean = dataset.inverse_transform_targets(y_mean_scaled)
    y_true = dataset.inverse_transform_targets(y_true_scaled)
    
    # Escalar desvio padrão para valores físicos
    if hasattr(dataset.target_scaler, "std") and dataset.target_scaler.std is not None:
        scale_factor = dataset.target_scaler.std.item() + 1e-8
    elif hasattr(dataset.target_scaler, "max") and dataset.target_scaler.max is not None:
        scale_factor = (dataset.target_scaler.max.item() - dataset.target_scaler.min.item()) + 1e-8
    else:
        scale_factor = 1.0
    
    y_std = y_std_scaled * scale_factor
    
    # Calibrar incerteza
    logger.info("Calibrando incerteza...")
    calibration_result = calibrate_uncertainty(
        y_true.numpy(),
        y_mean.numpy(),
        y_std.numpy(),
    )
    
    sigma_scale = calibration_result["sigma_scale"]
    nll_before = calibration_result["nll_before"]
    nll_after = calibration_result["nll_after"]
    metrics_before = calibration_result["metrics_before"]
    metrics_after = calibration_result["metrics_after"]
    
    # Mostrar resultados
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTADOS DA CALIBRAÇÃO")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Fator de escala ótimo (T): {:.4f}", sigma_scale)
    logger.info("  Interpretação: σ_calibrado = {:.4f} × σ_original", sigma_scale)
    logger.info("")
    logger.info("ANTES da calibração:")
    logger.info("  Coverage 50%: {:.3f} (nominal: 0.50)", metrics_before["coverage_50"])
    logger.info("  Coverage 68%: {:.3f} (nominal: 0.68)", metrics_before["coverage_68"])
    logger.info("  Coverage 90%: {:.3f} (nominal: 0.90)", metrics_before["coverage_90"])
    logger.info("  Coverage 95%: {:.3f} (nominal: 0.95)", metrics_before["coverage_95"])
    logger.info("  NLL: {:.4f}", nll_before)
    logger.info("  Z-score mean: {:.4f} (ideal: 0.0)", metrics_before["zscore_mean"])
    logger.info("  Z-score std: {:.4f} (ideal: 1.0)", metrics_before["zscore_std"])
    logger.info("")
    logger.info("DEPOIS da calibração:")
    logger.info("  Coverage 50%: {:.3f} (nominal: 0.50)", metrics_after["coverage_50"])
    logger.info("  Coverage 68%: {:.3f} (nominal: 0.68)", metrics_after["coverage_68"])
    logger.info("  Coverage 90%: {:.3f} (nominal: 0.90)", metrics_after["coverage_90"])
    logger.info("  Coverage 95%: {:.3f} (nominal: 0.95)", metrics_after["coverage_95"])
    logger.info("  NLL: {:.4f}", nll_after)
    logger.info("  Z-score mean: {:.4f} (ideal: 0.0)", metrics_after["zscore_mean"])
    logger.info("  Z-score std: {:.4f} (ideal: 1.0)", metrics_after["zscore_std"])
    
    # Gerar gráficos de calibração
    logger.info("")
    logger.info("Gerando gráficos de calibração...")
    
    figures_dir = exp_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    set_style(context="paper", font_family="sans-serif")
    
    # 1. Calibration curve (cobertura nominal vs empírica)
    coverage_nominal = [0.50, 0.68, 0.90, 0.95]
    coverage_before = [
        metrics_before["coverage_50"],
        metrics_before["coverage_68"],
        metrics_before["coverage_90"],
        metrics_before["coverage_95"],
    ]
    coverage_after = [
        metrics_after["coverage_50"],
        metrics_after["coverage_68"],
        metrics_after["coverage_90"],
        metrics_after["coverage_95"],
    ]
    
    fig = plot_calibration_curve(
        coverage_nominal=coverage_nominal,
        coverage_before=coverage_before,
        coverage_after=coverage_after,
    )
    save_figure(fig, figures_dir / "calibration_curve", formats=["png"])
    logger.info("Salvo: calibration_curve.png")
    
    # 2. Histograma dos z-scores
    y_std_calibrated = y_std.numpy() * sigma_scale
    
    fig = plot_zscore_histogram(
        y_true=y_true.numpy(),
        y_pred=y_mean.numpy(),
        y_std_before=y_std.numpy(),
        y_std_after=y_std_calibrated,
    )
    save_figure(fig, figures_dir / "zscore_histogram", formats=["png"])
    logger.info("Salvo: zscore_histogram.png")
    
    # Salvar resultados em arquivo .pt
    calibration_pt_path = exp_dir / "calibration.pt"
    torch.save(
        {
            "sigma_scale": sigma_scale,
            "nll_before": nll_before,
            "nll_after": nll_after,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
        },
        calibration_pt_path,
    )
    logger.info("")
    logger.info("Calibração salva em: {}", calibration_pt_path)
    
    # Salvar resultados em arquivo de texto
    calibration_txt_path = exp_dir / "calibration.txt"
    with open(calibration_txt_path, "w") as f:
        f.write("CALIBRAÇÃO DE INCERTEZA\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Experimento: {exp_dir.name}\n")
        f.write(f"Fator de escala ótimo (T): {sigma_scale:.6f}\n")
        f.write(f"Interpretação: σ_calibrado = {sigma_scale:.4f} × σ_original\n\n")
        f.write("ANTES da calibração:\n")
        f.write(f"  Coverage 50%: {metrics_before['coverage_50']:.3f} (nominal: 0.50)\n")
        f.write(f"  Coverage 68%: {metrics_before['coverage_68']:.3f} (nominal: 0.68)\n")
        f.write(f"  Coverage 90%: {metrics_before['coverage_90']:.3f} (nominal: 0.90)\n")
        f.write(f"  Coverage 95%: {metrics_before['coverage_95']:.3f} (nominal: 0.95)\n")
        f.write(f"  NLL: {nll_before:.4f}\n")
        f.write(f"  Z-score mean: {metrics_before['zscore_mean']:.4f} (ideal: 0.0)\n")
        f.write(f"  Z-score std: {metrics_before['zscore_std']:.4f} (ideal: 1.0)\n\n")
        f.write("DEPOIS da calibração:\n")
        f.write(f"  Coverage 50%: {metrics_after['coverage_50']:.3f} (nominal: 0.50)\n")
        f.write(f"  Coverage 68%: {metrics_after['coverage_68']:.3f} (nominal: 0.68)\n")
        f.write(f"  Coverage 90%: {metrics_after['coverage_90']:.3f} (nominal: 0.90)\n")
        f.write(f"  Coverage 95%: {metrics_after['coverage_95']:.3f} (nominal: 0.95)\n")
        f.write(f"  NLL: {nll_after:.4f}\n")
        f.write(f"  Z-score mean: {metrics_after['zscore_mean']:.4f} (ideal: 0.0)\n")
        f.write(f"  Z-score std: {metrics_after['zscore_std']:.4f} (ideal: 1.0)\n")
    logger.info("Métricas salvas em: {}", calibration_txt_path)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("CALIBRAÇÃO CONCLUÍDA!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()