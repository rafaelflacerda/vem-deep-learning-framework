"""
Script principal de treinamento da GNN para predição de deslocamentos em vigas 1D.

Este script orquestra o treinamento completo: carrega configurações,
valida parâmetros, setup de ambiente, executa treinamento via BeamGNNTrainer,
avaliação com incerteza, e geração de gráficos.

Uso:
    # Usar configuração padrão
    python scripts/train_gnn.py
    
    # Usar configuração customizada
    python scripts/train_gnn.py --config-path scripts/configs/sweep_random.yaml
    
    # Usar configuração com override de parâmetro
    python scripts/train_gnn.py --config-path scripts/configs/default.yaml \\
        --override training.learning_rate=1e-3 training.epochs=100
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger

import json

import wandb

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.experiment_config import ExperimentConfig
from src.data.dataset import BeamGraphDataset
from src.paths import create_experiment_dir, ensure_dir, paths
from src.training.trainer import BeamGNNTrainer
from src.training.validation import split_dataset
from src.training.metrics import compute_metrics
from src.utils.logger import configure_logger
from src.utils.visualization import (
    plot_beam_cases_comparison_variable,
    plot_beam_cases_comparison,
    plot_loss_curves,
    plot_r2_curves,
    save_figure,
    set_style,
)
from torch_geometric.loader import DataLoader


# =============================================================================
# ARGUMENTOS DE LINHA DE COMANDO
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parseia argumentos de linha de comando.
    
    Permite especificar:
    - --config-path: caminho para arquivo YAML de configuração
    - --override: sobrescrever valores de config no formato chave=valor
    
    Returns:
        Namespace com argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Treina GNN para predição em vigas 1D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Usar configuração padrão
  python scripts/train_gnn.py
  
  # Usar arquivo de config customizado
  python scripts/train_gnn.py --config-path scripts/configs/test_small.yaml
  
  # Override de hiperparâmetros
  python scripts/train_gnn.py \\
    --override training.learning_rate=1e-3 training.epochs=200
  
  # Combinar config + override
  python scripts/train_gnn.py --config-path scripts/configs/test.yaml \\
    --override model.hidden_dim=128 data.dataset_size=10000
        """,
    )
    
    parser.add_argument(
        "--config-path",
        type=str,
        default="scripts/configs/default.yaml",
        help="Caminho para arquivo de configuração YAML (padrão: scripts/configs/default.yaml)",
    )
    
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Sobrescrever valores de config. Formato: chave=valor (ex: training.learning_rate=1e-3)",
    )
    
    return parser.parse_args()


def apply_config_overrides(config: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    """
    Aplica sobrescrituras (overrides) na configuração.
    
    Percorre os overrides no formato "chave.subchave=valor" e atualiza
    a configuração. Por exemplo, "training.learning_rate=1e-3" vai navegar
    até config.training.learning_rate e atualizar para 1e-3.
    
    Args:
        config: Configuração a atualizar.
        overrides: Lista de strings no formato "chave=valor".
        
    Returns:
        Configuração atualizada.
    """
    if not overrides:
        return config
    
    # Converter config para dicionário para fazer updates
    config_dict = config.model_dump()
    
    for override in overrides:
        if "=" not in override:
            logger.warning("Override mal formatado (esperado 'chave=valor'): {}", override)
            continue
        
        key_path, value_str = override.split("=", 1)
        keys = key_path.split(".")
        
        # Navegar até a chave, criando dicts se necessário
        current = config_dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Tentar fazer parsing do valor (inferir tipo)
        final_key = keys[-1]
        try:
            # Tentar como float (incluindo notação científica)
            if "." in value_str or "e" in value_str.lower():
                try:
                    current[final_key] = float(value_str)
                except ValueError:
                    # Se falhar como float, manter como string
                    current[final_key] = value_str
            # Tentar como int
            elif value_str.isdigit() or (value_str[0] == "-" and value_str[1:].isdigit()):
                current[final_key] = int(value_str)
            # Tentar como booleano
            elif value_str.lower() in ("true", "false"):
                current[final_key] = value_str.lower() == "true"
            # Manter como string
            else:
                current[final_key] = value_str
        except (ValueError, IndexError) as e:
            logger.warning("Erro ao parsear override '{}': {}", override, e)
            continue
        
        logger.info("Override aplicado: {} = {}", key_path, current[final_key])
    
    # Recriar config com updates
    return ExperimentConfig(**config_dict)


def get_device() -> torch.device:
    """
    Retorna o melhor device disponível: CUDA > MPS > CPU.
    
    Prioridade:
    1. CUDA (NVIDIA GPUs) - melhor performance
    2. MPS (Apple Silicon) - boa performance em Macs
    3. CPU - fallback
    
    Returns:
        Device PyTorch para treinamento.
    """
    # Tentar CUDA primeiro (NVIDIA)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("Usando device: CUDA")
        logger.info("GPU: {} ({:.1f} GB VRAM)", gpu_name, gpu_memory)
        
        # Configurações de otimização para CUDA
        torch.backends.cudnn.benchmark = True  # Otimiza convoluções
        torch.backends.cuda.matmul.allow_tf32 = True  # Usa TF32 para matmul
        torch.backends.cudnn.allow_tf32 = True  # Usa TF32 para convs
        
        return device


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """
    Orquestra o treinamento completo de um modelo GNN.
    
    Fluxo:
    1. Parsear argumentos e carregar configuração
    2. Setup de logger, device, diretório de experimento
    3. Carregar dataset
    4. Criar e executar trainer (fixed split ou k-fold)
    5. Fazer avaliação final com incerteza (MC Dropout)
    6. Gerar gráficos de resultado
    7. Salvar artefatos (modelo, métricas, histórico)
    """
    
    # =========================================================================
    # STEP 1: CARREGAR E VALIDAR CONFIGURAÇÃO
    # =========================================================================
    
    args = parse_arguments()
    
    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
    
    logger.info("Carregando configuração de: {}", config_path)
    config = ExperimentConfig.from_yaml(config_path)
    
    # Aplicar overrides se fornecidos
    if args.override:
        logger.info("Aplicando overrides de linha de comando...")
        config = apply_config_overrides(config, args.override)
    
    # =========================================================================
    # STEP 2: SETUP INICIAL (LOGGER, DEVICE, DIRETÓRIO)
    # =========================================================================
    
    exp_dir = create_experiment_dir("gnn_beam")
    figures_dir = ensure_dir(exp_dir / "figures")
    
    configure_logger(
        category="training",
        experiment_name=f"gnn_beam_{config.data.dataset_size}",
        log_dir=exp_dir,
    )
    
    logger.info("=" * 70)
    logger.info("TREINAMENTO GNN - PREDIÇÃO EM VIGAS 1D")
    logger.info("=" * 70)
    logger.info("Diretório do experimento: {}", exp_dir)
    
    # Log da configuração
    config_dict = config.model_dump()
    for section_name, section_data in config_dict.items():
        if isinstance(section_data, dict):
            logger.info("{}:", section_name.upper())
            for key, value in section_data.items():
                logger.info("  {}: {}", key, value)
        else:
            logger.info("{}: {}", section_name, section_data)
        
    # Salvar configuração IMEDIATAMENTE para garantir rastreabilidade
    config.to_yaml(exp_dir / "config_used.yaml")
    logger.info("Configuração salva em: {}/config_used.yaml", exp_dir)

    # Salvar também em JSON para facilitar parsing automático depois
    config_json = exp_dir / "config_used.json"
    with open(config_json, 'w', encoding='utf-8') as f:
        json.dump(config.model_dump(), f, indent=2)
    logger.info("Configuração salva também em: {}/config_used.json", exp_dir)
    
    device = get_device()
    
    wandb.init(
        entity = "rafaelflacerda-poli-usp",
        project =  "tcc-vem-deep-learning",
        name = f"exp_{exp_dir.name}",
    )
    
    logger.info("W&B inicializado: projeto beam-gnn-hpo, experimento: {}", exp_dir.name)
    
    # =========================================================================
    # STEP 3: CARREGAR DATASET
    # =========================================================================
    
    logger.info("Carregando dataset...")
    
    dataset_path = (
        paths.data.processed
        / config.data.sampling_method
        / config.data.rho_folder
        / f"dataset_{config.data.dataset_size}.pt"
    )
    
    if not dataset_path.exists():
        logger.error("Dataset não encontrado: {}", dataset_path)
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")
    
    dataset = BeamGraphDataset(
        pt_path=str(dataset_path),
        feature_scaler=config.data.scaler_type,
        target_scaler=config.data.scaler_type,
        fit_scalers=True,
    )
    
    logger.info("Dataset carregado: {} amostras, {} features",
                dataset.n_samples, dataset.n_features)
    logger.info("Elementos por grafo: min={}, max={}, média={:.1f}",
                dataset.metadata["n_elements_min"],
                dataset.metadata["n_elements_max"],
                dataset.metadata["n_elements_mean"])
    
    # Extrair posições dos nós para gráficos finais
    sample_data = torch.load(dataset_path, weights_only=False)
    feature_names = sample_data["metadata"]["feature_names"]["all"]
    x_idx = feature_names.index("x")
    #positions = sample_data["features"][0, :, x_idx].numpy()
    
    # =========================================================================
    # STEP 4: INSTANCIAR TRAINER E EXECUTAR TREINAMENTO
    # =========================================================================
    
    logger.info("Inicializando trainer...")
    trainer = BeamGNNTrainer(config, device)
    
    # Executar treinamento (fixed split ou k-fold)
    if config.cv_mode == "fixed":
        logger.info("Modo de validação: split fixo")
        
        # Dividir dataset
        train_indices, val_indices = split_dataset(
            dataset, config.data.val_split
        )
        logger.info("Split: {} treino, {} validação", len(train_indices), len(val_indices))
        
        # Criar dataloaders
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers = 4,
            pin_memory = True,
            persistent_workers = True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers = 4,
            pin_memory = True,
            persistent_workers = True,
        )
        
        # Treinar
        logger.info("Iniciando treinamento...")
        training_result = trainer.run_fixed_split(train_loader, val_loader, exp_dir)
        
        train_losses = training_result["train_losses"]
        val_losses = training_result["val_losses"]
        train_r2s = training_result["train_r2s"]
        val_r2s = training_result["val_r2s"]
        best_epoch = training_result["best_epoch"]
        
    elif config.cv_mode == "kfold":
        logger.info("Modo de validação: {}-fold cross-validation", config.n_folds)
        
        # Treinar com k-fold
        logger.info("Iniciando treinamento...")
        kfold_result = trainer.run_kfold(dataset, exp_dir)
        
        # Usar o melhor fold para avaliação final
        best_fold_idx = kfold_result["best_fold_idx"]
        train_losses = kfold_result["best_train_losses"]
        val_losses = kfold_result["best_val_losses"]
        train_r2s = []
        val_r2s = []
        best_epoch = kfold_result["best_fold_results"]["best_epoch"]
        
        # Para avaliação final, usar o split do melhor fold
        from src.training.validation import get_kfold_splits
        fold_splits = get_kfold_splits(dataset, config.n_folds)
        train_indices, val_indices = fold_splits[best_fold_idx]
        
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers = 4,
            pin_memory = True,
            persistent_workers = True,
        )
    else:
        raise ValueError(f"cv_mode inválido: {config.cv_mode}")
    
    # =========================================================================
    # STEP 5: AVALIAÇÃO FINAL COM INCERTEZA
    # =========================================================================
    
    logger.info("Executando avaliação final com MC Dropout...")
    logger.info("Estimando incerteza com {} forward passes", config.evaluation.mc_samples)
    
    y_mean_scaled, y_std_scaled, y_true_scaled = trainer.predict_with_uncertainty(val_loader)
    
    # Converter para valores físicos (desfazer normalização)
    y_mean = dataset.inverse_transform_targets(y_mean_scaled)
    y_true = dataset.inverse_transform_targets(y_true_scaled)
    
    # Escalar desvio padrão
    if hasattr(dataset.target_scaler, "std") and dataset.target_scaler.std is not None:
        scale_factor = dataset.target_scaler.std.item() + 1e-8
    elif hasattr(dataset.target_scaler, "max") and dataset.target_scaler.max is not None:
        scale_factor = (dataset.target_scaler.max.item() - dataset.target_scaler.min.item()) + 1e-8
    else:
        scale_factor = 1.0
    
    y_std = y_std_scaled * scale_factor
    
    # Calcular métricas finais
    metrics = compute_metrics(y_true, y_mean)
    
    logger.info("Métricas finais (valores físicos):")
    logger.info("  MSE:  {:.6e}", metrics["mse"])
    logger.info("  RMSE: {:.6e}", metrics["rmse"])
    logger.info("  MAE:  {:.6e}", metrics["mae"])
    logger.info("  R²:   {:.6f}", metrics["r2"])
    
    # =========================================================================
    # STEP 6: PREPARAR DADOS PARA GRÁFICOS
    # =========================================================================

    # Com grafos de tamanhos variados, precisamos reconstruir por grafo
# y_mean, y_true, y_std são tensores concatenados de todos os nós de todos os grafos

    n_val_samples = len(val_indices)
    
    # Reconstruir predições por grafo
    y_true_per_graph = []
    y_mean_per_graph = []
    y_std_per_graph = []
    positions_per_graph = []
    
    offset = 0
    for i, idx in enumerate(val_indices):
        graph = dataset.data_list[idx]
        n_nodes = graph.x.shape[0]
        
        y_true_per_graph.append(y_true[offset:offset + n_nodes].numpy())
        y_mean_per_graph.append(y_mean[offset:offset + n_nodes].numpy())
        y_std_per_graph.append(y_std[offset:offset + n_nodes].numpy())
        
        # Extrair posições x deste grafo (coluna x_idx das features originais)
        positions_per_graph.append(graph.x[:, x_idx].numpy())
        
        offset += n_nodes
    
    # Calcular erro por amostra
    errors_per_sample = np.array([
        ((y_true_per_graph[i] - y_mean_per_graph[i]) ** 2).mean()
        for i in range(n_val_samples)
    ])
    
    best_idx = errors_per_sample.argmin()
    worst_idx = errors_per_sample.argmax()
    median_idx = np.argsort(errors_per_sample)[len(errors_per_sample) // 2]
    
    best_sample_id = val_indices[best_idx]
    worst_sample_id = val_indices[worst_idx]
    median_sample_id = val_indices[median_idx]
    
    logger.info("Casos selecionados para visualização:")
    logger.info("  Melhor (idx local: {}, ID global: {}, n_nodes: {}): MSE = {:.6e}", 
                best_idx, best_sample_id, len(y_true_per_graph[best_idx]), errors_per_sample[best_idx])
    logger.info("  Mediano (idx local: {}, ID global: {}, n_nodes: {}): MSE = {:.6e}", 
                median_idx, median_sample_id, len(y_true_per_graph[median_idx]), errors_per_sample[median_idx])
    logger.info("  Pior (idx local: {}, ID global: {}, n_nodes: {}): MSE = {:.6e}", 
                worst_idx, worst_sample_id, len(y_true_per_graph[worst_idx]), errors_per_sample[worst_idx])
    
    # =========================================================================
    # STEP 7: GERAR GRÁFICOS
    # =========================================================================
    
    logger.info("Gerando gráficos de avaliação...")
    
    set_style(context="paper", font_family="sans-serif")
    
    # Gráfico de loss
    fig = plot_loss_curves(
        train_losses,
        val_losses,
        log_scale=True,
        xlim=(0, config.training.epochs),
    )
    save_figure(fig, figures_dir / "loss_curves", formats=["png"])
    logger.info("Salvo: loss_curves.png")
    
    # Gráfico de R² (apenas para fixed split)
    if config.cv_mode == "fixed" and train_r2s and val_r2s:
        fig = plot_r2_curves(
            train_r2s,
            val_r2s,
            title="Curvas de R²",
            xlim=(0, config.training.epochs),
            train_color="#037A68",
            val_color="#E39774",
        )
        save_figure(fig, figures_dir / "r2_curves", formats=["png"])
        logger.info("Salvo: r2_curves.png")
    
    # Gráfico comparativo de vigas
    cases_data = [
        {
            "positions": positions_per_graph[best_idx],
            "y_vem": y_true_per_graph[best_idx],
            "y_nn": y_mean_per_graph[best_idx],
            "y_nn_std": y_std_per_graph[best_idx],
            "title": f"Melhor Caso ({len(y_true_per_graph[best_idx])} nós)",
            "error": errors_per_sample[best_idx],
        },
        {
            "positions": positions_per_graph[median_idx],
            "y_vem": y_true_per_graph[median_idx],
            "y_nn": y_mean_per_graph[median_idx],
            "y_nn_std": y_std_per_graph[median_idx],
            "title": f"Caso Mediano ({len(y_true_per_graph[median_idx])} nós)",
            "error": errors_per_sample[median_idx],
        },
        {
            "positions": positions_per_graph[worst_idx],
            "y_vem": y_true_per_graph[worst_idx],
            "y_nn": y_mean_per_graph[worst_idx],
            "y_nn_std": y_std_per_graph[worst_idx],
            "title": f"Pior Caso ({len(y_true_per_graph[worst_idx])} nós)",
            "error": errors_per_sample[worst_idx],
        },
    ]
    
    fig = plot_beam_cases_comparison_variable(
        cases_data,
        figsize=(15, 5),
        scale_y=1000.0,
        undeformed_color="black",
        vem_color="#037A68",
        nn_color="#326273",
        uncertainty_color="red",
        uncertainty_alpha=0.25,
        n_sigma=2.0,
    )

    save_figure(fig, figures_dir / "beam_comparison", formats=["png"])
    logger.info("Salvo: beam_comparison.png")
    
    
# =========================================================================
# STEP 8: SALVAR ARTEFATOS FINAIS E LOGAR RESULTADOS
# =========================================================================

    logger.info("Salvando artefatos finais...")

    # Salvar scalers
    torch.save(
        {
            "feature_scaler": dataset.feature_scaler.state_dict(),
            "target_scaler": dataset.target_scaler.state_dict(),
            "scaler_type": config.data.scaler_type,
        },
        exp_dir / "scalers.pt",
    )
    logger.info("Scalers salvos em: scalers.pt")
    
    # Finalizar W&B com métricas finais
    wandb.log({
        "final_val_mse": metrics["mse"],
        "final_rmse": metrics["rmse"],
        "final_mae": metrics["mae"],
        "final_r2": metrics["r2"],
        "best_epoch": best_epoch,
    })
    wandb.finish()
    logger.info("W&B finalizado")

    # Logging estruturado dos resultados finais (para arquivo de log)
    logger.info("=" * 70)
    logger.info("RESULTADOS FINAIS DO EXPERIMENTO")
    logger.info("=" * 70)

    logger.info("MÉTRICAS DE VALIDAÇÃO:")
    logger.info("  MSE:  {:.6e}", metrics["mse"])
    logger.info("  RMSE: {:.6e}", metrics["rmse"])
    logger.info("  MAE:  {:.6e}", metrics["mae"])
    logger.info("  R²:   {:.6f}", metrics["r2"])

    logger.info("CONFIGURAÇÃO DE TREINAMENTO USADA:")
    logger.info("  Dataset: {} amostras", config.data.dataset_size)
    logger.info("  Modelo: {} hidden dim, {} layers", config.model.hidden_dim, config.model.num_layers)
    logger.info("  Treinamento: {} epochs (melhor em época {})", config.training.epochs, best_epoch)
    logger.info("  Learning rate: {:.2e}", config.training.learning_rate)

    logger.info("VALIDAÇÃO:")
    logger.info("  Modo: {}", config.cv_mode)
    logger.info("  Amostras: {}", len(val_indices))
    if config.cv_mode == "kfold":
        logger.info("  Folds: {}", config.n_folds)

    logger.info("=" * 70)

    # Salvar arquivo de métricas em texto (para consulta rápida)
    with open(exp_dir / "metrics.txt", "w") as f:
        f.write("MÉTRICAS DE AVALIAÇÃO FINAL\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {config.data.dataset_size} amostras\n")
        f.write(f"Validação: {len(val_indices)} amostras\n")
        f.write(f"Modo: {config.cv_mode}\n")
        if config.cv_mode == "kfold":
            f.write(f"Número de folds: {config.n_folds}\n")
        f.write(f"Melhor época: {best_epoch}\n\n")
        f.write("Métricas (valores físicos):\n")
        f.write(f"  MSE:  {metrics['mse']:.6e}\n")
        f.write(f"  RMSE: {metrics['rmse']:.6e}\n")
        f.write(f"  MAE:  {metrics['mae']:.6e}\n")
        f.write(f"  R²:   {metrics['r2']:.6f}\n")
    logger.info("Métricas salvas em: metrics.txt")

    # Salvar histórico de treinamento
    torch.save(
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_r2s": train_r2s,
            "val_r2s": val_r2s,
        },
        exp_dir / "training_history.pt",
    )
    logger.info("Histórico de treinamento salvo em: training_history.pt")
    
    # =========================================================================
    # FINALIZAÇÃO
    # =========================================================================
    
    logger.info("=" * 70)
    logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    logger.info("=" * 70)
    logger.info("Diretório: {}", exp_dir)
    logger.info("Gráficos: {}", figures_dir)


if __name__ == "__main__":
    main()