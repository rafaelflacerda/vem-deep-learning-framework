"""
Script de treinamento da GNN para predição de deslocamentos em vigas 1D.

Uso:
    python scripts/train_gnn.py

O script:
1. Carrega dados preprocessados (.pt)
2. Faz split train/validation
3. Treina a GNN com MC Dropout
4. Avalia no conjunto de validação
5. Gera gráficos de avaliação
6. Salva modelo e scalers

Configurações podem ser alteradas na seção CONFIGURAÇÕES DO EXPERIMENTO.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

# Adiciona src ao path para imports funcionarem
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import BeamGraphDataset
from src.modeling.gnn import BeamGNN
from src.paths import create_experiment_dir, ensure_dir, paths
from src.utils.logger import configure_logger
from src.utils.visualization import (
    plot_beam_cases_comparison,
    plot_loss_curves,
    plot_r2_curves,
    save_figure,
    set_style,
)

# =============================================================================
# CONFIGURAÇÕES DO EXPERIMENTO
# =============================================================================

CONFIG = {
    # Dados
    "sampling_method": "Sobol",  # Ou 'LHS'
    "dataset_size": 100000,  # Qual dataset usar (10, 100, 1000, 10000, etc.)
    "val_split": 0.30,  # Fração dos dados para validação
    "scaler_type": "minmax",  # 'standard', 'minmax', ou 'none'
    "cv_mode": "fixed",  # 'fixed' para split fixo, 'kfold' para k-fold cross-validation
    "n_folds": 5,  # Número de folds,
    "rho_folder": "rho_0.010",
    # Modelo
    "input_dim": 12,  # Número de features (não alterar a menos que mude o preprocessing)
    "hidden_dim": 64,  # Dimensão do espaço latente
    "output_dim": 1,  # Dimensão da saída (1 = deslocamento vertical)
    "num_layers": 6,  # Número de camadas de message passing
    "dropout": 0.1,  # Taxa de dropout
    # Treinamento
    "epochs": 500,
    "batch_size": 64,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,  # Regularização L2
    "loss_type": "mse",  # 'huber' ou 'mse'
    "huber_delta": 1.0,
    # MC Dropout para incerteza
    "mc_samples": 50,  # Número de forward passes para estimar incerteza
    # Gráficos
    "n_profile_samples": 6,  # Quantos perfis individuais plotar
}


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================


def get_device() -> torch.device:
    """
    Retorna o device MPS.

    Raises:
        RuntimeError: Se MPS não estiver disponível.
    """
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS não está disponível neste sistema. "
            "Este script requer um Mac com Apple Silicon e PyTorch com suporte a MPS."
        )

    if not torch.backends.mps.is_built():
        raise RuntimeError(
            "PyTorch não foi compilado com suporte a MPS. "
            "Reinstale o PyTorch com suporte a MPS."
        )

    return torch.device("mps")


def split_dataset(
    dataset: BeamGraphDataset,
    val_split: float,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Divide índices do dataset em treino e validação.

    Args:
        dataset: Dataset completo.
        val_split: Fração para validação (0 a 1).
        seed: Seed para reprodutibilidade.

    Returns:
        Tupla (train_indices, val_indices).
    """
    n_samples = len(dataset)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_samples, generator=generator).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    return train_indices, val_indices


def get_kfold_splits(
    dataset: BeamGraphDataset,
    n_folds: int,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """
    Gera splits para k-fold cross-validation.

    Args:
        dataset: Dataset completo.
        n_folds: Número de folds.
        seed: Seed para reprodutibilidade.

    Returns:
        Lista de tuplas (train_indices, val_indices) para cada fold.
    """
    n_samples = len(dataset)
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    splits = []
    for train_idx, val_idx in kfold.split(range(n_samples)):
        splits.append((train_idx.tolist(), val_idx.tolist()))

    return splits


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Treina o modelo por uma época.

    Returns:
        Loss média da época.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index)
        loss = criterion(out.squeeze(), batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """
    Avalia o modelo no conjunto de validação.

    Returns:
        Tupla (loss_média, todas_predições, todos_targets).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)

        out = model(batch.x, batch.edge_index)
        loss = criterion(out.squeeze(), batch.y)

        total_loss += loss.item()
        n_batches += 1

        all_preds.append(out.squeeze().cpu())
        all_targets.append(batch.y.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return total_loss / n_batches, all_preds, all_targets


def predict_with_uncertainty_batched(
    model: BeamGNN,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Faz predição com incerteza para todo o DataLoader.

    Returns:
        Tupla (means, stds, targets) concatenados.
    """
    model.train()  # Ativa dropout

    all_means = []
    all_stds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Múltiplos forward passes
            predictions = []
            for _ in range(n_samples):
                out = model(batch.x, batch.edge_index)
                predictions.append(out.squeeze())

            predictions = torch.stack(predictions)  # (n_samples, n_nodes_batch)

            mean = predictions.mean(dim=0)
            std = predictions.std(dim=0)

            all_means.append(mean.cpu())
            all_stds.append(std.cpu())
            all_targets.append(batch.y.cpu())

    return torch.cat(all_means), torch.cat(all_stds), torch.cat(all_targets)


def compute_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> dict[str, float]:
    """
    Calcula métricas de avaliação.

    Returns:
        Dicionário com MSE, RMSE, MAE, R².
    """
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    mse = float(((y_true - y_pred) ** 2).mean())
    rmse = float(mse**0.5)
    mae = float(abs(y_true - y_pred).mean())

    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def compute_r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calcula R² entre predições e valores reais."""
    y_true_np = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.numpy() if isinstance(y_pred, torch.Tensor) else y_pred

    ss_res = ((y_true_np - y_pred_np) ** 2).sum()
    ss_tot = ((y_true_np - y_true_np.mean()) ** 2).sum()

    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def train_fold(
    model: BeamGNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    exp_dir: Path,
    fold_name: str = "",
) -> tuple[float, int, list[float], list[float], BeamGNN]:
    """
    Treina o modelo para um fold.

    Returns:
        Tupla (best_val_loss, best_epoch, train_losses, val_losses, best_model)
    """
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # Treinar
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        # Avaliar
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Salvar melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            model_path = exp_dir / f"best_model{fold_name}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": CONFIG,
                },
                model_path,
            )

        # Log a cada 10 épocas ou na última
        if epoch % 10 == 0 or epoch == epochs:
            fold_str = f" [{fold_name}]" if fold_name else ""
            logger.info(
                "Época {}/{}{} | Train Loss: {:.6f} | Val Loss: {:.6f}",
                epoch,
                epochs,
                fold_str,
                train_loss,
                val_loss,
            )

    # Carregar melhor modelo
    model_path = exp_dir / f"best_model{fold_name}.pt"
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    return best_val_loss, best_epoch, train_losses, val_losses, model


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================


def main():
    # =========================================================================
    # SETUP INICIAL
    # =========================================================================

    # Criar diretório do experimento
    exp_dir = create_experiment_dir("gnn_beam")
    figures_dir = ensure_dir(exp_dir / "figures")

    # Configurar logger
    configure_logger(
        category="training",
        experiment_name=f"gnn_beam_{CONFIG['dataset_size']}",
        log_dir=exp_dir,
    )

    logger.info("Iniciando treinamento da GNN")
    logger.info("Diretório do experimento: {}", exp_dir)

    # Log das configurações
    logger.info("Configurações:")
    for key, value in CONFIG.items():
        logger.info("  {}: {}", key, value)

    # Configurar device
    device = get_device()
    logger.info("Device: {}", device)

    # =========================================================================
    # CARREGAR DADOS
    # =========================================================================

    logger.info("Carregando dataset...")

    dataset_path = (
        paths.data.processed
        / CONFIG["sampling_method"]
        / CONFIG["rho_folder"]  # ex: "rho_0.050"
        / f"dataset_{CONFIG['dataset_size']}.pt"
    )

    if not dataset_path.exists():
        logger.error("Dataset não encontrado: {}", dataset_path)
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")

    dataset = BeamGraphDataset(
        pt_path=str(dataset_path),
        feature_scaler=CONFIG["scaler_type"],
        target_scaler=CONFIG["scaler_type"],
        fit_scalers=True,
    )

    logger.info(
        "Dataset carregado: {} amostras, {} nós, {} features",
        dataset.n_samples,
        dataset.n_nodes,
        dataset.n_features,
    )

    # Extrair posições dos nós para gráficos
    sample_data = torch.load(dataset_path, weights_only=False)
    feature_names = sample_data["metadata"]["feature_names"]["all"]
    x_idx = feature_names.index("x")
    positions = sample_data["features"][0, :, x_idx].numpy()

    # =========================================================================
    # ESCOLHER MODO DE VALIDAÇÃO: FIXED OU K-FOLD
    # =========================================================================

    if CONFIG["cv_mode"] == "fixed":
        logger.info("Usando validação com split fixo")

        # Split train/val
        train_indices, val_indices = split_dataset(dataset, CONFIG["val_split"])
        logger.info(
            "Split: {} treino, {} validação", len(train_indices), len(val_indices)
        )

        # Criar subsets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
        )

        # Criar modelo
        logger.info("Criando modelo...")
        model = BeamGNN(
            input_dim=CONFIG["input_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            output_dim=CONFIG["output_dim"],
            num_layers=CONFIG["num_layers"],
            dropout=CONFIG["dropout"],
        )
        model = model.to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Modelo criado: {} parâmetros treináveis", n_params)

        # Optimizer, scheduler e loss
        optimizer = AdamW(
            model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.75,
            patience=20,
        )

        if CONFIG.get("loss_type", "mse") == "huber":
            criterion = nn.HuberLoss(delta=CONFIG.get("huber_delta", 1.0))
        else:
            criterion = nn.MSELoss()

        # Treinamento
        logger.info("Iniciando treinamento...")

        train_losses = []
        val_losses = []
        train_r2s = []
        val_r2s = []
        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(1, CONFIG["epochs"] + 1):
            # Treinar
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            train_losses.append(train_loss)

            # Avaliar no treino
            _, train_preds, train_targets = evaluate(
                model, train_loader, criterion, device
            )
            train_r2 = compute_r2(train_targets, train_preds)
            train_r2s.append(train_r2)

            # Avaliar na validação
            val_loss, val_preds, val_targets = evaluate(
                model, val_loader, criterion, device
            )
            val_losses.append(val_loss)
            val_r2 = compute_r2(val_targets, val_preds)
            val_r2s.append(val_r2)

            scheduler.step(val_loss)

            # Salvar melhor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_r2": train_r2,
                        "val_r2": val_r2,
                        "config": CONFIG,
                    },
                    exp_dir / "best_model.pt",
                )

            if epoch % 10 == 0 or epoch == CONFIG["epochs"]:
                logger.info(
                    "Época {}/{} | Train Loss: {:.6f} | Val Loss: {:.6f} | Train R²: {:.4f} | Val R²: {:.4f}",
                    epoch,
                    CONFIG["epochs"],
                    train_loss,
                    val_loss,
                    train_r2,
                    val_r2,
                )

        logger.info("Treinamento concluído!")
        logger.info("Melhor época: {} com Val Loss: {:.6f}", best_epoch, best_val_loss)

        # Carregar melhor modelo
        checkpoint = torch.load(exp_dir / "best_model.pt", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    elif CONFIG["cv_mode"] == "kfold":
        logger.info("Usando validação com {}-fold cross-validation", CONFIG["n_folds"])

        # Gerar splits dos folds
        fold_splits = get_kfold_splits(dataset, CONFIG["n_folds"])

        # Armazenar resultados de todos os folds
        all_fold_results = []
        all_train_losses = []
        all_val_losses = []

        for fold_idx, (train_indices, val_indices) in enumerate(fold_splits, 1):
            logger.info("=" * 70)
            logger.info("Treinando Fold {}/{}", fold_idx, CONFIG["n_folds"])
            logger.info("=" * 70)
            logger.info(
                "Split: {} treino, {} validação", len(train_indices), len(val_indices)
            )

            # Criar subsets para este fold
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)

            # DataLoaders para este fold
            train_loader = DataLoader(
                train_dataset,
                batch_size=CONFIG["batch_size"],
                shuffle=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=CONFIG["batch_size"],
                shuffle=False,
            )

            # Criar modelo novo para este fold
            model = BeamGNN(
                input_dim=CONFIG["input_dim"],
                hidden_dim=CONFIG["hidden_dim"],
                output_dim=CONFIG["output_dim"],
                num_layers=CONFIG["num_layers"],
                dropout=CONFIG["dropout"],
            )
            model = model.to(device)

            # Optimizer, scheduler e loss para este fold
            optimizer = AdamW(
                model.parameters(),
                lr=CONFIG["learning_rate"],
                weight_decay=CONFIG["weight_decay"],
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.75,
                patience=20,
            )

            if CONFIG.get("loss_type", "mse") == "huber":
                criterion = nn.HuberLoss(delta=CONFIG.get("huber_delta", 1.0))
            else:
                criterion = nn.MSELoss()

            # Treinar este fold
            best_val_loss, best_epoch, train_losses, val_losses, best_model = (
                train_fold(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    device=device,
                    epochs=CONFIG["epochs"],
                    exp_dir=exp_dir,
                    fold_name=f"_fold{fold_idx}",
                )
            )

            # Armazenar resultados deste fold
            all_fold_results.append(
                {
                    "fold": fold_idx,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                }
            )
            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)

            logger.info(
                "Fold {} concluído | Melhor Val Loss: {:.6f} na época {}",
                fold_idx,
                best_val_loss,
                best_epoch,
            )

        # Calcular médias entre folds
        mean_val_loss = np.mean([r["best_val_loss"] for r in all_fold_results])
        std_val_loss = np.std([r["best_val_loss"] for r in all_fold_results])

        logger.info("=" * 70)
        logger.info("RESULTADOS DO K-FOLD CROSS-VALIDATION")
        logger.info("=" * 70)
        logger.info("Val Loss Médio: {:.6f} ± {:.6f}", mean_val_loss, std_val_loss)

        for result in all_fold_results:
            logger.info(
                "  Fold {}: {:.6f} (época {})",
                result["fold"],
                result["best_val_loss"],
                result["best_epoch"],
            )

        # Para avaliação final, usar o melhor fold
        best_fold_idx = np.argmin([r["best_val_loss"] for r in all_fold_results])
        logger.info("Usando Fold {} para avaliação final", best_fold_idx + 1)

        # Carregar melhor modelo do melhor fold
        checkpoint = torch.load(
            exp_dir / f"best_model_fold{best_fold_idx + 1}.pt", weights_only=False
        )
        model = BeamGNN(
            input_dim=CONFIG["input_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            output_dim=CONFIG["output_dim"],
            num_layers=CONFIG["num_layers"],
            dropout=CONFIG["dropout"],
        )
        model = model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Usar o split do melhor fold para avaliação
        train_indices, val_indices = fold_splits[best_fold_idx]
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
        )

        # Para os gráficos, usar as losses do melhor fold
        train_losses = all_train_losses[best_fold_idx]
        val_losses = all_val_losses[best_fold_idx]
        train_r2s = []
        val_r2s = []
        best_val_loss = all_fold_results[best_fold_idx]["best_val_loss"]
        best_epoch = all_fold_results[best_fold_idx]["best_epoch"]

    else:
        raise ValueError(
            f"cv_mode inválido: {CONFIG['cv_mode']}. Use 'fixed' ou 'kfold'."
        )

    # =========================================================================
    # AVALIAÇÃO FINAL (COMUM PARA AMBOS OS MODOS)
    # =========================================================================

    logger.info("Carregando melhor modelo para avaliação final...")

    # Avaliação com incerteza
    logger.info(
        "Calculando predições com incerteza via MC Dropout ({} samples)...",
        CONFIG["mc_samples"],
    )

    y_mean_scaled, y_std_scaled, y_true_scaled = predict_with_uncertainty_batched(
        model, val_loader, device, CONFIG["mc_samples"]
    )

    # Converter para valores físicos
    y_mean = dataset.inverse_transform_targets(y_mean_scaled)
    y_true = dataset.inverse_transform_targets(y_true_scaled)

    # Escalar std
    if hasattr(dataset.target_scaler, "std") and dataset.target_scaler.std is not None:
        scale_factor = dataset.target_scaler.std.item() + 1e-8
    elif (
        hasattr(dataset.target_scaler, "max") and dataset.target_scaler.max is not None
    ):
        scale_factor = (
            dataset.target_scaler.max.item() - dataset.target_scaler.min.item()
        ) + 1e-8
    else:
        scale_factor = 1.0
    y_std = y_std_scaled * scale_factor

    # Métricas finais
    metrics = compute_metrics(y_true, y_mean)

    logger.info("Métricas finais (valores físicos):")
    logger.info("  MSE:  {:.6e}", metrics["mse"])
    logger.info("  RMSE: {:.6e}", metrics["rmse"])
    logger.info("  MAE:  {:.6e}", metrics["mae"])
    logger.info("  R²:   {:.6f}", metrics["r2"])

    # =========================================================================
    # PREPARAR DADOS PARA GRÁFICOS DA VIGA
    # =========================================================================

    n_val_samples = len(val_indices)
    n_nodes = dataset.n_nodes

    y_true_reshaped = y_true.reshape(n_val_samples, n_nodes)
    y_mean_reshaped = y_mean.reshape(n_val_samples, n_nodes)
    y_std_reshaped = y_std.reshape(n_val_samples, n_nodes)

    errors_per_sample = ((y_true_reshaped - y_mean_reshaped) ** 2).mean(axis=1)

    best_idx = errors_per_sample.argmin()
    worst_idx = errors_per_sample.argmax()
    median_idx = np.argsort(errors_per_sample)[len(errors_per_sample) // 2]

    logger.info("Casos selecionados para visualização:")
    logger.info(
        "  Melhor caso (idx {}): MSE = {:.6e}", best_idx, errors_per_sample[best_idx]
    )
    logger.info(
        "  Caso mediano (idx {}): MSE = {:.6e}",
        median_idx,
        errors_per_sample[median_idx],
    )
    logger.info(
        "  Pior caso (idx {}): MSE = {:.6e}", worst_idx, errors_per_sample[worst_idx]
    )

    # =========================================================================
    # GERAR GRÁFICOS
    # =========================================================================

    logger.info("Gerando gráficos de avaliação...")

    set_style(context="paper", font_family="sans-serif")

    # Loss Curves
    fig = plot_loss_curves(
        train_losses,
        val_losses,
        log_scale=True,
        xlim=(0, CONFIG["epochs"]),
    )
    save_figure(fig, figures_dir / "loss_curves", formats=["png"])
    logger.info("Salvo: loss_curves.png")

    # R² Curves (apenas para modo fixed)
    if CONFIG["cv_mode"] == "fixed" and train_r2s and val_r2s:
        fig = plot_r2_curves(
            train_r2s,
            val_r2s,
            title="Curvas de R²",
            xlim=(0, CONFIG["epochs"]),
            train_color="#037A68",
            val_color="#E39774",
        )
        save_figure(fig, figures_dir / "r2_curves", formats=["png"])
        logger.info("Salvo: r2_curves.png")

    # Gráfico comparativo da viga
    cases_data = [
        {
            "y_vem": y_true_reshaped[best_idx],
            "y_nn": y_mean_reshaped[best_idx],
            "y_nn_std": y_std_reshaped[best_idx],
            "title": "Melhor Caso",
            "error": errors_per_sample[best_idx],
        },
        {
            "y_vem": y_true_reshaped[median_idx],
            "y_nn": y_mean_reshaped[median_idx],
            "y_nn_std": y_std_reshaped[median_idx],
            "title": "Caso Mediano",
            "error": errors_per_sample[median_idx],
        },
        {
            "y_vem": y_true_reshaped[worst_idx],
            "y_nn": y_mean_reshaped[worst_idx],
            "y_nn_std": y_std_reshaped[worst_idx],
            "title": "Pior Caso",
            "error": errors_per_sample[worst_idx],
        },
    ]

    fig = plot_beam_cases_comparison(
        positions,
        cases_data,
        figsize=(15, 5),
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
    # SALVAR ARTEFATOS FINAIS
    # =========================================================================

    torch.save(
        {
            "feature_scaler": dataset.feature_scaler.state_dict(),
            "target_scaler": dataset.target_scaler.state_dict(),
            "scaler_type": CONFIG["scaler_type"],
        },
        exp_dir / "scalers.pt",
    )
    logger.info("Scalers salvos em: {}", exp_dir / "scalers.pt")

    with open(exp_dir / "metrics.txt", "w") as f:
        f.write("Métricas de Avaliação\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Dataset: {CONFIG['dataset_size']} amostras\n")
        f.write(f"Validação: {len(val_indices)} amostras\n")
        f.write(f"Modo: {CONFIG['cv_mode']}\n")
        if CONFIG["cv_mode"] == "kfold":
            f.write(f"Número de folds: {CONFIG['n_folds']}\n")
        f.write(f"Melhor época: {best_epoch}\n\n")
        f.write("Métricas (valores físicos):\n")
        f.write(f"  MSE:  {metrics['mse']:.6e}\n")
        f.write(f"  RMSE: {metrics['rmse']:.6e}\n")
        f.write(f"  MAE:  {metrics['mae']:.6e}\n")
        f.write(f"  R²:   {metrics['r2']:.6f}\n")
    logger.info("Métricas salvas em: {}", exp_dir / "metrics.txt")

    torch.save(
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_r2s": train_r2s,
            "val_r2s": val_r2s,
        },
        exp_dir / "training_history.pt",
    )
    logger.info(
        "Histórico de treinamento salvo em: {}", exp_dir / "training_history.pt"
    )

    logger.info("=" * 70)
    logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO")
    logger.info("=" * 70)
    logger.info("Diretório do experimento: {}", exp_dir)
    logger.info("Gráficos salvos em: {}", figures_dir)


if __name__ == "__main__":
    main()
