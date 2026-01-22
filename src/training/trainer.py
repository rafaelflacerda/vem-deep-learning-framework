"""
Classe trainer para coordenar treinamento de GNN para vigas 1D.

Este módulo contém a classe BeamGNNTrainer que encapsula toda a lógica
de treinamento, avaliação e predição com incerteza.
"""

import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from src.modeling.gnn import BeamGNN
from src.config.experiment_config import ExperimentConfig
from src.training.metrics import compute_r2, compute_metrics

import wandb


class BeamGNNTrainer:
    """
    Trainer para modelos GNN em problemas de vigas 1D.
    
    Encapsula a lógica completa de treinamento: setup do modelo, loop de épocas,
    avaliação, e predição com incerteza via MC Dropout.
    
    Attributes:
        config: Configuração do experimento (tipo ExperimentConfig).
        device: Device PyTorch (cpu, cuda, mps).
        model: Instância do modelo BeamGNN.
        optimizer: Otimizador (AdamW).
        scheduler: Scheduler de learning rate (ReduceLROnPlateau).
        criterion: Função de loss (MSE ou Huber).
    """
    
    def __init__(self, config: ExperimentConfig, device: torch.device):
        """
        Inicializa o trainer.
        
        Args:
            config: Configuração do experimento.
            device: Device a usar (mps, cuda, cpu).
        """
        self.config = config
        self.device = device
        
        # Criar modelo
        self.model = BeamGNN(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.model.output_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            activation=config.model.activation, 
        )
        self.model = self.model.to(device)

        if torch.__version__ >= "2.0.0" and self.device.type == "cuda":
            logger.info("Compilando modelo com torch.compile()...")
            try:
                self.model = torch.compile(
                    self.model,
                    mode="default",  # Opções: "default", "reduce-overhead", "max-autotune"
                )
                logger.info("Modelo compilado com sucesso")
            except Exception as e:
                logger.warning("Falha ao compilar modelo: {}. Continuando sem compilação.", e)

        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        if self.use_amp:
            logger.info("Mixed Precision (AMP) ativado para acelerar treinamento em CUDA.")
        
        # Contar parâmetros
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Modelo criado: {} parâmetros treináveis", n_params)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        logger.info("Otimizador criado: {}", config.training.optimizer_type)
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        logger.info("Scheduler criado: {}", config.training.scheduler_type)
        
        # Setup criterion
        if config.training.loss_type == "huber":
            self.criterion = nn.HuberLoss(delta=config.training.huber_delta)
        else:
            self.criterion = nn.MSELoss()
        
        logger.info("Trainer inicializado com device: {}", device)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> float:
        """
        Treina o modelo por uma época.
        
        Percorre todo o dataloader uma vez, calculando loss e fazendo
        backpropagation. Não retorna predições, apenas a loss média.
        
        Args:
            train_loader: DataLoader com dados de treinamento.
            
        Returns:
            Loss média da época.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            batch = batch.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    out = self.model(batch.x, batch.edge_index)
                    loss = self.criterion(out.squeeze(), batch.y)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
            
                out = self.model(batch.x, batch.edge_index)
                loss = self.criterion(out.squeeze(), batch.y)
            
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """
        Avalia o modelo em um conjunto de validação.
        
        Percorre o dataloader sem fazer backpropagation, apenas computando
        loss e coletando predições e targets. Retorna o loss médio e os
        tensores completos de predições e targets para cálculo de métricas.
        
        Args:
            val_loader: DataLoader com dados de validação.
            
        Returns:
            Tupla (loss_média, predições, targets).
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        all_preds = []
        all_targets = []
        
        for batch in val_loader:
            batch = batch.to(self.device, non_blocking=True)
            
            out = self.model(batch.x, batch.edge_index)
            loss = self.criterion(out.squeeze(), batch.y)
            
            total_loss += loss.item()
            n_batches += 1
            
            # Manter na GPU - não fazer .cpu() aqui!
            all_preds.append(out.squeeze())      # ← Fica na GPU
            all_targets.append(batch.y)          # ← Fica na GPU
        
        # Concatenar tudo na GPU primeiro
        all_preds = torch.cat(all_preds)        # ← Tudo ainda na GPU
        all_targets = torch.cat(all_targets)    # ← Tudo ainda na GPU
        
        # AGORA SIM: uma única transferência GPU→CPU no final
        all_preds = all_preds.cpu()
        all_targets = all_targets.cpu()
        
        return total_loss / n_batches, all_preds, all_targets
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        loader: DataLoader,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Faz predições com estimação de incerteza via MC Dropout.
        
        Faz múltiplos forward passes (com dropout ativo) e coleta as predições.
        A média desses passes é a predição final; o desvio padrão é a incerteza.
        
        O número de passes é determinado por config.evaluation.mc_samples.
        
        Args:
            loader: DataLoader com dados para os quais fazer predições.
            
        Returns:
            Tupla (means, stds, targets) onde cada um é um tensor concatenado
            de todas as amostras.
        """
        self.model.train()  # Ativa dropout
        
        all_means = []
        all_stds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device, non_blocking=True)
                
                # Múltiplos forward passes
                predictions = []
                for _ in range(self.config.evaluation.mc_samples):
                    out = self.model(batch.x, batch.edge_index)
                    predictions.append(out.squeeze())
                
                predictions = torch.stack(predictions)  # (n_samples, n_nodes_batch)
                
                mean = predictions.mean(dim=0)
                std = predictions.std(dim=0)
                
                all_means.append(mean.cpu())
                all_stds.append(std.cpu())
                all_targets.append(batch.y.cpu())
        
        return torch.cat(all_means), torch.cat(all_stds), torch.cat(all_targets)
    
    def _train_fold(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        exp_dir: Path,
        fold_name: str = "",
    ) -> tuple[float, int, list[float], list[float]]:
        """
        Treina o modelo para um fold específico (interno ao trainer).
        
        Coordena o loop de épocas para um single fold: treina uma época,
        avalia, salva checkpoint se melhor, e repete.
        
        Args:
            train_loader: DataLoader de treinamento deste fold.
            val_loader: DataLoader de validação deste fold.
            exp_dir: Diretório onde salvar checkpoints.
            fold_name: Nome identificador do fold (ex: "_fold1") para logging.
            
        Returns:
            Tupla (best_val_loss, best_epoch, train_losses, val_losses).
        """
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_epoch = 0
        
        for epoch in range(1, self.config.training.epochs + 1):
            # Treinar
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Avaliar
            val_loss, _, _ = self.evaluate(val_loader)
            val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            # Salvar melhor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                model_path = exp_dir / f"best_model{fold_name}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "config": self.config.model_dump(),
                    },
                    model_path,
                )
            
            # Logar métricas do fold no W&B
            fold_suffix = fold_name if fold_name else ""
            wandb.log({
                f"epoch{fold_suffix}": epoch,
                f"train_loss{fold_suffix}": train_loss,
                f"val_loss{fold_suffix}": val_loss,
                f"val_mse{fold_suffix}": val_loss,
            })
             
            # Log a cada 10 épocas ou na última
            if epoch % 10 == 0 or epoch == self.config.training.epochs:
                fold_str = f" [{fold_name}]" if fold_name else ""
                logger.info(
                    "Época {}/{}{} | Train Loss: {:.6f} | Val Loss: {:.6f}",
                    epoch,
                    self.config.training.epochs,
                    fold_str,
                    train_loss,
                    val_loss,
                )
        
        # Carregar melhor modelo
        model_path = exp_dir / f"best_model{fold_name}.pt"
        checkpoint = torch.load(model_path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        return best_val_loss, best_epoch, train_losses, val_losses
    
    def run_fixed_split(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        exp_dir: Path,
    ) -> dict:
        """
        Executa treinamento com split fixo (não k-fold).
        
        Coordena o loop de épocas para um split simples treino/validação.
        Retorna histórico completo de losses e R² para análise posterior.
        
        Args:
            train_loader: DataLoader de treinamento.
            val_loader: DataLoader de validação.
            exp_dir: Diretório para salvar checkpoints.
            
        Returns:
            Dicionário contendo:
                - 'train_losses': lista de losses de treinamento por época
                - 'val_losses': lista de losses de validação por época
                - 'train_r2s': lista de R² de treinamento por época
                - 'val_r2s': lista de R² de validação por época
                - 'best_epoch': melhor época encontrada
                - 'best_val_loss': melhor validation loss encontrada
        """
        logger.info("Iniciando treinamento com split fixo...")
        
        train_losses = []
        val_losses = []
        train_r2s = []
        val_r2s = []
        best_val_loss = float("inf")
        best_epoch = 0
        
        for epoch in range(1, self.config.training.epochs + 1):
            # Treinar
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Avaliar no treino
            _, train_preds, train_targets = self.evaluate(train_loader)
            train_r2 = compute_r2(train_targets, train_preds)
            train_r2s.append(train_r2)
            
            # Avaliar na validação
            val_loss, val_preds, val_targets = self.evaluate(val_loader)
            val_losses.append(val_loss)
            val_r2 = compute_r2(val_targets, val_preds)
            val_r2s.append(val_r2)
            
            self.scheduler.step(val_loss)
            
            # Salvar melhor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_r2": train_r2,
                        "val_r2": val_r2,
                        "config": self.config.model_dump(),
                    },
                    exp_dir / "best_model.pt",
                )
                
            # Logar métricas no W&B a cada época
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mse": val_loss,
                "train_r2": train_r2,
                "val_r2": val_r2,
            })
            
            if epoch % 10 == 0 or epoch == self.config.training.epochs:
                logger.info(
                    "Época {}/{} | Train Loss: {:.6f} | Val Loss: {:.6f} | Train R²: {:.4f} | Val R²: {:.4f}",
                    epoch,
                    self.config.training.epochs,
                    train_loss,
                    val_loss,
                    train_r2,
                    val_r2,
                )
        
        logger.info("Treinamento concluído!")
        logger.info("Melhor época: {} com Val Loss: {:.6f}", best_epoch, best_val_loss)
        
        # Carregar melhor modelo
        checkpoint = torch.load(exp_dir / "best_model.pt", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_r2s": train_r2s,
            "val_r2s": val_r2s,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }

    
    def run_kfold(
        self,
        dataset,
        exp_dir: Path,
    ) -> dict:
        """
        Executa treinamento com k-fold cross-validation.
        
        Faz k-fold do dataset, treina um modelo para cada fold, e retorna
        resultados agregados. O melhor fold é usado para retornar um modelo
        final e dados de avaliação.
        
        Args:
            dataset: Dataset BeamGraphDataset completo.
            exp_dir: Diretório para salvar checkpoints.
            
        Returns:
            Dicionário contendo resultados de todos os folds e do melhor fold.
        """
        logger.info("Iniciando k-fold cross-validation com {} folds", self.config.n_folds)
        
        # Gerar splits dos folds
        fold_splits = self._get_kfold_splits(dataset)
        
        # Armazenar resultados de todos os folds
        all_fold_results = []
        all_train_losses = []
        all_val_losses = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(fold_splits, 1):
            # ================================================================
            # RESETAR MODELO PARA CADA FOLD
            # Necessário para evitar conflitos de estado quando rodando sweeps
            # com diferentes hiperparâmetros ou para garantir independência
            # entre folds
            # ================================================================
            from src.modeling.gnn import BeamGNN
            
            self.model = BeamGNN(
                input_dim=self.config.model.input_dim,
                hidden_dim=self.config.model.hidden_dim,
                output_dim=self.config.model.output_dim,
                num_layers=self.config.model.num_layers,
                dropout=self.config.model.dropout,
                activation=self.config.model.activation,
            ).to(self.device)

            if torch.__version__ >= "2.0.0" and self.device.type == "cuda":
                logger.info("Compilando modelo com torch.compile()...")
                try:
                    self.model = torch.compile(
                        self.model,
                        mode="default",  # Opções: "default", "reduce-overhead", "max-autotune"
                    )
                    logger.info("Modelo compilado com sucesso")
                except Exception as e:
                    logger.warning("Falha ao compilar modelo: {}. Continuando sem compilação.", e)
            
            # Recriar optimizer e scheduler para o novo modelo
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
            
            # Recriar scaler se usando AMP
            if self.use_amp:
                self.scaler = torch.amp.GradScaler('cuda')
            
            logger.info("Modelo resetado para fold {}", fold_idx)
            # ================================================================
            
            logger.info("=" * 70)
            logger.info("Treinando Fold {}/{}", fold_idx, self.config.n_folds)
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
                batch_size=self.config.training.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
            )
            
            # Treinar este fold
            best_val_loss, best_epoch, train_losses, val_losses = self._train_fold(
                train_loader=train_loader,
                val_loader=val_loader,
                exp_dir=exp_dir,
                fold_name=f"_fold{fold_idx}",
            )
            
            # Armazenar resultados deste fold
            all_fold_results.append({
                "fold": fold_idx,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            })
            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
            
            logger.info(
                "Fold {} concluído | Melhor Val Loss: {:.6f} na época {}",
                fold_idx,
                best_val_loss,
                best_epoch,
            )
        
        # Encontrar melhor fold
        import numpy as np
        best_fold_idx = np.argmin([r["best_val_loss"] for r in all_fold_results])
        
        logger.info("=" * 70)
        logger.info("RESULTADOS DO K-FOLD CROSS-VALIDATION")
        logger.info("=" * 70)
        mean_val_loss = np.mean([r["best_val_loss"] for r in all_fold_results])
        std_val_loss = np.std([r["best_val_loss"] for r in all_fold_results])
        logger.info("Val Loss Médio: {:.6f} ± {:.6f}", mean_val_loss, std_val_loss)
        
        for result in all_fold_results:
            logger.info(
                "  Fold {}: {:.6f} (época {})",
                result["fold"],
                result["best_val_loss"],
                result["best_epoch"],
            )
        
        logger.info("Usando Fold {} para avaliação final", best_fold_idx + 1)
        
        # Carregar melhor modelo do melhor fold
        checkpoint = torch.load(
            exp_dir / f"best_model_fold{best_fold_idx + 1}.pt", weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        return {
            "all_fold_results": all_fold_results,
            "all_train_losses": all_train_losses,
            "all_val_losses": all_val_losses,
            "best_fold_idx": best_fold_idx,
            "best_fold_results": all_fold_results[best_fold_idx],
            "best_train_losses": all_train_losses[best_fold_idx],
            "best_val_losses": all_val_losses[best_fold_idx],
        }
        
    
    
    def _create_optimizer(self):
        """
        Cria o otimizador baseado na configuração.
        
        Suporta: adam, adamw, sgd.
        """
        optimizer_type = self.config.training.optimizer_type.lower()
        lr = self.config.training.learning_rate
        wd = self.config.training.weight_decay
        
        if optimizer_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=wd
                )
        elif optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=wd,
                betas=(self.config.training.adamw_beta1, self.config.training.adamw_beta2),
                eps=self.config.training.adamw_epsilon,
                )
        elif optimizer_type == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=wd,
                momentum=self.config.training.sgd_momentum,
                nesterov=self.config.training.sgd_nesterov
            )
        else:
            raise ValueError(f"Otimizador desconhecido: {optimizer_type}")
    
    def _create_scheduler(self):
        """
        Cria o scheduler de learning rate baseado na configuração.
        
        Suporta: reduce_lr_on_plateau, cosine_annealing, step_lr, linear.
        """
        scheduler_type = self.config.training.scheduler_type.lower()
        
        if scheduler_type == "reduce_lr_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.training.scheduler_factor,
                patience=self.config.training.scheduler_patience,
            )
        elif scheduler_type == "cosine_annealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.scheduler_t_max,
            )
        elif scheduler_type == "step_lr":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.scheduler_step_size,
                gamma=0.1,
            )
        elif scheduler_type == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.config.training.epochs,
            )
        else:
            raise ValueError(f"Scheduler desconhecido: {scheduler_type}")
    
    @staticmethod
    def _split_dataset(
        dataset,
        val_split: float,
        seed: int = 42,
    ) -> tuple[list[int], list[int]]:
        """Divide índices do dataset em treino e validação."""
        n_samples = len(dataset)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val
        
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_samples, generator=generator).tolist()
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        return train_indices, val_indices
    
    @staticmethod
    def _get_kfold_splits(
        dataset,
        n_folds: int = 5,
        seed: int = 42,
    ) -> list[tuple[list[int], list[int]]]:
        """Gera splits para k-fold cross-validation."""
        n_samples = len(dataset)
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        splits = []
        for train_idx, val_idx in kfold.split(range(n_samples)):
            splits.append((train_idx.tolist(), val_idx.tolist()))
        
        return splits