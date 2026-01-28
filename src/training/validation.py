"""
Funções para estratégias de divisão de dados para validação.

Este módulo contém funções para realizar diferentes tipos de split do dataset,
como split fixo treino/validação e k-fold cross-validation.
"""

import torch
from sklearn.model_selection import KFold

from src.data.dataset import BeamGraphDataset


def split_dataset(
    dataset: BeamGraphDataset,
    val_split: float,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Divide índices do dataset em treino e validação com um split fixo.

    Esta função realiza uma divisão simples, determinística do dataset em
    dois subconjuntos: treino e validação. A divisão é controlada por um
    seed para reproducibilidade. Use esta função quando você quer um
    experimento rápido ou quando a quantidade de dados é grande o suficiente
    que k-fold não é necessário.

    Args:
        dataset: Dataset BeamGraphDataset completo.
        val_split: Fração dos dados reservada para validação. Deve estar
            entre 0.0 e 1.0. Por exemplo, 0.30 significa 30% para validação
            e 70% para treinamento.
        seed: Seed para gerador aleatório, para reproducibilidade.
            Padrão é 42.

    Returns:
        Tupla (train_indices, val_indices) onde cada elemento é uma lista
        de inteiros representando os índices das amostras em cada conjunto.

    Example:
        >>> dataset = BeamGraphDataset(...)
        >>> train_idx, val_idx = split_dataset(dataset, val_split=0.30)
        >>> train_dataset = Subset(dataset, train_idx)
        >>> val_dataset = Subset(dataset, val_idx)
    """
    n_samples = len(dataset)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    # Gerar permutação aleatória dos índices
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_samples, generator=generator).tolist()

    # Dividir em treino e validação
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    return train_indices, val_indices


def get_kfold_splits(
    dataset: BeamGraphDataset,
    n_folds: int,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """
    Gera múltiplos splits para k-fold cross-validation.

    Esta função implementa k-fold cross-validation, dividindo o dataset em
    k folds não-sobrepostos. Cada fold é usado como conjunto de validação
    uma vez, enquanto os outros k-1 folds são usados para treinamento.
    Use esta função quando você quer uma avaliação mais robusta e tem
    tempo computacional disponível para treinar k modelos diferentes.

    A divisão é determinística (controlada por seed), então chamar esta
    função múltiplas vezes com a mesma seed gera os mesmos splits.

    Args:
        dataset: Dataset BeamGraphDataset completo.
        n_folds: Número de folds. Valor típico é 5 ou 10. Deve ser >= 2.
        seed: Seed para reprodutibilidade. Padrão é 42.

    Returns:
        Lista de tuplas, cada uma contendo (train_indices, val_indices)
        para um fold específico. O comprimento desta lista é n_folds.
        Para cada fold, train_indices contém aproximadamente (k-1)/k das
        amostras, e val_indices contém aproximadamente 1/k das amostras.

    Example:
        >>> dataset = BeamGraphDataset(...)
        >>> splits = get_kfold_splits(dataset, n_folds=5)
        >>> for fold_idx, (train_idx, val_idx) in enumerate(splits):
        ...     train_dataset = Subset(dataset, train_idx)
        ...     val_dataset = Subset(dataset, val_idx)
        ...     # Treinar modelo neste fold
    """
    n_samples = len(dataset)

    # Inicializar KFold com seed para reproducibilidade
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Gerar splits
    splits = []
    for train_idx, val_idx in kfold.split(range(n_samples)):
        splits.append((train_idx.tolist(), val_idx.tolist()))

    return splits
