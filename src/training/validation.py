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

def split_dataset_stratified(
    dataset,
    val_split: float,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Divide índices do dataset em treino e validação de forma estratificada por n_elements.
    
    Esta função garante que a proporção val_split seja respeitada dentro de cada
    grupo de n_elements. Isso resulta em treino e validação com a mesma distribuição
    de tamanhos de grafos, reduzindo oscilação na val_loss.
    
    Por exemplo, com val_split=0.30:
    - Grafos com n_elements=5: 70% vão para treino, 30% para validação
    - Grafos com n_elements=50: 70% vão para treino, 30% para validação
    - E assim por diante para cada valor único de n_elements
    
    Args:
        dataset: Dataset BeamGraphDataset completo. Cada elemento deve ter
            o atributo n_elements acessível.
        val_split: Fração dos dados reservada para validação (0.0 a 1.0).
        seed: Seed para gerador aleatório, para reproducibilidade.
        
    Returns:
        Tupla (train_indices, val_indices) onde cada elemento é uma lista
        de inteiros representando os índices das amostras em cada conjunto.
    """
    # Agrupar índices por n_elements
    from collections import defaultdict
    
    indices_by_n_elements: dict[int, list[int]] = defaultdict(list)
    
    for idx in range(len(dataset)):
        n_elements = dataset.data_list_scaled[idx].n_elements
        indices_by_n_elements[n_elements].append(idx)
    
    # Fazer split dentro de cada grupo
    train_indices = []
    val_indices = []
    
    generator = torch.Generator().manual_seed(seed)
    
    for n_elements in sorted(indices_by_n_elements.keys()):
        group_indices = indices_by_n_elements[n_elements]
        n_group = len(group_indices)
        
        # Embaralhar índices deste grupo
        perm = torch.randperm(n_group, generator=generator).tolist()
        shuffled_indices = [group_indices[i] for i in perm]
        
        # Calcular ponto de corte
        n_val = int(n_group * val_split)
        n_train = n_group - n_val
        
        # Dividir
        train_indices.extend(shuffled_indices[:n_train])
        val_indices.extend(shuffled_indices[n_train:])
    
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