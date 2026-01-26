"""
Prepara datasets Sobol para treinamento da GNN.

Lê CSVs raw (params + results), calcula features e monta arquivos .pt.

Uso:
    python scripts/prepare_sobol_dataset.py --n-samples 100
    python scripts/prepare_sobol_dataset.py --n-samples 1000
    python scripts/prepare_sobol_dataset.py --all  # Processa todos os CSVs disponíveis
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.paths import ensure_dir, paths
from src.utils.logger import configure_logger

rho_max = "rho_0.010_E_fixed_102_elements"

# =============================================================================
# CONSTANTES
# =============================================================================

N_ELEMENTS = 102  # Número fixo de elementos
N_NODES = N_ELEMENTS + 1  # 22 nós


# =============================================================================
# FUNÇÕES DE CÁLCULO DE FEATURES
# =============================================================================


def compute_global_features(params: pd.DataFrame) -> np.ndarray:
    """
    Extrai features globais diretamente do DataFrame de parâmetros.

    Args:
        params: DataFrame com colunas [case_id, L, q, E, I]

    Returns:
        Array shape (n_samples, 4) com [E, I, L, q]
    """
    return params[["I", "L", "q"]].values


def compute_derived_features(params: pd.DataFrame) -> np.ndarray:
    """
    Calcula features derivadas a partir dos parâmetros globais.

    Args:
        params: DataFrame com colunas [case_id, L, q, E, I]

    Returns:
        Array shape (n_samples, 3) com [EI, q_scale, q_over_EI]
    """
    E = params["E"].values
    I = params["I"].values
    L = params["L"].values
    q = params["q"].values

    #EI = E * I
    q_scale = (q * L**4) / (E * I)
    q_over_EI = q / (E * I)

    return np.stack([q_scale, q_over_EI], axis=1)


def compute_positional_features(params: pd.DataFrame) -> np.ndarray:
    """
    Calcula features posicionais para cada nó.

    Para cada caso, calcula x e suas potências normalizadas para os 22 nós.

    Args:
        params: DataFrame com colunas [case_id, L, q, E, I]

    Returns:
        Array shape (n_samples, n_nodes, 5) com [x, x_norm, x_norm², x_norm³, x_norm⁴]
    """
    n_samples = len(params)
    L_values = params["L"].values  # shape: (n_samples,)

    # Índices dos nós (0 a 21)
    node_indices = np.arange(N_NODES)  # shape: (22,)

    # Coordenadas x para todos os casos e nós
    # Broadcasting: (n_samples, 1) * (1, 22) = (n_samples, 22)
    x = (node_indices / N_ELEMENTS) * L_values[:, np.newaxis]

    # Coordenadas normalizadas
    x_norm = node_indices / N_ELEMENTS  # shape: (22,)
    x_norm = np.broadcast_to(x_norm, (n_samples, N_NODES))  # shape: (n_samples, 22)

    # Potências das coordenadas normalizadas
    x_norm_2 = x_norm**2
    x_norm_3 = x_norm**3
    x_norm_4 = x_norm**4

    # Stack: (n_samples, 22, 5)
    return np.stack([x, x_norm, x_norm_2, x_norm_3, x_norm_4], axis=2)


def combine_features(
    global_feats: np.ndarray, derived_feats: np.ndarray, positional_feats: np.ndarray
) -> np.ndarray:
    """
    Combina features globais, derivadas e posicionais em um único tensor.

    Features globais e derivadas são replicadas para cada nó.

    Args:
        global_feats: shape (n_samples, 4)
        derived_feats: shape (n_samples, 3)
        positional_feats: shape (n_samples, n_nodes, 5)

    Returns:
        Array shape (n_samples, n_nodes, 12)
    """
    n_samples = global_feats.shape[0]

    # Replicar features globais e derivadas para cada nó
    # (n_samples, 4) -> (n_samples, 1, 4) -> (n_samples, 22, 4)
    global_feats_rep = np.broadcast_to(
        global_feats[:, np.newaxis, :], (n_samples, N_NODES, 3)
    )

    # (n_samples, 3) -> (n_samples, 1, 3) -> (n_samples, 22, 3)
    derived_feats_rep = np.broadcast_to(
        derived_feats[:, np.newaxis, :], (n_samples, N_NODES, 2)
    )

    # Concatenar: (n_samples, 22, 4+3+5=12)
    return np.concatenate(
        [global_feats_rep, derived_feats_rep, positional_feats], axis=2
    )


# =============================================================================
# FUNÇÕES DE PARSING DOS RESULTADOS
# =============================================================================


def parse_vector_column(series: pd.Series) -> np.ndarray:
    """
    Parseia coluna de vetores do CSV (valores separados por ponto-e-vírgula).

    Args:
        series: Série do pandas com strings "val1;val2;val3;..."

    Returns:
        Array shape (n_samples, n_nodes) com os valores convertidos para float
    """
    n_samples = len(series)
    values = np.zeros((n_samples, N_NODES))

    for i, vec_str in enumerate(series):
        # Remover espaços e separar por ponto-e-vírgula
        vec_str = vec_str.strip()
        parts = vec_str.split(";")

        if len(parts) != N_NODES:
            raise ValueError(
                f"Amostra {i}: esperado {N_NODES} valores, "
                f"encontrado {len(parts)} em '{vec_str[:50]}...'"
            )

        values[i] = [float(p.strip()) for p in parts]

    return values


def extract_targets(
    results: pd.DataFrame, target_type: str = "displacements"
) -> np.ndarray:
    """
    Extrai targets (deslocamentos ou rotações) do DataFrame de resultados.

    Args:
        results: DataFrame com colunas [case_id, n_elements, displacements, rotations]
        target_type: "displacements" ou "rotations"

    Returns:
        Array shape (n_samples, n_nodes)
    """
    if target_type not in results.columns:
        raise ValueError(
            f"Coluna '{target_type}' não encontrada no DataFrame. "
            f"Colunas disponíveis: {list(results.columns)}"
        )

    return parse_vector_column(results[target_type])


# =============================================================================
# FUNÇÕES DE CRIAÇÃO DO GRAFO
# =============================================================================


def create_edge_index() -> np.ndarray:
    """
    Cria edge_index para grafo linear bidirecional com 22 nós.

    Conecta cada nó ao seu vizinho à esquerda e à direita.

    Returns:
        Array shape (2, 42) com formato [source_nodes, target_nodes]
    """
    edges = []

    for i in range(N_NODES - 1):
        # Aresta i -> i+1
        edges.append([i, i + 1])
        # Aresta i+1 -> i (bidirecional)
        edges.append([i + 1, i])

    return np.array(edges, dtype=np.int64).T


# =============================================================================
# FUNÇÃO PRINCIPAL DE PROCESSAMENTO
# =============================================================================


def process_dataset(n_samples: int) -> Path:
    """
    Processa um dataset completo (params + results) e salva arquivo .pt.

    Args:
        n_samples: Número de amostras no dataset

    Returns:
        Path do arquivo .pt criado
    """
    logger.info("Processando dataset com {} amostras", n_samples)

    # Caminhos dos arquivos
    params_path = (
        paths.data.raw
        / "Sobol"
        / "params"
        / rho_max
        / f"params_{n_samples}_samples.csv"
    )
    results_path = (
        paths.data.raw
        / "Sobol"
        / "results"
        / rho_max
        / f"results_{n_samples}_samples.csv"
    )
    output_path = paths.data.processed / "Sobol" / rho_max / f"dataset_{n_samples}.pt"

    # Verificar existência dos arquivos
    if not params_path.exists():
        raise FileNotFoundError(f"Arquivo de parâmetros não encontrado: {params_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Arquivo de resultados não encontrado: {results_path}")

    logger.info("Carregando CSVs...")
    params = pd.read_csv(params_path)
    results = pd.read_csv(results_path)

    # Validar número de amostras
    if len(params) != n_samples:
        raise ValueError(
            f"Params: esperado {n_samples} linhas, encontrado {len(params)}"
        )
    if len(results) != n_samples:
        raise ValueError(
            f"Results: esperado {n_samples} linhas, encontrado {len(results)}"
        )

    # Validar case_ids
    if not np.array_equal(params["case_id"].values, results["case_id"].values):
        raise ValueError("case_ids não correspondem entre params e results")

    logger.info("Calculando features globais...")
    global_feats = compute_global_features(params)

    logger.info("Calculando features derivadas...")
    derived_feats = compute_derived_features(params)

    logger.info("Calculando features posicionais...")
    positional_feats = compute_positional_features(params)

    logger.info("Combinando todas as features...")
    features = combine_features(global_feats, derived_feats, positional_feats)

    logger.info("Extraindo targets (displacements)...")
    targets = extract_targets(results, target_type="displacements")

    logger.info("Criando edge_index...")
    edge_index = create_edge_index()

    # Converter para tensores PyTorch
    features_tensor = torch.from_numpy(features).float()
    targets_tensor = torch.from_numpy(targets).float()
    edge_index_tensor = torch.from_numpy(edge_index).long()

    # Montar dicionário com metadados
    data_dict = {
        "features": features_tensor,
        "targets": targets_tensor,
        "edge_index": edge_index_tensor,
        "metadata": {
            "n_samples": n_samples,
            "n_nodes": N_NODES,
            "n_features": 10,
            "feature_names": {
                "global": ["I", "L", "q"],
                "derived": ["q_scale", "q_over_EI"],
                "positional": [
                    "x",
                    "x_normalized",
                    "x_normalized_2",
                    "x_normalized_3",
                    "x_normalized_4",
                ],
                "all": [
                    "I",
                    "L",
                    "q",  # globais
                    "q_scale",
                    "q_over_EI",  # derivadas
                    "x",
                    "x_normalized",
                    "x_normalized_2",
                    "x_normalized_3",
                    "x_normalized_4",  # posicionais
                ],
            },
            "target_names": ["displacements"],  # Facilita adicionar mais targets depois
        },
    }

    # Salvar arquivo .pt
    ensure_dir(output_path.parent)
    torch.save(data_dict, output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Dataset salvo: {} ({:.2f} MB)", output_path, file_size_mb)

    return output_path


# =============================================================================
# CLI
# =============================================================================


def find_available_datasets() -> list[int]:
    """
    Busca todos os datasets disponíveis no diretório raw/Sobol/params.

    Returns:
        Lista de tamanhos de datasets encontrados (ordenada)
    """
    params_dir = paths.data.raw / "Sobol" / "params" / rho_max

    if not params_dir.exists():
        return []

    sizes = []
    for csv_file in params_dir.glob("params_*_samples.csv"):
        # Extrair número do nome do arquivo
        name = csv_file.stem  # "params_100_samples"
        parts = name.split("_")
        if len(parts) == 3 and parts[2] == "samples":
            try:
                size = int(parts[1])
                sizes.append(size)
            except ValueError:
                continue

    return sorted(sizes)


def main():
    parser = argparse.ArgumentParser(
        description="Prepara datasets Sobol para treinamento da GNN"
    )
    parser.add_argument(
        "--n-samples", type=int, help="Número de amostras do dataset a processar"
    )
    parser.add_argument(
        "--all", action="store_true", help="Processar todos os datasets disponíveis"
    )

    args = parser.parse_args()

    # Configurar logger
    configure_logger(
        category="preprocessing", experiment_name="sobol_dataset", level="INFO"
    )

    logger.info("=" * 70)
    logger.info("PREPARAÇÃO DE DATASETS SOBOL")
    logger.info("=" * 70)

    if args.all:
        # Processar todos os datasets disponíveis
        available = find_available_datasets()

        if not available:
            logger.error(
                "Nenhum dataset encontrado em {}",
                paths.data.raw / "Sobol" / rho_max / "params",
            )
            return

        logger.info("Datasets disponíveis: {}", available)

        for n_samples in available:
            try:
                process_dataset(n_samples)
                logger.info("")
            except Exception as e:
                logger.error("Erro ao processar dataset {}: {}", n_samples, e)
                logger.exception("Detalhes do erro:")

    elif args.n_samples:
        # Processar dataset específico
        try:
            process_dataset(args.n_samples)
        except Exception as e:
            logger.error("Erro ao processar dataset: {}", e)
            logger.exception("Detalhes do erro:")

    else:
        logger.error("Especifique --n-samples ou --all")
        parser.print_help()
        return

    logger.info("=" * 70)
    logger.info("PROCESSAMENTO CONCLUÍDO")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
