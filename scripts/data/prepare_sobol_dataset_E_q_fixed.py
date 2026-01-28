"""
Prepara datasets Sobol para treinamento da GNN.

Lê CSVs raw (params + results), calcula features e monta arquivos .pt.
Suporta grafos de tamanhos variados (diferentes números de elementos por caso).

Inclui edge features (distância entre nós adjacentes) para uso com GATv2Conv.

Uso:
    python scripts/data/prepare_sobol_dataset_E_q_fixed.py --n-samples 2500
    python scripts/data/prepare_sobol_dataset_E_q_fixed.py --all
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch_geometric.data import Data

from src.paths import ensure_dir, paths
from src.utils.logger import configure_logger


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# Pasta onde estão os dados (ajuste conforme necessário)
RHO_FOLDER = "rho_0.010_E_q_fixed"


# =============================================================================
# FUNÇÕES DE CÁLCULO DE FEATURES (POR CASO INDIVIDUAL)
# =============================================================================


def compute_features_for_case(
    I: float, L: float, q: float, E: float, n_elements: int
) -> np.ndarray:
    """
    Calcula todas as features para um único caso.

    As features são calculadas para cada nó do grafo:
    - Globais (replicadas): I, L
    - Derivadas (replicadas): q_scale, q_over_EI
    - Posicionais (variam por nó): x, x_norm, x_norm², x_norm³, x_norm⁴

    Nota: q foi removido das features pois agora é constante.
          q_scale e q_over_EI são mantidos pois variam com I e L.

    Args:
        I: Momento de inércia [m^4]
        L: Comprimento da viga [m]
        q: Carga distribuída [N/m]
        E: Módulo de elasticidade [Pa]
        n_elements: Número de elementos neste caso

    Returns:
        Array shape (n_nodes, 9) com as features de cada nó
    """
    n_nodes = n_elements + 1

    # Features derivadas (constantes para todos os nós deste caso)
    # Estas ainda variam entre casos porque dependem de I e L
    q_scale = (q * L**4) / (E * I)
    q_over_EI = q / (E * I)

    # Features posicionais (variam por nó)
    node_indices = np.arange(n_nodes)
    x = (node_indices / n_elements) * L
    x_norm = node_indices / n_elements
    x_norm_2 = x_norm**2
    x_norm_3 = x_norm**3
    x_norm_4 = x_norm**4

    # Montar array de features: shape (n_nodes, 9)
    # Removido q (era coluna 2), agora temos 9 features ao invés de 10
    features = np.zeros((n_nodes, 9), dtype=np.float32)

    # Features globais (colunas 0-1) - replicadas para todos os nós
    features[:, 0] = I
    features[:, 1] = L
    # q removido - era constante

    # Features derivadas (colunas 2-3) - replicadas para todos os nós
    features[:, 2] = q_scale
    features[:, 3] = q_over_EI

    # Features posicionais (colunas 4-8) - variam por nó
    features[:, 4] = x
    features[:, 5] = x_norm
    features[:, 6] = x_norm_2
    features[:, 7] = x_norm_3
    features[:, 8] = x_norm_4

    return features


def create_edge_index_for_n_nodes(n_nodes: int) -> np.ndarray:
    """
    Cria edge_index para grafo linear bidirecional.

    Conecta cada nó ao seu vizinho imediato em ambas as direções.
    Para n_nodes nós, temos (n_nodes - 1) arestas em cada direção,
    totalizando 2 * (n_nodes - 1) arestas.

    Args:
        n_nodes: Número de nós no grafo

    Returns:
        Array shape (2, 2*(n_nodes-1)) com [source_nodes, target_nodes]
    """
    edges = []

    for i in range(n_nodes - 1):
        # Aresta i -> i+1
        edges.append([i, i + 1])
        # Aresta i+1 -> i (direção oposta)
        edges.append([i + 1, i])

    return np.array(edges, dtype=np.int64).T


# =============================================================================
# NOVA FUNÇÃO: CÁLCULO DE EDGE FEATURES
# =============================================================================


def compute_edge_features(x_positions: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """
    Calcula edge features para cada aresta do grafo.

    Para cada aresta (i, j), calcula a distância física |x_j - x_i|.
    Esta informação permite que a rede saiba a escala local da discretização:
    - Grafos com poucos elementos terão arestas "longas" (distância grande)
    - Grafos com muitos elementos terão arestas "curtas" (distância pequena)

    Args:
        x_positions: Array 1D com posições físicas dos nós [m]
        edge_index: Array shape (2, n_edges) com [sources, targets]

    Returns:
        Array shape (n_edges, 1) com a distância de cada aresta
    """
    sources = edge_index[0]
    targets = edge_index[1]

    # Distância física entre nós conectados
    distances = np.abs(x_positions[targets] - x_positions[sources])

    # Retornar como (n_edges, 1) para consistência com PyTorch Geometric
    return distances.reshape(-1, 1).astype(np.float32)


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================


def parse_displacement_string(disp_str: str) -> np.ndarray:
    """
    Converte string de deslocamentos separados por ; para array numpy.

    Args:
        disp_str: String no formato "val1;val2;val3;..."

    Returns:
        Array 1D com os valores de deslocamento
    """
    parts = disp_str.strip().split(";")
    return np.array([float(p.strip()) for p in parts], dtype=np.float32)


# =============================================================================
# FUNÇÃO PRINCIPAL DE PROCESSAMENTO
# =============================================================================


def process_dataset(n_samples: int) -> Path:
    """
    Processa um dataset completo (params + results) e salva arquivo .pt.

    Cada caso é convertido em um objeto Data do PyTorch Geometric,
    permitindo grafos de tamanhos variados. Inclui edge features.

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
        / RHO_FOLDER
        / f"params_{n_samples}_samples.csv"
    )
    results_path = (
        paths.data.raw
        / "Sobol"
        / "results"
        / RHO_FOLDER
        / f"results_{n_samples}_samples.csv"
    )
    output_path = paths.data.processed / "Sobol" / RHO_FOLDER / f"dataset_{n_samples}.pt"

    # Verificar existência dos arquivos
    if not params_path.exists():
        raise FileNotFoundError(f"Arquivo de parâmetros não encontrado: {params_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Arquivo de resultados não encontrado: {results_path}")

    logger.info("Carregando CSVs...")
    params_df = pd.read_csv(params_path)
    results_df = pd.read_csv(results_path)

    # Extrair case_id original (remover sufixo _nelXXX se existir)
    def extract_original_case_id(case_id: str) -> str:
        if "_nel" in case_id:
            return case_id.rsplit("_nel", 1)[0]
        return case_id

    results_df["case_id_original"] = results_df["case_id"].apply(extract_original_case_id)

    # Filtrar apenas resultados cujo case_id original existe nos parâmetros
    valid_case_ids = set(params_df["case_id"])
    results_df = results_df[results_df["case_id_original"].isin(valid_case_ids)].reset_index(drop=True)

    n_samples_actual = len(results_df)
    logger.info("Total de grafos a processar: {}", n_samples_actual)

    # Criar dicionário de parâmetros para lookup rápido
    params_dict = params_df.set_index("case_id").to_dict("index")

    # Processar cada caso individualmente
    logger.info("Processando casos individuais...")
    data_list: list[Data] = []
    n_elements_list: list[int] = []

    for idx in range(n_samples_actual):
        # Extrair case_id original para buscar parâmetros
        case_id_original = results_df.iloc[idx]["case_id_original"]
        params = params_dict[case_id_original]

        # Extrair parâmetros deste caso
        I = float(params["I"])
        L = float(params["L"])
        q = float(params["q"])
        E = float(params["E"])

        # Extrair resultados deste caso
        n_elements = int(results_df.iloc[idx]["n_elements"])
        displacements = parse_displacement_string(results_df.iloc[idx]["displacements"])

        # Validar consistência
        n_nodes = n_elements + 1
        if len(displacements) != n_nodes:
            raise ValueError(
                f"Caso {idx}: n_elements={n_elements} implica {n_nodes} nós, "
                f"mas displacements tem {len(displacements)} valores"
            )

        # Calcular node features
        features = compute_features_for_case(I, L, q, E, n_elements)

        # Criar edge_index
        edge_index = create_edge_index_for_n_nodes(n_nodes)

        # =====================================================================
        # NOVO: Calcular edge features (distância entre nós adjacentes)
        # =====================================================================
        # A coluna 4 das features contém as posições físicas x [m]
        x_positions = features[:, 4]
        edge_attr = compute_edge_features(x_positions, edge_index)

        # Criar objeto Data com edge_attr
        data = Data(
            x=torch.from_numpy(features),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),  # NOVO: edge features
            y=torch.from_numpy(displacements),
            n_elements=n_elements,
        )

        data_list.append(data)
        n_elements_list.append(n_elements)

        # Log de progresso a cada 10%
        if (idx + 1) % max(1, n_samples_actual // 10) == 0:
            logger.info("  Processados {}/{} casos", idx + 1, n_samples_actual)

    # Calcular estatísticas de n_elements
    n_elements_array = np.array(n_elements_list)
    unique_n_elements = np.unique(n_elements_array)

    logger.info("Estatísticas de n_elements:")
    logger.info("  Mínimo: {}", n_elements_array.min())
    logger.info("  Máximo: {}", n_elements_array.max())
    logger.info("  Média: {:.1f}", n_elements_array.mean())
    logger.info("  Valores únicos: {}", len(unique_n_elements))

    # Montar dicionário com metadados
    dataset_dict = {
        "data_list": data_list,
        "metadata": {
            "n_samples": n_samples_actual,
            "n_features": 9,
            "n_edge_features": 1,  # NOVO: documentar edge features
            "n_elements_min": int(n_elements_array.min()),
            "n_elements_max": int(n_elements_array.max()),
            "n_elements_mean": float(n_elements_array.mean()),
            "n_elements_unique": unique_n_elements.tolist(),
            "feature_names": {
                "global": ["I", "L"],
                "derived": ["q_scale", "q_over_EI"],
                "positional": [
                    "x",
                    "x_normalized",
                    "x_normalized_2",
                    "x_normalized_3",
                    "x_normalized_4",
                ],
                "all": [
                    "I", "L",
                    "q_scale", "q_over_EI",
                    "x", "x_normalized", "x_normalized_2",
                    "x_normalized_3", "x_normalized_4",
                ],
            },
            "edge_feature_names": ["distance"],  # NOVO
            "target_names": ["displacements"],
            "fixed_params": {
                "E": 200e9,
                "q": 5000.0,
            },
        },
    }

    # Salvar arquivo .pt
    ensure_dir(output_path.parent)
    torch.save(dataset_dict, output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Dataset salvo: {} ({:.2f} MB)", output_path, file_size_mb)

    return output_path


# =============================================================================
# CLI
# =============================================================================


def find_available_datasets() -> list[int]:
    """
    Busca todos os datasets disponíveis no diretório raw/Sobol/results.

    Procura no diretório de results porque é ele que contém a informação
    de n_elements por caso.

    Returns:
        Lista de tamanhos de datasets encontrados (ordenada)
    """
    results_dir = paths.data.raw / "Sobol" / "results" / RHO_FOLDER

    if not results_dir.exists():
        return []

    sizes = []
    for csv_file in results_dir.glob("results_*_samples.csv"):
        name = csv_file.stem  # "results_100_samples"
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
        description="Prepara datasets Sobol para treinamento da GNN (grafos de tamanhos variados)"
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
        category="preprocessing", experiment_name="sobol_dataset_variable", level="INFO"
    )

    logger.info("=" * 70)
    logger.info("PREPARAÇÃO DE DATASETS SOBOL (GRAFOS VARIÁVEIS + EDGE FEATURES)")
    logger.info("=" * 70)

    if args.all:
        available = find_available_datasets()

        if not available:
            logger.error(
                "Nenhum dataset encontrado em {}",
                paths.data.raw / "Sobol" / "results" / RHO_FOLDER,
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