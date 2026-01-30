"""
Prepara datasets Sobol para treinamento da GNN (VERSÃO OTIMIZADA).

Usa multiprocessing para processar casos em paralelo.
Otimizado para máquinas com múltiplas vCPUs.

Uso:
    python scripts/data/prepare_sobol_dataset_E_q_fixed.py --n-samples 1000000 --workers 14
    python scripts/data/prepare_sobol_dataset_E_q_fixed.py --all --workers 20
"""

import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

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

RHO_FOLDER = "slenderness_method"


# =============================================================================
# FUNÇÕES DE CÁLCULO (OTIMIZADAS)
# =============================================================================


def compute_features_for_case(
    I: float, L: float, q: float, E: float, n_elements: int
) -> np.ndarray:
    """Calcula features para um caso."""
    n_nodes = n_elements + 1

    q_scale = (q * L**4) / (E * I)
    q_over_EI = q / (E * I)

    node_indices = np.arange(n_nodes, dtype=np.float32)
    x_norm = node_indices / n_elements
    x = x_norm * L

    # Vetorizar potências
    x_norm_powers = np.column_stack([
        x_norm,
        x_norm ** 2,
        x_norm ** 3,
        x_norm ** 4
    ])

    # Montar array de uma vez
    features = np.empty((n_nodes, 9), dtype=np.float32)
    features[:, 0] = I
    features[:, 1] = L
    features[:, 2] = q_scale
    features[:, 3] = q_over_EI
    features[:, 4] = x
    features[:, 5:9] = x_norm_powers

    return features


def create_edge_index_for_n_nodes(n_nodes: int) -> np.ndarray:
    """Cria edge_index para grafo linear bidirecional."""
    n_edges = n_nodes - 1
    
    # Criar todas as arestas de uma vez
    forward = np.column_stack([
        np.arange(n_edges),
        np.arange(1, n_nodes)
    ])
    backward = np.column_stack([
        np.arange(1, n_nodes),
        np.arange(n_edges)
    ])
    
    edges = np.vstack([forward, backward])
    return edges.T.astype(np.int64)


def parse_displacement_string(disp_str: str) -> np.ndarray:
    """Converte string de deslocamentos para array (otimizado)."""
    return np.fromstring(disp_str.replace(';', ' '), dtype=np.float32, sep=' ')


# =============================================================================
# PROCESSAMENTO PARALELO
# =============================================================================


def process_single_case(args):
    """
    Processa um único caso. Função separada para uso com multiprocessing.
    
    Args:
        args: tupla (idx, row_data, params_dict)
        
    Returns:
        tupla (Data, n_elements) ou None se erro
    """
    idx, row_data, params_dict = args
    
    try:
        # Extrair case_id original
        case_id_original = row_data["case_id_original"]
        params = params_dict[case_id_original]

        # Extrair parâmetros (acessar diretamente, sem conversão repetida)
        I = params["I"]
        L = params["L"]
        q = params["q"]
        E = params["E"]

        # Extrair resultados
        n_elements = int(row_data["n_elements"])
        displacements = parse_displacement_string(row_data["displacements"])
        
        # Validar
        n_nodes = n_elements + 1
        if len(displacements) != n_nodes:
            logger.warning(
                f"Caso {idx}: inconsistência n_elements={n_elements} vs "
                f"len(displacements)={len(displacements)}, pulando..."
            )
            return None

        # Calcular features e edge_index
        features = compute_features_for_case(I, L, q, E, n_elements)
        edge_index = create_edge_index_for_n_nodes(n_nodes)

        # Criar objeto Data
        data = Data(
            x=torch.from_numpy(features),
            edge_index=torch.from_numpy(edge_index),
            y=torch.from_numpy(displacements),
            n_elements=n_elements,
        )

        return (data, n_elements)
        
    except Exception as e:
        logger.warning(f"Erro ao processar caso {idx}: {e}")
        return None


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================


def process_dataset(n_samples: int, n_workers: int = None) -> Path:
    """
    Processa dataset com multiprocessing.
    
    Args:
        n_samples: Número de amostras
        n_workers: Número de workers (None = usa todos os cores)
        
    Returns:
        Path do arquivo .pt criado
    """
    # MUDANÇA 1: Estratégia de compartilhamento PyTorch
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_descriptor')
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 2)
    
    logger.info("Processando dataset com {} amostras usando {} workers", 
                n_samples, n_workers)

    # Caminhos
    params_path = (
        paths.data.raw / "Sobol" / "params" / RHO_FOLDER 
        / f"params_{n_samples}_samples.csv"
    )
    results_path = (
        paths.data.raw / "Sobol" / "results" / RHO_FOLDER 
        / f"results_{n_samples}_samples.csv"
    )
    output_path = (
        paths.data.processed / "Sobol" / RHO_FOLDER 
        / f"dataset_{n_samples}.pt"
    )

    if not params_path.exists():
        raise FileNotFoundError(f"Params não encontrado: {params_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Results não encontrado: {results_path}")

    logger.info("Carregando CSVs...")
    params_df = pd.read_csv(params_path)
    results_df = pd.read_csv(results_path)

    # Extrair case_id original
    def extract_original_case_id(case_id: str) -> str:
        if "_nel" in case_id:
            return case_id.rsplit("_nel", 1)[0]
        return case_id

    results_df["case_id_original"] = results_df["case_id"].apply(
        extract_original_case_id
    )

    # Filtrar válidos
    valid_case_ids = set(params_df["case_id"])
    results_df = results_df[
        results_df["case_id_original"].isin(valid_case_ids)
    ].reset_index(drop=True)

    n_samples_actual = len(results_df)
    logger.info("Total de grafos: {}", n_samples_actual)

    # Converter params_df para dict
    params_dict = params_df.set_index("case_id").to_dict("index")

    # MUDANÇA 2: Preparação otimizada de args
    logger.info("Preparando dados para processamento paralelo...")
    results_records = results_df.to_dict('records')
    args_list = [
        (idx, results_records[idx], params_dict)
        for idx in range(n_samples_actual)
    ]

    # Processar em paralelo
    logger.info("Iniciando processamento paralelo...")
    data_list = []
    n_elements_list = []
    
    with Pool(processes=n_workers) as pool:
        # MUDANÇA 3: chunksize fixo menor
        results = pool.imap_unordered(
            process_single_case, 
            args_list,
            chunksize=1000
        )
        
        # Coletar resultados
        for i, result in enumerate(results, 1):
            if result is not None:
                data, n_elements = result
                data_list.append(data)
                n_elements_list.append(n_elements)
            
            # Log progresso
            if i % max(1, n_samples_actual // 20) == 0:
                logger.info("  Processados {}/{} casos", i, n_samples_actual)

    logger.info("Casos processados com sucesso: {}/{}", 
                len(data_list), n_samples_actual)

    # Estatísticas
    n_elements_array = np.array(n_elements_list)
    unique_n_elements = np.unique(n_elements_array)

    logger.info("Estatísticas de n_elements:")
    logger.info("  Mínimo: {}", n_elements_array.min())
    logger.info("  Máximo: {}", n_elements_array.max())
    logger.info("  Média: {:.1f}", n_elements_array.mean())
    logger.info("  Valores únicos: {}", len(unique_n_elements))

    # Montar dicionário
    dataset_dict = {
        "data_list": data_list,
        "metadata": {
            "n_samples": len(data_list),
            "n_features": 9,
            "n_elements_min": int(n_elements_array.min()),
            "n_elements_max": int(n_elements_array.max()),
            "n_elements_mean": float(n_elements_array.mean()),
            "n_elements_unique": unique_n_elements.tolist(),
            "feature_names": {
                "global": ["I", "L"],
                "derived": ["q_scale", "q_over_EI"],
                "positional": [
                    "x", "x_normalized", "x_normalized_2",
                    "x_normalized_3", "x_normalized_4",
                ],
                "all": [
                    "I", "L", "q_scale", "q_over_EI",
                    "x", "x_normalized", "x_normalized_2",
                    "x_normalized_3", "x_normalized_4",
                ],
            },
            "target_names": ["displacements"],
            "fixed_params": {"E": 200e9, "q": 5000.0},
        },
    }

    # Salvar
    ensure_dir(output_path.parent)
    torch.save(dataset_dict, output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Dataset salvo: {} ({:.2f} MB)", output_path, file_size_mb)

    return output_path


# =============================================================================
# CLI
# =============================================================================


def find_available_datasets() -> list[int]:
    """Busca datasets disponíveis."""
    results_dir = paths.data.raw / "Sobol" / "results" / RHO_FOLDER

    if not results_dir.exists():
        return []

    sizes = []
    for csv_file in results_dir.glob("results_*_samples.csv"):
        name = csv_file.stem
        parts = name.split("_")
        if len(parts) == 3 and parts[2] == "samples":
            try:
                sizes.append(int(parts[1]))
            except ValueError:
                continue

    return sorted(sizes)


def main():
    parser = argparse.ArgumentParser(
        description="Prepara datasets Sobol (versão otimizada com multiprocessing)"
    )
    parser.add_argument("--n-samples", type=int, help="Número de amostras")
    parser.add_argument("--all", action="store_true", help="Processar todos")
    parser.add_argument(
        "--workers", 
        type=int, 
        default=None,
        help="Número de workers (padrão: CPU count - 2)"
    )

    args = parser.parse_args()

    configure_logger(
        category="preprocessing", 
        experiment_name="sobol_dataset_variable", 
        level="INFO"
    )

    logger.info("=" * 70)
    logger.info("PREPARAÇÃO DE DATASETS SOBOL (VERSÃO PARALELA)")
    logger.info("=" * 70)

    if args.all:
        available = find_available_datasets()
        if not available:
            logger.error("Nenhum dataset encontrado")
            return

        logger.info("Datasets disponíveis: {}", available)

        for n_samples in available:
            try:
                process_dataset(n_samples, args.workers)
                logger.info("")
            except Exception as e:
                logger.error("Erro: {}", e)
                logger.exception("Detalhes:")

    elif args.n_samples:
        try:
            process_dataset(args.n_samples, args.workers)
        except Exception as e:
            logger.error("Erro: {}", e)
            logger.exception("Detalhes:")

    else:
        logger.error("Especifique --n-samples ou --all")
        parser.print_help()
        return

    logger.info("=" * 70)
    logger.info("PROCESSAMENTO CONCLUÍDO")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()