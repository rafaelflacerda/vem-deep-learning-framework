"""
Processa datasets Sobol em partes e depois junta tudo.

Estratégia:
1. Processa cada results_part_XXX.csv separadamente -> dataset_part_XXX.pt
2. Junta todos os dataset_part_XXX.pt em um único dataset_final.pt

Uso:
    # Processar todas as partes
    python scripts/data/prepare_sobol_from_parts.py --process-parts --workers 14
    
    # Juntar todas as partes
    python scripts/data/prepare_sobol_from_parts.py --merge-parts
    
    # Fazer tudo de uma vez
    python scripts/data/prepare_sobol_from_parts.py --all --workers 14
"""

import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

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
N_SAMPLES = 1000000  # Usado apenas para encontrar o arquivo de params


# =============================================================================
# FUNÇÕES DE CÁLCULO (MESMAS DE ANTES)
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

    x_norm_powers = np.column_stack([
        x_norm,
        x_norm ** 2,
        x_norm ** 3,
        x_norm ** 4
    ])

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
    """Converte string de deslocamentos para array."""
    return np.fromstring(disp_str.replace(';', ' '), dtype=np.float32, sep=' ')


# =============================================================================
# PROCESSAMENTO PARALELO
# =============================================================================


def process_single_case(args):
    """Processa um único caso."""
    idx, row_data, params_dict = args
    
    try:
        case_id_original = row_data["case_id_original"]
        params = params_dict[case_id_original]

        I = params["I"]
        L = params["L"]
        q = params["q"]
        E = params["E"]

        n_elements = int(row_data["n_elements"])
        displacements = parse_displacement_string(row_data["displacements"])
        
        n_nodes = n_elements + 1
        if len(displacements) != n_nodes:
            return None

        features = compute_features_for_case(I, L, q, E, n_elements)
        edge_index = create_edge_index_for_n_nodes(n_nodes)

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
# PROCESSAMENTO DE UMA PARTE
# =============================================================================


def process_single_part(part_num: int, n_workers: int, params_dict: dict) -> Path:
    """
    Processa uma única parte (results_part_XXX.csv) e salva dataset_part_XXX.pt.
    
    Args:
        part_num: Número da parte (1-12)
        n_workers: Número de workers para multiprocessing
        params_dict: Dicionário com parâmetros (já carregado)
        
    Returns:
        Path do arquivo .pt criado
    """
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_descriptor')
    
    logger.info("=" * 70)
    logger.info(f"PROCESSANDO PARTE {part_num:03d}")
    logger.info("=" * 70)
    
    # Caminhos
    results_path = (
        paths.data.raw / "Sobol" / "results" / RHO_FOLDER 
        / f"results_part_{part_num:03d}.csv"
    )
    output_path = (
        paths.data.processed / "Sobol" / RHO_FOLDER 
        / f"dataset_part_{part_num:03d}.pt"
    )
    
    if not results_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {results_path}")
    
    logger.info("Carregando CSV da parte {}...", part_num)
    results_df = pd.read_csv(results_path)
    
    # Extrair case_id original
    def extract_original_case_id(case_id: str) -> str:
        if "_nel" in case_id:
            return case_id.rsplit("_nel", 1)[0]
        return case_id
    
    results_df["case_id_original"] = results_df["case_id"].apply(
        extract_original_case_id
    )
    
    n_samples_actual = len(results_df)
    logger.info("Total de grafos na parte {}: {}", part_num, n_samples_actual)
    
    # Preparar argumentos
    logger.info("Preparando dados...")
    results_records = results_df.to_dict('records')
    args_list = [
        (idx, results_records[idx], params_dict)
        for idx in range(n_samples_actual)
    ]
    
    # Processar em paralelo
    logger.info("Processando com {} workers...", n_workers)
    data_list = []
    n_elements_list = []
    
    with Pool(processes=n_workers) as pool:
        results = pool.imap_unordered(
            process_single_case, 
            args_list,
            chunksize=50
        )
        
        for i, result in enumerate(results, 1):
            if result is not None:
                data, n_elements = result
                data_list.append(data)
                n_elements_list.append(n_elements)
            
            if i % max(1, n_samples_actual // 10) == 0:
                logger.info("  Processados {}/{} casos", i, n_samples_actual)
    
    logger.info("Casos processados: {}/{}", len(data_list), n_samples_actual)
    
    # Estatísticas
    n_elements_array = np.array(n_elements_list)
    unique_n_elements = np.unique(n_elements_array)
    
    # Montar dicionário
    dataset_dict = {
        "data_list": data_list,
        "metadata": {
            "part_num": part_num,
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
    logger.info("Parte {} salva: {} ({:.2f} MB)", 
                part_num, output_path, file_size_mb)
    
    return output_path


# =============================================================================
# PROCESSAMENTO DE TODAS AS PARTES
# =============================================================================


def process_all_parts(n_workers: int) -> list[Path]:
    """
    Processa todas as partes (results_part_001.csv até results_part_012.csv).
    
    Args:
        n_workers: Número de workers para multiprocessing
        
    Returns:
        Lista de paths dos arquivos .pt criados
    """
    logger.info("=" * 70)
    logger.info("PROCESSAMENTO DE TODAS AS PARTES")
    logger.info("=" * 70)
    
    # Carregar params uma única vez
    params_path = (
        paths.data.raw / "Sobol" / "params" / RHO_FOLDER 
        / f"params_{N_SAMPLES}_samples.csv"
    )
    
    if not params_path.exists():
        raise FileNotFoundError(f"Params não encontrado: {params_path}")
    
    logger.info("Carregando parâmetros globais...")
    params_df = pd.read_csv(params_path)
    params_dict = params_df.set_index("case_id").to_dict("index")
    logger.info("Parâmetros carregados: {} casos", len(params_dict))
    
    # Descobrir quantas partes existem
    results_dir = paths.data.raw / "Sobol" / "results" / RHO_FOLDER
    part_files = sorted(results_dir.glob("results_part_*.csv"))
    
    if not part_files:
        raise FileNotFoundError(f"Nenhum results_part_*.csv encontrado em {results_dir}")
    
    logger.info("Partes encontradas: {}", len(part_files))
    for f in part_files:
        logger.info("  - {}", f.name)
    
    # Processar cada parte
    output_files = []
    for part_file in part_files:
        # Extrair número da parte do nome do arquivo
        part_num = int(part_file.stem.split('_')[-1])
        
        try:
            output_path = process_single_part(part_num, n_workers, params_dict)
            output_files.append(output_path)
            logger.info("")
        except Exception as e:
            logger.error("Erro ao processar parte {}: {}", part_num, e)
            logger.exception("Detalhes:")
    
    return output_files


# =============================================================================
# JUNÇÃO DAS PARTES
# =============================================================================


def merge_all_parts() -> Path:
    """
    Junta todos os dataset_part_XXX.pt em um único dataset_final.pt.
    
    Returns:
        Path do arquivo final
    """
    logger.info("=" * 70)
    logger.info("JUNÇÃO DE TODAS AS PARTES")
    logger.info("=" * 70)
    
    parts_dir = paths.data.processed / "Sobol" / RHO_FOLDER
    part_files = sorted(parts_dir.glob("dataset_part_*.pt"))
    
    if not part_files:
        raise FileNotFoundError(f"Nenhum dataset_part_*.pt encontrado em {parts_dir}")
    
    logger.info("Partes encontradas: {}", len(part_files))
    for f in part_files:
        logger.info("  - {}", f.name)
    
    # Carregar primeira parte como base
    logger.info("Carregando parte base...")
    merged_dict = torch.load(part_files[0])
    total_samples = len(merged_dict["data_list"])
    logger.info("  Amostras na base: {}", total_samples)
    
    # Juntar demais partes
    for i, filepath in enumerate(part_files[1:], start=2):
        logger.info("Carregando parte {}/{}...", i, len(part_files))
        
        current_dict = torch.load(filepath)
        n_samples = len(current_dict["data_list"])
        logger.info("  Amostras: {}", n_samples)
        
        merged_dict["data_list"].extend(current_dict["data_list"])
        total_samples += n_samples
    
    # Atualizar metadados
    logger.info("Atualizando metadados...")
    n_elements_list = [data.n_elements for data in merged_dict["data_list"]]
    n_elements_array = np.array(n_elements_list)
    unique_n_elements = np.unique(n_elements_array)
    
    merged_dict["metadata"]["n_samples"] = total_samples
    merged_dict["metadata"]["n_elements_min"] = int(n_elements_array.min())
    merged_dict["metadata"]["n_elements_max"] = int(n_elements_array.max())
    merged_dict["metadata"]["n_elements_mean"] = float(n_elements_array.mean())
    merged_dict["metadata"]["n_elements_unique"] = unique_n_elements.tolist()
    
    # Remover metadata específico de parte
    if "part_num" in merged_dict["metadata"]:
        del merged_dict["metadata"]["part_num"]
    
    # Salvar
    output_path = parts_dir / f"dataset_{N_SAMPLES}.pt"
    logger.info("Salvando arquivo final: {}", output_path)
    torch.save(merged_dict, output_path)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Total de amostras: {}", total_samples)
    logger.info("Tamanho do arquivo: {:.2f} MB", file_size_mb)
    
    return output_path


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Processa datasets Sobol em partes e junta"
    )
    parser.add_argument(
        "--process-parts",
        action="store_true",
        help="Processar todas as partes (results_part_XXX.csv -> dataset_part_XXX.pt)"
    )
    parser.add_argument(
        "--merge-parts",
        action="store_true",
        help="Juntar todas as partes (dataset_part_XXX.pt -> dataset_final.pt)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fazer tudo: processar + juntar"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de workers (padrão: CPU count - 2)"
    )

    args = parser.parse_args()
    
    if args.workers is None:
        args.workers = max(1, cpu_count() - 2)

    configure_logger(
        category="preprocessing",
        experiment_name="sobol_from_parts",
        level="INFO"
    )

    logger.info("=" * 70)
    logger.info("PREPARAÇÃO DE DATASETS SOBOL EM PARTES")
    logger.info("=" * 70)

    try:
        if args.all or args.process_parts:
            process_all_parts(args.workers)
        
        if args.all or args.merge_parts:
            merge_all_parts()
        
        if not (args.all or args.process_parts or args.merge_parts):
            logger.error("Especifique --process-parts, --merge-parts ou --all")
            parser.print_help()
            return

    except Exception as e:
        logger.error("Erro: {}", e)
        logger.exception("Detalhes:")
        return

    logger.info("=" * 70)
    logger.info("PROCESSAMENTO CONCLUÍDO")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()