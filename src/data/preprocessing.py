"""
Funções para preprocessamento de dados VEM.

Converte JSONs raw (params + results) em arquivos .pt otimizados para treino.
"""

import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from src.paths import paths


def load_feature_config(config_path: Path) -> dict[str, Any]:
    """
    Carrega configuração de features do arquivo YAML.

    Args:
        config_path: Path para o arquivo YAML de configuração.

    Returns:
        Dicionário com configuração de features, targets e dataset_sizes.

    Raises:
        FileNotFoundError: Se o arquivo de configuração não existir.
        ValueError: Se o YAML estiver malformado ou faltar campos obrigatórios.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {config_path}"
        )

    try:
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Erro ao carregar YAML: {e}")

    # Validar campos obrigatórios
    required_fields = ["features", "targets", "dataset_sizes"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Campo obrigatório '{field}' ausente no YAML")

    # Validar estrutura de features
    if "global" not in config["features"]:
        raise ValueError("Campo 'features.global' ausente no YAML")

    logger.debug(
        "Configuração carregada: {} features globais, {} globais derivadas, {} per-node, {} targets",
        len(config["features"].get("global", [])),
        len(config["features"].get("global_derived", [])),
        len(config["features"].get("per_node", [])),
        len(config["targets"]),
    )

    return config


def compute_global_features(
    params_json: dict[str, Any], feature_names: list[str]
) -> list[float]:
    """
    Extrai features globais (brutas) de params JSON.

    Args:
        params_json: Dicionário carregado de params/case_XXXXXX.json.
        feature_names: Lista de nomes de features globais.

    Returns:
        Lista de valores das features globais.
    """
    features = []
    for name in feature_names:
        if name not in params_json:
            raise KeyError(f"Feature global '{name}' não encontrada em params JSON")
        features.append(float(params_json[name]))
    return features


def compute_derived_features(
    global_features: dict[str, float], derived_names: list[str]
) -> list[float]:
    """
    Calcula features derivadas a partir de features globais.

    Args:
        global_features: Dicionário {nome: valor} das features globais.
        derived_names: Lista de nomes de features derivadas a calcular.

    Returns:
        Lista de valores das features derivadas.
    """
    derived = []

    for name in derived_names:
        if name == "EI":
            value = global_features["E"] * global_features["I"]
        elif name == "q_scale":
            E = global_features["E"]
            I = global_features["I"]
            L = global_features["L"]
            q = global_features["q"]
            value = q * L**4 / (E * I)
        elif name == "q_over_EI":
            E = global_features["E"]
            I = global_features["I"]
            q = global_features["q"]
            value = q / (E * I)
        else:
            raise ValueError(f"Feature derivada '{name}' não reconhecida")

        derived.append(float(value))

    return derived


def compute_per_node_features(
    node_coords: list[float], L: float, per_node_names: list[str]
) -> list[list[float]]:
    """
    Calcula features específicas por nó.

    Args:
        node_coords: Lista de coordenadas dos nós.
        L: Comprimento total da viga.
        per_node_names: Lista de nomes de features por nó.

    Returns:
        Lista de listas, onde cada sublista contém as features de um nó.
        Shape: (n_nodes, n_per_node_features)
    """
    n_nodes = len(node_coords)
    per_node_features = []

    for x_i in node_coords:
        node_feats = []
        x_norm = x_i / L if L != 0 else 0.0

        for name in per_node_names:
            if name == "x":
                value = x_i
            elif name == "x_normalized":
                value = x_norm
            elif name == "x_normalized_2":
                value = x_norm**2
            elif name == "x_normalized_3":
                value = x_norm**3
            elif name == "x_normalized_4":
                value = x_norm**4
            else:
                raise ValueError(f"Feature per-node '{name}' não reconhecida")
            node_feats.append(float(value))
        per_node_features.append(node_feats)

    return per_node_features


def extract_features_per_sample(
    params_json: dict[str, Any],
    global_names: list[str],
    global_derived_names: list[str],
    per_node_names: list[str],
) -> torch.Tensor:
    """
    Extrai todas as features para uma amostra.

    Args:
        params_json: Dicionário do params JSON.
        global_names: Features globais brutas.
        global_derived_names: Features globais derivadas.
        per_node_names: Features por nó.

    Returns:
        Tensor de shape (n_nodes, n_total_features) onde:
        n_total_features = len(global) + len(global_derived) + len(per_node)
    """
    # 1. Features globais brutas
    global_values = compute_global_features(params_json, global_names)
    global_dict = {
        name: val for name, val in zip(global_names, global_values, strict=False)
    }

    # 2. Features globais derivadas
    derived_values = compute_derived_features(global_dict, global_derived_names)

    # 3. Features por nó
    node_coords = params_json["node_coords"]
    L = global_dict["L"]
    per_node_values = compute_per_node_features(node_coords, L, per_node_names)

    # 4. Combinar tudo
    # Para cada nó: [globais, derivadas, per_node_específicas_desse_nó]
    n_nodes = len(node_coords)
    combined_features = []

    for i in range(n_nodes):
        node_features = (
            global_values  # Features globais (mesmas para todos)
            + derived_values  # Features derivadas (mesmas para todos)
            + per_node_values[i]  # Features específicas deste nó
        )
        combined_features.append(node_features)

    return torch.tensor(combined_features, dtype=torch.float32)


def extract_targets(
    results_json: dict[str, Any], target_names: list[str]
) -> torch.Tensor:
    """
    Extrai targets de results JSON.

    Args:
        results_json: Dicionário carregado de results/result_XXXXXX.json.
        target_names: Lista de nomes de targets a extrair.

    Returns:
        Tensor de shape (n_nodes,) ou (n_nodes, n_targets)
    """
    targets = []

    for name in target_names:
        if name not in results_json:
            raise KeyError(f"Target '{name}' não encontrado em results JSON")

        target_values = results_json[name]

        if not isinstance(target_values, list):
            raise ValueError(
                f"Target '{name}' deveria ser lista, mas é {type(target_values)}"
            )

        targets.append([float(v) for v in target_values])

    # Se apenas 1 target, retorna (n_nodes,)
    # Se múltiplos targets, retorna (n_nodes, n_targets)
    if len(targets) == 1:
        return torch.tensor(targets[0], dtype=torch.float32)
    else:
        return torch.tensor(targets, dtype=torch.float32).T


def preprocess_single_dataset(
    n_samples: int,
    global_names: list[str],
    global_derived_names: list[str],
    per_node_names: list[str],
    target_names: list[str],
    raw_base_dir: Path,
    processed_base_dir: Path,
) -> Path:
    """
    Preprocessa um único dataset (converte JSONs → .pt).

    Args:
        n_samples: Número de amostras no dataset.
        global_names: Features globais brutas.
        global_derived_names: Features globais derivadas.
        per_node_names: Features por nó.
        target_names: Targets a extrair.
        raw_base_dir: Diretório raiz dos dados raw.
        processed_base_dir: Diretório raiz dos dados processados.

    Returns:
        Path do arquivo .pt criado.
    """
    dataset_name = f"{n_samples}_samples"
    raw_dir = raw_base_dir / dataset_name

    if not raw_dir.exists():
        raise FileNotFoundError(f"Diretório raw não encontrado: {raw_dir}")

    params_dir = raw_dir / "params"
    results_dir = raw_dir / "results"

    if not params_dir.exists():
        raise FileNotFoundError(f"Diretório params não encontrado: {params_dir}")
    if not results_dir.exists():
        raise FileNotFoundError(f"Diretório results não encontrado: {results_dir}")

    logger.info("Processando dataset: {} ({} amostras)", dataset_name, n_samples)

    features_list = []
    targets_list = []
    case_ids = []

    # Processar cada amostra
    for i in range(n_samples):
        case_filename = f"case_{i:06d}.json"
        result_filename = f"result_{i:06d}.json"

        params_path = params_dir / case_filename
        results_path = results_dir / result_filename

        # Carregar JSONs
        with open(params_path) as f:
            params_json = json.load(f)

        with open(results_path) as f:
            results_json = json.load(f)

        # Extrair features (retorna tensor (n_nodes, n_features))
        features = extract_features_per_sample(
            params_json, global_names, global_derived_names, per_node_names
        )

        # Extrair targets (retorna tensor (n_nodes,) ou (n_nodes, n_targets))
        targets = extract_targets(results_json, target_names)

        features_list.append(features)
        targets_list.append(targets)
        case_ids.append(params_json.get("case_id", f"case_{i:06d}"))

        # Log progresso
        if (i + 1) % max(1, n_samples // 10) == 0 or (i + 1) % 1000 == 0:
            logger.info("Progresso: {}/{} amostras processadas", i + 1, n_samples)

    # Stack features e targets
    # features: (n_samples, n_nodes, n_features)
    # targets: (n_samples, n_nodes) ou (n_samples, n_nodes, n_targets)
    features_tensor = torch.stack(features_list)
    targets_tensor = torch.stack(targets_list)

    logger.debug(
        "Shapes: features={}, targets={}", features_tensor.shape, targets_tensor.shape
    )

    # Montar dicionário do dataset
    dataset = {
        "features": features_tensor,
        "targets": targets_tensor,
        "metadata": {
            "case_ids": case_ids,
            "n_samples": n_samples,
            "n_nodes": features_tensor.shape[1],
            "feature_names": {
                "global": global_names,
                "global_derived": global_derived_names,
                "per_node": per_node_names,
                "all": global_names + global_derived_names + per_node_names,
            },
            "target_names": target_names,
        },
    }

    # Salvar .pt
    processed_base_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"dataset_{n_samples}.pt"
    output_path = processed_base_dir / output_filename

    torch.save(dataset, output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Dataset salvo: {} ({:.2f} MB)", output_path, file_size_mb)

    return output_path


def preprocess_all_datasets(
    config_path: Path,
    raw_base_dir: Path | None = None,
    processed_base_dir: Path | None = None,
) -> list[Path]:
    """
    Preprocessa todos os datasets especificados na configuração.

    Args:
        config_path: Path para o arquivo YAML de configuração.
        raw_base_dir: Diretório raiz dos dados raw. Se None, usa paths.data.raw.
        processed_base_dir: Diretório raiz dos processados. Se None, usa paths.data.processed.

    Returns:
        Lista de Paths dos arquivos .pt criados.
    """
    # Carregar configuração
    config = load_feature_config(config_path)

    global_names = config["features"]["global"]
    global_derived_names = config["features"].get("global_derived", [])
    per_node_names = config["features"].get("per_node", [])
    target_names = config["targets"]
    dataset_sizes = config["dataset_sizes"]

    # Usar paths do projeto se não fornecido
    if raw_base_dir is None:
        raw_base_dir = paths.data.raw
    if processed_base_dir is None:
        processed_base_dir = paths.data.processed

    logger.info("Iniciando preprocessamento de {} datasets", len(dataset_sizes))
    logger.info("Features globais: {}", ", ".join(global_names))
    logger.info("Features derivadas: {}", ", ".join(global_derived_names))
    logger.info("Features per-node: {}", ", ".join(per_node_names))
    logger.info("Targets: {}", ", ".join(target_names))

    output_paths = []  # ← ADICIONE ESTA LINHA

    # ← ADICIONE TODO ESTE BLOCO:
    for n_samples in dataset_sizes:
        try:
            output_path = preprocess_single_dataset(
                n_samples=n_samples,
                global_names=global_names,
                global_derived_names=global_derived_names,
                per_node_names=per_node_names,
                target_names=target_names,
                raw_base_dir=raw_base_dir,
                processed_base_dir=processed_base_dir,
            )
            output_paths.append(output_path)
        except Exception as e:
            logger.error("Erro ao processar dataset {}_samples: {}", n_samples, e)
            raise

    logger.info("Preprocessamento concluído! {} datasets criados", len(output_paths))

    return output_paths
