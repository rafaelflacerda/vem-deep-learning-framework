"""
Funções de I/O para dados do solver VEM.
"""

import csv
import json
from pathlib import Path


def load_params_json(filepath: Path) -> dict:
    """Carrega arquivo JSON de parâmetros."""
    with open(filepath) as f:
        return json.load(f)


def save_results_json(results: dict, filepath: Path) -> None:
    """Salva resultados em JSON."""
    with open(filepath, "w") as f:
        json.dump(results, f)


def save_results_json_pretty(results: dict, filepath: Path) -> None:
    """Salva resultados em JSON formatado (para leitura humana)."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)


def params_json_to_csv(json_path: Path, csv_path: Path | None = None) -> Path:
    """
    Converte arquivo JSON de parâmetros para CSV.

    Args:
        json_path: Caminho do arquivo JSON.
        csv_path: Caminho do CSV de saída. Se None, usa mesmo nome do JSON.

    Returns:
        Caminho do arquivo CSV criado.
    """
    if csv_path is None:
        csv_path = json_path.with_suffix(".csv")

    params = load_params_json(json_path)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "L", "q", "E", "I"])

        for case_id in sorted(params.keys()):
            p = params[case_id]
            writer.writerow([case_id, p["L"], p["q"], p["E"], p["I"]])

    return csv_path


def results_csv_to_json(csv_path: Path, json_path: Path | None = None) -> Path:
    """
    Converte arquivo CSV de resultados para JSON.

    Args:
        csv_path: Caminho do arquivo CSV.
        json_path: Caminho do JSON de saída. Se None, usa mesmo nome do CSV.

    Returns:
        Caminho do arquivo JSON criado.
    """
    if json_path is None:
        json_path = csv_path.with_suffix(".json")

    results = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            case_id = row["case_id"]
            n_elements = int(row["n_elements"])

            displacements = [float(x) for x in row["displacements"].split(";") if x]
            rotations = [float(x) for x in row["rotations"].split(";") if x]

            results[case_id] = {
                "n_elements": n_elements,
                "displacements": displacements,
                "rotations": rotations,
            }

    save_results_json(results, json_path)
    return json_path


def append_result_to_csv(
    filepath: Path,
    case_id: str,
    n_elements: int,
    displacements: list[float],
    rotations: list[float],
) -> None:
    """Adiciona uma linha de resultado ao CSV."""
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        disp_str = ";".join(f"{x:.17g}" for x in displacements)
        rot_str = ";".join(f"{x:.17g}" for x in rotations)
        writer.writerow([case_id, n_elements, disp_str, rot_str])


def init_results_csv(filepath: Path) -> None:
    """Cria arquivo CSV de resultados com header."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "n_elements", "displacements", "rotations"])


def load_params_csv(filepath: Path) -> dict:
    """Carrega arquivo CSV de parâmetros."""
    params = {}

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            params[row["case_id"]] = {
                "L": float(row["L"]),
                "q": float(row["q"]),
                "E": float(row["E"]),
                "I": float(row["I"]),
            }

    return params
