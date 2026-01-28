"""
Filtra resultados do solver VEM removendo casos com erro alto em relação à solução analítica.

Para cada caso, calcula a soma dos deslocamentos nodais (VEM vs analítico).
Remove casos onde |soma_vem - soma_analitica| > threshold.

Uso:
    python scripts/data/filter_results_by_analytical.py \
        --results data/raw/Sobol/results/teste2/results_25000_samples.csv \
        --params data/raw/Sobol/params/rho_0.010_E_fixed_102_elements/params_25000_samples.csv \
        --dry-run

    # Com threshold customizado:
    python scripts/data/filter_results_by_analytical.py \
        --results results.csv --params params.csv --threshold 1e-8
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def compute_analytical_displacements(
    L: float, q: float, E: float, I: float, n_elements: int
) -> np.ndarray:
    """
    Calcula deslocamentos analíticos para viga cantilever com carga uniforme.

    Fórmula: w(x) = q / (24*E*I) * x^2 * (x^2 - 4*L*x + 6*L^2)

    Args:
        L: Comprimento da viga [m]
        q: Carga distribuída [N/m] (positivo = para baixo)
        E: Módulo de elasticidade [Pa]
        I: Momento de inércia [m^4]
        n_elements: Número de elementos

    Returns:
        Array com deslocamentos em cada nó (negativos = para baixo)
    """
    n_nodes = n_elements + 1
    x = np.linspace(0, L, n_nodes)

    # w(x) = q / (24*E*I) * x^2 * (x^2 - 4*L*x + 6*L^2)
    # Sinal negativo porque q positivo causa deslocamento para baixo
    coef = -q / (24 * E * I)
    w = coef * x**2 * (x**2 - 4 * L * x + 6 * L**2)

    return w


def parse_displacement_string(disp_str: str) -> list[float]:
    """Converte string de deslocamentos separados por ; para lista de floats."""
    return [float(v) for v in disp_str.split(";")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filtra resultados VEM por comparação com solução analítica"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Caminho para CSV de resultados (case_id, n_elements, displacements, rotations)",
    )
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="Caminho para CSV de parâmetros (case_id, L, q, E, I)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5e-3,
        help="Threshold para remoção (default: 1e-10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas calcula estatisticas sem criar arquivo filtrado",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    params_path = Path(args.params)
    threshold = args.threshold

    if not results_path.exists():
        raise FileNotFoundError(f"Arquivo de resultados não encontrado: {results_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"Arquivo de parâmetros não encontrado: {params_path}")

    # Carregar parâmetros em um dicionário
    params_dict: dict[str, tuple[float, float, float, float]] = {}
    with open(params_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row["case_id"].strip()
            L = float(row["L"])
            q = float(row["q"])
            E = float(row["E"])
            I = float(row["I"])
            params_dict[case_id] = (L, q, E, I)

    # Processar resultados
    output_path = results_path.with_name(
        results_path.stem + "_filtered" + results_path.suffix
    )

    total = 0
    kept = 0
    removed = 0
    removed_by_n_elements: dict[int, int] = {}
    errors: list[tuple[str, int, float]] = []
    
    if args.dry_run:
        f_out = None
        writer = None
    else:
        f_out = open(output_path, "w", newline="")
        writer = None  # será criado depois de ler o header

    try:
        with open(results_path, newline="") as f_in:
            reader = csv.DictReader(f_in)
            
            if not args.dry_run:
                writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
                writer.writeheader()

            for row in reader:
                total += 1
                case_id_full = row["case_id"].strip()
                n_elements = int(row["n_elements"])
                displacements_vem = parse_displacement_string(row["displacements"])

                # Extrair case_id original (remover sufixo _nelXXX se existir)
                if "_nel" in case_id_full:
                    case_id = case_id_full.rsplit("_nel", 1)[0]
                else:
                    case_id = case_id_full

                if case_id not in params_dict:
                    print(f"AVISO: case_id {case_id} não encontrado nos parâmetros, pulando...")
                    removed += 1
                    continue

                L, q, E, I = params_dict[case_id]

                # Calcular solução analítica
                displacements_analytical = compute_analytical_displacements(L, q, E, I, n_elements)

                # Calcular erro
                soma_vem = sum(abs(d) for d in displacements_vem)
                soma_analytical = sum(abs(d) for d in displacements_analytical)
                erro = abs(soma_vem - soma_analytical)

                if erro > threshold:
                    removed += 1
                    removed_by_n_elements[n_elements] = removed_by_n_elements.get(n_elements, 0) + 1
                    errors.append((case_id_full, n_elements, erro))
                else:
                    kept += 1
                    if writer is not None:
                        writer.writerow(row)
    finally:
        if f_out is not None:
            f_out.close()

    # Estatísticas
    print("=" * 70)
    print("FILTRAGEM DE RESULTADOS POR SOLUÇÃO ANALÍTICA")
    print("=" * 70)
    print(f"Arquivo de entrada:  {results_path}")
    if args.dry_run:
        print(f"Arquivo de saída:    (dry-run, nenhum arquivo criado)")
    else:
        print(f"Arquivo de saída:    {output_path}")
    print(f"Threshold:           {threshold:.2e}")
    print("-" * 70)
    print(f"Total de casos:      {total}")
    print(f"Casos mantidos:      {kept} ({100*kept/total:.2f}%)")
    print(f"Casos removidos:     {removed} ({100*removed/total:.2f}%)")

    if removed_by_n_elements:
        print("-" * 70)
        print("Casos removidos por n_elements:")
        for n_elem in sorted(removed_by_n_elements.keys()):
            count = removed_by_n_elements[n_elem]
            print(f"  n_elements={n_elem:3d}: {count} casos removidos")

    if errors:
        print("-" * 70)
        print("Top 10 maiores erros:")
        errors_sorted = sorted(errors, key=lambda x: x[2], reverse=True)[:10]
        for case_id, n_elem, erro in errors_sorted:
            print(f"  {case_id} (n_elem={n_elem}): erro = {erro:.6e}")

    print("=" * 70)


if __name__ == "__main__":
    main()