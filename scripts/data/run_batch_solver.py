"""
Script OTIMIZADO para executar o solver VEM em batch (Streaming Mode).
Consumo de RAM constante e baixo, independente do número de samples.

Uso:
    uv run python3 scripts/dataset_generator/run_batch_solver.py --n_samples 1000 --num_workers 8
"""

import argparse
import csv
import subprocess
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.data.io import init_results_csv
from src.paths import paths

SOLVER_EXECUTABLE = paths.solver_vem / "bin" / "batch_solver"
PARAMS_DIR = paths.data.raw / "Sobol" / "params"
RESULTS_DIR = paths.data.raw / "Sobol" / "results"

N_ELEMENTS_LIST = [10, 20, 30, 40, 50]


def run_single_case_optimized(args: tuple) -> tuple | None:
    """
    Executa o solver.
    OTIMIZAÇÃO: Retorna strings cruas para evitar conversão desnecessária float/string.
    """
    case_id, L, q, E, I, n_elements, executable_str = args

    cmd = [
        executable_str,
        str(case_id),
        str(L),
        str(q),
        str(E),
        str(I),
        str(n_elements),
    ]

    try:
        # capture_output=True cria buffers em memória, mas como limpamos rápido, ok.
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return None

        # O output esperado é: case_id|n_elem|disp_str|rot_str
        parts = result.stdout.strip().split("|")
        if len(parts) != 4:
            return None

        # Truque de performance: Retornamos as strings originais de deslocamento/rotação.
        # Não convertemos para float. Isso economiza MUITA CPU e RAM.
        return (parts[0], parts[1], parts[2], parts[3])

    except Exception:
        return None


def task_generator(params_csv_path: Path, executable_str: str) -> Iterator[tuple]:
    """
    Gera tarefas lendo o CSV linha a linha (Lazy Loading).
    Nunca carrega o arquivo inteiro na RAM.
    """
    with open(params_csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            n_elements = N_ELEMENTS_LIST[i % len(N_ELEMENTS_LIST)]
            yield (
                row["case_id"],
                row["L"],
                row["q"],
                row["E"],
                row["I"],
                n_elements,
                executable_str,
            )


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, r = divmod(seconds, 3600)
        m, s = divmod(r, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"


def count_file_lines(filepath: Path) -> int:
    """Conta linhas rapidamente para barra de progresso."""
    with open(filepath, "rb") as f:
        return sum(1 for _ in f) - 1  # Remove header


def main():
    parser = argparse.ArgumentParser(
        description="Executa solver VEM em batch (Otimizado)"
    )
    parser.add_argument("--n_samples", type=int, default=10, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    params_csv = PARAMS_DIR / f"params_{args.n_samples}_samples.csv"
    if not params_csv.exists():
        print(f"Arquivo não encontrado: {params_csv}")
        return

    if not SOLVER_EXECUTABLE.exists():
        print(f"Executável não encontrado: {SOLVER_EXECUTABLE}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = RESULTS_DIR / f"results_{args.n_samples}_samples.csv"

    # Inicializa CSV (Header)
    init_results_csv(results_csv)

    print("Contando casos...")
    total = count_file_lines(params_csv)

    print(f"Casos: {total}")
    print(f"Workers: {args.num_workers}")
    print(f"Output: {results_csv}")
    print("-" * 40)

    start_time = time.time()
    completed = 0
    success = 0

    executable_str = str(SOLVER_EXECUTABLE)

    # Otimização de Chunksize para balancear carga x overhead
    chunksize = max(10, total // (args.num_workers * 50))

    # Abrimos o arquivo de saída UMA VEZ e mantemos aberto
    # Isso é 100x mais rápido que abrir/fechar a cada linha
    with open(results_csv, "a", newline="") as f_out:
        writer = csv.writer(f_out)

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # O executor consome o gerador sob demanda
            tasks = task_generator(params_csv, executable_str)

            for result in executor.map(
                run_single_case_optimized, tasks, chunksize=chunksize
            ):
                completed += 1

                if result is not None:
                    # result já é (case_id, n_elem, disp_str, rot_str)
                    writer.writerow(result)
                    success += 1

                if completed % 100 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    print(
                        f"\rProgresso: {completed}/{total} ({success} ok) | "
                        f"ETA: {format_time(eta)} | "
                        f"Taxa: {rate:.1f} casos/s",
                        end="",
                        flush=True,
                    )

    print("\n" + "-" * 40)
    elapsed = time.time() - start_time
    print(f"Tempo total: {format_time(elapsed)}")
    print(f"Casos processados: {success}/{total}")


if __name__ == "__main__":
    main()
