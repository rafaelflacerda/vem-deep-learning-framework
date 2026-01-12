"""
Gera resultados do solver VEM (viga 1D) usando o binding (polivem_py),
lendo params_<_n_samples_>_samples.csv e escrevendo results_<_n_samples_>_samples.csv.

Formato do CSV de entrada:
case_id,L,q,E,I

Formato do CSV de saída:
case_id,n_elements,displacements,rotations

Uso:
  uv run --active python scripts/dataset_generator/run_batch_solver_binding_csv.py --n_samples 1000 --num_workers 8
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.paths import paths

# --- Paths (sem hardcode) ---
PARAMS_DIR = paths.data.raw / "Sobol" / "params" / "rho_0.050"
RESULTS_DIR = paths.data.raw / "Sobol" / "results" / "rho_0.050"

# --- Config editável ---
N_ELEMENTS_LIST = [21]  # round-robin
MODEL_ORDER = 3  # fixo

# Cantilever (engaste no nó 0), mesmo padrão do C++
SUPP = np.array([[0, 1, 1, 0]], dtype=np.int32)

# Binding carregado 1x por worker
_pv = None


def _init_worker() -> None:
    global _pv
    from polivem import polivem_py as pv

    _pv = pv


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"


def _read_params_csv(params_path: Path) -> list[tuple[str, float, float, float, float]]:
    """
    Retorna lista de (case_id, L, q, E, I) na ordem do arquivo.
    """
    rows: list[tuple[str, float, float, float, float]] = []
    with open(params_path, newline="") as f:
        reader = csv.DictReader(f)
        expected = {"case_id", "L", "q", "E", "I"}
        missing = expected - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV {params_path} está sem colunas: {sorted(missing)}")

        for r in reader:
            case_id = r["case_id"].strip()
            L = float(r["L"])
            q = float(r["q"])
            E = float(r["E"])
            I = float(r["I"])
            rows.append((case_id, L, q, E, I))
    return rows


def _solve_case(
    case_id: str, L: float, q_val: float, E: float, I: float, n_elements: int
):
    """
    Resolve um caso usando binding e retorna (case_id, n_elements, displacements:list, rotations:list)
    """
    pv = _pv
    if pv is None:
        raise RuntimeError("Binding não inicializado. Use initializer=_init_worker.")

    # Mesh
    beam = pv.mesh.Beam()
    beam.horizontal_bar_disc(float(L), int(n_elements))

    nodes = np.asarray(beam.nodes, dtype=np.float64)
    elements = np.asarray(beam.elements, dtype=np.int32)

    # Solver
    solver = pv.solver.BeamSolver(nodes, elements, MODEL_ORDER)
    solver.setInertiaMoment(float(I))
    solver.setSupp(SUPP)

    # Carga distribuída: no CSV é positiva; aplicamos negativa (para baixo)
    q = np.full((2, 1), -float(q_val), dtype=np.float64)
    solver.setDistributedLoad(q, elements)

    # Montagem + condensação
    K = solver.buildGlobalK(float(E))
    KII = solver.buildStaticCondensation(K, "KII")
    KIM = solver.buildStaticCondensation(K, "KIM")
    KMI = solver.buildStaticCondensation(K, "KMI")
    KMM = solver.buildStaticCondensation(K, "KMM")

    R = solver.buildGlobalDistributedLoad()
    RI = solver.buildStaticDistVector(R, "RI")
    RM = solver.buildStaticDistVector(R, "RM")

    Kc = solver.condense_matrix(KII, KIM, KMI, KMM)
    Rc = solver.condense_vector(RI, RM, KIM, KMM)

    Kc = solver.applyDBCMatrix(Kc)
    Rc = solver.applyDBCVec(Rc)

    Kc = np.asarray(Kc, dtype=np.float64)
    Rc = np.asarray(Rc, dtype=np.float64).reshape(-1)

    u = np.linalg.solve(Kc, Rc).reshape(-1)

    displacements = u[0::2].tolist()
    rotations = u[1::2].tolist()
    return case_id, int(n_elements), displacements, rotations


def _worker_entry(task: tuple[str, float, float, float, float, int]):
    case_id, L, q, E, I, n_elements = task
    try:
        out = _solve_case(case_id, L, q, E, I, n_elements)
        return out, None
    except Exception as e:
        return None, f"{case_id}\t{type(e).__name__}: {e}"


def _write_results_csv_streaming(
    out_csv: Path,
    failures_log: Path,
    results_iter: Iterable[
        tuple[tuple[str, int, list[float], list[float]] | None, str | None]
    ],
    total: int,
) -> tuple[int, int]:
    """
    Escreve CSV linha a linha e um log de falhas.
    Retorna (success, failed).
    """
    tmp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")

    success = 0
    failed = 0

    with open(tmp_csv, "w", newline="") as f_out, open(failures_log, "w") as f_fail:
        writer = csv.writer(f_out)
        writer.writerow(["case_id", "n_elements", "displacements", "rotations"])

        start = time.time()
        completed = 0

        for result, err in results_iter:
            completed += 1

            if err is not None:
                failed += 1
                f_fail.write(err + "\n")
            else:
                success += 1
                case_id, n_elem, disps, rots = result

                disp_str = ";".join(f"{x:.17g}" for x in disps)
                rot_str = ";".join(f"{x:.17g}" for x in rots)

                writer.writerow([case_id, n_elem, disp_str, rot_str])

            if completed % 100 == 0 or completed == total:
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (total - completed) / rate if rate > 0 else 0.0
                print(
                    f"\rProgresso: {completed}/{total} | ok={success} fail={failed} | "
                    f"Tempo: {_format_time(elapsed)} | ETA: {_format_time(eta)} | {rate:.1f} casos/s",
                    end="",
                    flush=True,
                )

    tmp_csv.replace(out_csv)
    print()
    return success, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera dataset VEM 1D via binding (CSV→CSV)"
    )
    parser.add_argument("--n_samples", type=int, required=True, default=100)
    parser.add_argument(
        "--num_workers", type=int, default=max(1, (os.cpu_count() or 8) - 2)
    )
    parser.add_argument("--overwrite", action="store_true", default=True)
    args = parser.parse_args()

    params_csv = PARAMS_DIR / f"params_{args.n_samples}_samples.csv"
    out_csv = RESULTS_DIR / f"results_{args.n_samples}_samples.csv"
    failures_log = RESULTS_DIR / f"results_{args.n_samples}_samples.failures.log"

    if not params_csv.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {params_csv}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Resultado já existe: {out_csv} (use --overwrite)")

    rows = _read_params_csv(params_csv)
    total = len(rows)
    if total == 0:
        raise ValueError(f"CSV vazio: {params_csv}")

    # Round-robin de n_elements
    tasks: list[tuple[str, float, float, float, float, int]] = []
    for i, (case_id, L, q, E, I) in enumerate(rows):
        n_elements = N_ELEMENTS_LIST[i % len(N_ELEMENTS_LIST)]
        tasks.append((case_id, L, q, E, I, int(n_elements)))

    print(f"Params: {params_csv}")
    print(f"Out:    {out_csv}")
    print(
        f"Casos: {total} | Workers: {args.num_workers} | n_elements: {N_ELEMENTS_LIST} (round-robin)"
    )
    print("-" * 90)

    chunksize = max(1, total // (args.num_workers * 10))

    def results_generator():
        with ProcessPoolExecutor(
            max_workers=args.num_workers, initializer=_init_worker
        ) as ex:
            for item in ex.map(_worker_entry, tasks, chunksize=chunksize):
                yield item

    ok, fail = _write_results_csv_streaming(
        out_csv, failures_log, results_generator(), total
    )

    print("-" * 90)
    print(f"Concluído: ok={ok} fail={fail}")
    print(f"CSV: {out_csv}")
    print(f"Falhas: {failures_log}")


if __name__ == "__main__":
    main()
