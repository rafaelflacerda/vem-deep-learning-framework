"""
Resolve um único caso de viga 1D usando o binding polivem_py.
Compara com solução analítica para viga em balanço com carga distribuída uniforme.

Uso:
  uv run --active python scripts/dataset_generator/solve_single_case.py --L 10.0 --q 5000.0 --E 30e9 --I 0.0001 --n_elements 21
"""

from __future__ import annotations

import argparse

import numpy as np
from polivem import polivem_py as pv


def analytical_cantilever(
    x: float, L: float, q: float, E: float, I: float
) -> tuple[float, float]:
    """
    Solução analítica para viga em balanço com carga distribuída uniforme.

    Args:
        x: Posição ao longo da viga (m), x=0 no engaste
        L: Comprimento da viga (m)
        q: Carga distribuída (N/m, positiva para baixo)
        E: Módulo de elasticidade (Pa)
        I: Momento de inércia (m^4)

    Returns:
        (displacement, slope): deslocamento vertical (m) e rotação (rad) em x

    Fórmulas:
        w(x) = -q*x²/(24*E*I) * (6*L² - 4*L*x + x²)
        θ(x) = -q*x/(6*E*I) * (3*L² - 3*L*x + x²)
    """
    displacement = -(q * x**2) / (24 * E * I) * (6 * L**2 - 4 * L * x + x**2)
    slope = -(q * x) / (6 * E * I) * (3 * L**2 - 3 * L * x + x**2)
    return displacement, slope


def solve_single_beam(
    L: float,
    q: float,
    E: float,
    I: float,
    n_elements: int = 21,
    model_order: int = 3,
) -> tuple[list[float], list[float], np.ndarray]:
    """
    Resolve uma viga 1D em balanço (cantilever) com carga distribuída uniforme.

    Args:
        L: Comprimento da viga (m)
        q: Carga distribuída (N/m, positiva para baixo)
        E: Módulo de elasticidade (Pa)
        I: Momento de inércia (m^4)
        n_elements: Número de elementos
        model_order: Ordem do modelo VEM

    Returns:
        (displacements, rotations, node_positions): listas com valores em cada nó e array com posições x
    """
    # Engaste no nó 0: desloc vertical (1) e rotação (1) fixos
    SUPP = np.array([[0, 1, 1, 0]], dtype=np.int32)

    # Criar malha
    beam = pv.mesh.Beam()
    beam.horizontal_bar_disc(float(L), int(n_elements))

    nodes = np.asarray(beam.nodes, dtype=np.float64)
    elements = np.asarray(beam.elements, dtype=np.int32)

    # Configurar solver
    solver = pv.solver.BeamSolver(nodes, elements, model_order)
    solver.setInertiaMoment(float(I))
    solver.setSupp(SUPP)

    # Carga distribuída: aplicamos negativa (para baixo)
    q_load = np.full((2, 1), -float(q), dtype=np.float64)
    solver.setDistributedLoad(q_load, elements)

    # Montagem do sistema
    K = solver.buildGlobalK(float(E))
    KII = solver.buildStaticCondensation(K, "KII")
    KIM = solver.buildStaticCondensation(K, "KIM")
    KMI = solver.buildStaticCondensation(K, "KMI")
    KMM = solver.buildStaticCondensation(K, "KMM")

    R = solver.buildGlobalDistributedLoad()
    RI = solver.buildStaticDistVector(R, "RI")
    RM = solver.buildStaticDistVector(R, "RM")

    # Condensação
    Kc = solver.condense_matrix(KII, KIM, KMI, KMM)
    Rc = solver.condense_vector(RI, RM, KIM, KMM)

    # Aplicar condições de contorno
    Kc = solver.applyDBCMatrix(Kc)
    Rc = solver.applyDBCVec(Rc)

    # Converter para NumPy e resolver
    Kc = np.asarray(Kc, dtype=np.float64)
    Rc = np.asarray(Rc, dtype=np.float64).reshape(-1)

    u = np.linalg.solve(Kc, Rc).reshape(-1)

    # Separar deslocamentos (índices pares) e rotações (índices ímpares)
    displacements = u[0::2].tolist()
    rotations = u[1::2].tolist()

    # Posições dos nós (assumindo que nodes tem shape (n_nodes, 2) com [x, y])
    node_positions = nodes[:, 0]

    return displacements, rotations, node_positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve um único caso de viga 1D")
    parser.add_argument("--L", type=float, default=5.0, help="Comprimento da viga (m)")
    parser.add_argument(
        "--q", type=float, default=-1000.0, help="Carga distribuída (N/m)"
    )
    parser.add_argument(
        "--E", type=float, default=2.1e11, help="Módulo de elasticidade (Pa)"
    )
    parser.add_argument(
        "--I", type=float, default=0.000001, help="Momento de inércia (m^4)"
    )
    parser.add_argument(
        "--n_elements", type=int, default=79, help="Número de elementos"
    )

    args = parser.parse_args()

    print("Resolvendo viga em balanço:")
    print(f"  L = {args.L} m")
    print(f"  q = {args.q} N/m")
    print(f"  E = {args.E:.2e} Pa")
    print(f"  I = {args.I:.2e} m^4")
    print(f"  n_elements = {args.n_elements}")
    print()

    displacements, rotations, node_positions = solve_single_beam(
        L=args.L,
        q=args.q,
        E=args.E,
        I=args.I,
        n_elements=args.n_elements,
    )

    # Calcular solução analítica no último nó (ponta livre, x=L)
    disp_analytical_max, slope_analytical_max = analytical_cantilever(
        x=args.L,
        L=args.L,
        q=args.q,
        E=args.E,
        I=args.I,
    )

    # Valores numéricos na ponta (último nó)
    disp_numerical_max = displacements[-1]
    slope_numerical_max = rotations[-1]

    print("=" * 80)
    print("SOLUÇÃO ANALÍTICA NA PONTA LIVRE (x = L):")
    print(f"  Deslocamento máximo: {disp_analytical_max:.6e} m")
    print(f"  Rotação máxima:      {slope_analytical_max:.6e} rad")
    print()
    print("SOLUÇÃO NUMÉRICA (VEM) NA PONTA LIVRE:")
    print(f"  Deslocamento máximo: {disp_numerical_max:.6e} m")
    print(f"  Rotação máxima:      {slope_numerical_max:.6e} rad")
    print()
    print("ERRO RELATIVO:")
    erro_disp = (
        abs((disp_numerical_max - disp_analytical_max) / disp_analytical_max) * 100
    )
    erro_slope = (
        abs((slope_numerical_max - slope_analytical_max) / slope_analytical_max) * 100
    )
    print(f"  Deslocamento: {erro_disp:.4f}%")
    print(f"  Rotação:      {erro_slope:.4f}%")
    print("=" * 80)
    print()

    print(f"Resultados por nó ({len(displacements)} nós):")
    print()
    print(
        "Nó |   x (m)   | Desloc VEM (m) | Desloc Analítico (m) | Erro (%) | Rot VEM (rad) | Rot Analítica (rad) | Erro (%)"
    )
    print("-" * 130)

    for i, (x, d_num, r_num) in enumerate(
        zip(node_positions, displacements, rotations, strict=False)
    ):
        d_ana, r_ana = analytical_cantilever(x, args.L, args.q, args.E, args.I)

        # Evitar divisão por zero no engaste (onde d_ana = r_ana = 0)
        if abs(d_ana) > 1e-15:
            err_d = abs((d_num - d_ana) / d_ana) * 100
        else:
            err_d = 0.0 if abs(d_num) < 1e-15 else float("inf")

        if abs(r_ana) > 1e-15:
            err_r = abs((r_num - r_ana) / r_ana) * 100
        else:
            err_r = 0.0 if abs(r_num) < 1e-15 else float("inf")

        print(
            f"{i:2d} | {x:9.4f} | {d_num:14.6e} | {d_ana:20.6e} | {err_d:8.4f} | "
            f"{r_num:13.6e} | {r_ana:19.6e} | {err_r:8.4f}"
        )


if __name__ == "__main__":
    main()
