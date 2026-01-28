"""
Geração de amostras de parâmetros para o solver VEM usando Sobol sampling.

Os parâmetros E e I são amostrados em escala logarítmica (variam em ordens de magnitude).
O parâmetro L é amostrado em escala linear.

Versão simplificada: E e q são fixos, apenas I e L são amostrados.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats.qmc import Sobol

# Usa o gerenciador central de paths do projeto
from src.paths import ensure_dir, paths

# =============================================================================
# CONFIGURAÇÃO - Modifique aqui conforme necessário
# =============================================================================

# Número de amostras válidas desejadas (após o filtro)
N_TARGET = 2500

# Tamanho do bloco Sobol (deve ser potência de 2)
BLOCK_SIZE = 1024

# Limite máximo de tentativas para evitar loop infinito
MAX_ITERATIONS = 25000

# Ranges dos parâmetros
# I: definido em escala log10
# L: definido em escala linear
PARAM_RANGES = {
    "I_log10": (-5.0, -3.0),  # 10^-6 a 10^-3 m^4
    "L": (1.0, 8.0),  # 1 a 10 m
}

# Critério de pequenos deslocamentos: rho = w_max/L < RHO_MAX
RHO_MAX = 0.02

# Valores fixos de E e q
E_FIXED = 200e9  # 200 GPa (aço)
Q_FIXED = 5000.0  # 5 kN/m

# Seed para reprodutibilidade (None para não fixar)
SEED = 42


# =============================================================================
# FUNÇÕES
# =============================================================================


def transform_i_l(samples: np.ndarray, ranges: dict) -> np.ndarray:
    """
    Transforma amostras de [0,1]^2 para os ranges físicos de I, L.

    Colunas de retorno: [I, L]
    """
    n = samples.shape[0]
    out = np.zeros((n, 2), dtype=np.float64)

    # I: escala log
    I_log_min, I_log_max = ranges["I_log10"]
    I_log = I_log_min + samples[:, 0] * (I_log_max - I_log_min)
    out[:, 0] = 10**I_log

    # L: escala linear
    L_min, L_max = ranges["L"]
    out[:, 1] = L_min + samples[:, 1] * (L_max - L_min)

    return out


def compute_rho(E: float, I: np.ndarray, L: np.ndarray, q: float) -> np.ndarray:
    """Calcula rho = q*L^3/(8*E*I) vetorizado."""
    return q * (L**3) / (8.0 * E * I)


def generate_valid_samples(
    n_target: int,
    ranges: dict,
    rho_max: float,
    E_fixed: float,
    q_fixed: float,
    block_size: int,
    max_iterations: int,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Gera amostras Sobol em blocos até atingir n_target amostras válidas.

    Com E e q fixos, amostramos apenas (I, L) e filtramos por rho <= rho_max.

    Retorna:
        - Array (n_target, 4) com colunas [E, I, L, q]
        - Dicionário com estatísticas da geração
    """
    sampler = Sobol(d=2, scramble=True, seed=seed)

    valid_samples: list[np.ndarray] = []
    total_generated = 0
    total_rejected = 0
    iteration = 0

    while len(valid_samples) < n_target and iteration < max_iterations:
        raw_block = sampler.random(block_size)
        total_generated += block_size

        # Transforma para I e L
        il = transform_i_l(raw_block, ranges)
        I = il[:, 0]
        L = il[:, 1]

        # Calcula rho para cada amostra
        rho = compute_rho(E_fixed, I, L, q_fixed)

        # Filtra amostras que satisfazem o critério de pequenos deslocamentos
        mask = rho <= rho_max
        rejected_in_block = int((~mask).sum())
        total_rejected += rejected_in_block

        if mask.any():
            I_valid = I[mask]
            L_valid = L[mask]

            # Monta amostras [E, I, L, q] com E e q fixos
            for i in range(len(I_valid)):
                if len(valid_samples) < n_target:
                    sample = np.array([E_fixed, I_valid[i], L_valid[i], q_fixed])
                    valid_samples.append(sample)

        iteration += 1

    stats = {
        "target": n_target,
        "generated": total_generated,
        "rejected": total_rejected,
        "acceptance_rate": (
            (total_generated - total_rejected) / total_generated
            if total_generated > 0
            else 0.0
        ),
        "iterations": iteration,
        "rho_max": rho_max,
        "E_fixed": E_fixed,
        "q_fixed": q_fixed,
    }

    return np.array(valid_samples), stats


def save_samples(samples: np.ndarray, output_dir: Path, n_samples: int) -> None:
    """Salva todas as amostras em um único arquivo JSON."""
    ensure_dir(output_dir)

    all_cases = {}
    for i, (E, I, L, q) in enumerate(samples):
        case_key = f"case_{i:06d}"
        all_cases[case_key] = {
            "L": float(L),
            "q": float(q),
            "E": float(E),
            "I": float(I),
        }

    filepath = output_dir / f"params_{n_samples}_samples.json"
    with open(filepath, "w") as f:
        json.dump(all_cases, f, indent=2)


def main():
    print(f"Gerando {N_TARGET} amostras válidas via Sobol...")
    print(f"E fixo: {E_FIXED:.2e} Pa")
    print(f"q fixo: {Q_FIXED:.2e} N/m")
    print(f"Critério: rho < {RHO_MAX}")
    print()

    samples, stats = generate_valid_samples(
        n_target=N_TARGET,
        ranges=PARAM_RANGES,
        rho_max=RHO_MAX,
        E_fixed=E_FIXED,
        q_fixed=Q_FIXED,
        block_size=BLOCK_SIZE,
        max_iterations=MAX_ITERATIONS,
        seed=SEED,
    )

    print(f"Amostras geradas (tentadas): {stats['generated']}")
    print(f"Amostras rejeitadas (rho > {RHO_MAX}): {stats['rejected']}")
    print(f"Taxa de aceitação: {stats['acceptance_rate']:.1%}")
    print(f"Iterações necessárias: {stats['iterations']}")
    print(f"Amostras válidas obtidas: {len(samples)}")
    print()

    if len(samples) < N_TARGET:
        print(f"AVISO: Não foi possível atingir N_TARGET={N_TARGET} amostras.")
        print("Considere ajustar os ranges ou aumentar MAX_ITERATIONS.")
        print()

    # Diretório de saída
    output_dir = paths.data.raw / "Sobol" / "params" / "rho_0.010_E_q_fixed"

    print(f"Salvando em {output_dir}/ ...")
    save_samples(samples, output_dir, len(samples))

    print("Concluído.")


if __name__ == "__main__":
    main()
