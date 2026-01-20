"""
Geração de amostras de parâmetros para o solver VEM usando Sobol sampling.

Os parâmetros E e I são amostrados em escala logarítmica (variam em ordens de magnitude).
O parâmetro L é amostrado em escala linear.

Mudança principal:
- Em vez de amostrar q e depois rejeitar por rho, agora amostramos (E, I, L) e
  calculamos q_max_case a partir do limite rho_max, e então amostramos q (log-uniforme)
  dentro do intervalo [q_min, min(q_max_global, q_max_case)].
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
# E e I: definidos em escala log10 (ex: [9, 11] significa 10^9 a 10^11 Pa)
# L: definido em escala linear
# q: definido em escala log10
PARAM_RANGES = {
    #"E_log10": (10.84, 11.32),  # Alumínio (70 GPa) até aço (210 GPa)
    "I_log10": (-6.0, -3.0),  # 10^-6 a 10^-3 m^4
    "L": (1.0, 10.0),  # 1 a 10 m
    "q_log10": (2.0, 5.0),  # 10^2 a 10^5 N/m
}

# Critério de pequenos deslocamentos: rho = w_max/L < RHO_MAX
RHO_MAX = 0.01

E_fixed = 200e9

# Seed para reprodutibilidade (None para não fixar)
SEED = 42


# =============================================================================
# FUNÇÕES
# =============================================================================


def transform_e_i_l(samples: np.ndarray, ranges: dict) -> np.ndarray:
    """
    Transforma amostras de [0,1]^3 para os ranges físicos de E, I, L.

    Colunas de retorno: [E, I, L]
    """
    n = samples.shape[0]
    out = np.zeros((n, 2), dtype=np.float64)

    # E: escala log
    # E_log_min, E_log_max = ranges["E_log10"]
    # E_log = E_log_min + samples[:, 0] * (E_log_max - E_log_min)
    # out[:, 0] = 10**E_log

    # I: escala log
    I_log_min, I_log_max = ranges["I_log10"]
    I_log = I_log_min + samples[:, 0] * (I_log_max - I_log_min)
    out[:, 0] = 10**I_log

    # L: escala linear
    L_min, L_max = ranges["L"]
    out[:, 1] = L_min + samples[:, 1] * (L_max - L_min)

    return out


def sample_q_log_uniform(u: np.ndarray, q_min: float, q_max: np.ndarray) -> np.ndarray:
    """
    Amostra q com distribuição log-uniforme no intervalo [q_min, q_max_i] para cada caso i.

    u: vetor em [0,1] (um por amostra)
    q_max: vetor (N,) com máximo permitido por caso
    """
    log_qmin = np.log10(q_min)
    log_qmax = np.log10(q_max)
    return 10 ** (log_qmin + u * (log_qmax - log_qmin))


def compute_rho_from_components(
    E: np.ndarray, I: np.ndarray, L: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Calcula rho = q*L^3/(8*E*I) vetorizado."""
    return q * (L**3) / (8.0 * E * I)


def generate_valid_samples(
    n_target: int,
    ranges: dict,
    rho_max: float,
    block_size: int,
    max_iterations: int,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Gera amostras Sobol em blocos até atingir n_target amostras válidas.

    Estratégia nova (sem rejection por rho):
      1) Amostra (E, I, L) via Sobol
      2) Calcula q_max_case = 8*rho_max*E*I/L^3
      3) Define q_max_effective = min(q_max_global, q_max_case)
      4) Só mantém casos com q_max_effective > q_min
      5) Amostra q log-uniforme em [q_min, q_max_effective] usando um u~U(0,1)

    Retorna:
        - Array (n_target, 4) com colunas [E, I, L, q]
        - Dicionário com estatísticas da geração
    """
    sampler = Sobol(d=3, scramble=True, seed=seed)

    q_log_min, q_log_max = ranges["q_log10"]
    q_min = 10**q_log_min
    q_max_global = 10**q_log_max

    valid_samples: list[np.ndarray] = []
    total_generated = 0
    total_rejected = 0
    iteration = 0

    while len(valid_samples) < n_target and iteration < max_iterations:
        raw_block = sampler.random(block_size)
        total_generated += block_size

        # Usa 3 dimensões para (E, I, L) e a 4ª como "u" para amostrar q
        eil = transform_e_i_l(raw_block[:, :2], ranges)
        #E = eil[:, 0]
        I = eil[:, 0]
        L = eil[:, 1]
        u = raw_block[:, 2]

        # Limite por rho: q < 8*rho_max*E*I/L^3
        q_max_case = 8.0 * rho_max * E_fixed * I / (L**3)

        # Limite final: não pode passar do q_max_global
        q_max_effective = np.minimum(q_max_global, q_max_case)

        # Mantém apenas casos com intervalo viável
        mask = q_max_effective > q_min
        rejected_in_block = int((~mask).sum())
        total_rejected += rejected_in_block

        if mask.any():
            q = sample_q_log_uniform(u[mask], q_min=q_min, q_max=q_max_effective[mask])

            # Monta amostras [E, I, L, q]
            E_array = np.full(len(I[mask]), E_fixed)
            samples_block = np.column_stack([E_array, I[mask], L[mask], q])

            # (Opcional, mas seguro): checa numericamente rho <= rho_max (deve sempre passar)
            # Se quiser manter 100% "hard", dá pra remover essa checagem.
            rho = compute_rho_from_components(
                samples_block[:, 0],
                samples_block[:, 1],
                samples_block[:, 2],
                samples_block[:, 3],
            )
            safe_mask = rho <= rho_max * (1.0 + 1e-12)  # tolerância numérica
            samples_block = samples_block[safe_mask]

            for sample in samples_block:
                if len(valid_samples) < n_target:
                    valid_samples.append(sample)

        iteration += 1

    stats = {
        "target": n_target,
        "generated": total_generated,
        "rejected": total_rejected,  # rejeitados por "intervalo inviável" (q_max <= q_min)
        "acceptance_rate": (
            (total_generated - total_rejected) / total_generated
            if total_generated > 0
            else 0.0
        ),
        "iterations": iteration,
        "rho_max": rho_max,
        "q_min": q_min,
        "q_max_global": q_max_global,
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
    print(f"Critério: rho < {RHO_MAX} (via q_max_case)")
    print()

    samples, stats = generate_valid_samples(
        n_target=N_TARGET,
        ranges=PARAM_RANGES,
        rho_max=RHO_MAX,
        block_size=BLOCK_SIZE,
        max_iterations=MAX_ITERATIONS,
        seed=SEED,
    )

    print(f"Amostras geradas (tentadas): {stats['generated']}")
    print(f"Amostras rejeitadas (q_max_case <= q_min): {stats['rejected']}")
    print(f"Taxa de aceitação: {stats['acceptance_rate']:.1%}")
    print(f"Iterações necessárias: {stats['iterations']}")
    print(f"Amostras válidas obtidas: {len(samples)}")
    print()

    if len(samples) < N_TARGET:
        print(f"AVISO: Não foi possível atingir N_TARGET={N_TARGET} amostras.")
        print("Considere ajustar os ranges ou aumentar MAX_ITERATIONS.")
        print()

    # Agora usando o seu gerenciador de paths
    if RHO_MAX == 0.05:
        output_dir = paths.data.raw / "Sobol" / "params" / "rho_0.050"
    elif RHO_MAX == 0.01:
        output_dir = paths.data.raw / "Sobol" / "params" / "rho_0.010_E_fixed_102_elements"
    elif RHO_MAX == 0.025:
        output_dir = paths.data.raw / "Sobol" / "params" / "rho_0.025"

    print(f"Salvando em {output_dir}/ ...")
    save_samples(samples, output_dir, len(samples))

    print("Concluído.")


if __name__ == "__main__":
    main()
