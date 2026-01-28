"""
Geração de amostras de parâmetros para o solver VEM usando Sobol sampling.

ALTERAÇÃO: Agora a amostragem é baseada na RAZÃO DE ESBELTEZ (lambda = L/h)
e no Comprimento (L), gerando o Momento de Inércia (I) de forma derivada.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats.qmc import Sobol

# Usa o gerenciador central de paths do projeto (mantenha o seu import original)
try:
    from src.paths import ensure_dir, paths
except ImportError:
    # Fallback para rodar standalone se não tiver a estrutura de pastas
    def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
    class paths:
        class data:
            raw = Path("./data/raw")

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

N_TARGET = 2500
BLOCK_SIZE = 1024
MAX_ITERATIONS = 25000

# NOVOS RANGES
# lambda (esbeltez): L/h. Para balanços, valores típicos vão de 5 a 15 (concreto/aço).
# Colocamos 8 a 20 para explorar um espaço amplo, mas realista.
PARAM_RANGES = {
    "lambda": (8.0, 20.0),  # L/h (adimensional)
    "L": (1.0, 8.0),        # Metros
}

RHO_MAX = 0.02
E_FIXED = 200e9   # Aço
Q_FIXED = 5000.0  # 5 kN/m
SEED = 42

# =============================================================================
# FUNÇÕES
# =============================================================================

def transform_params(samples: np.ndarray, ranges: dict) -> np.ndarray:
    """
    Transforma amostras do hipercubo unitário [0,1]^2 para [I, L]
    usando a lógica da Razão de Esbeltez.
    
    Input samples[:, 0] -> lambda (esbeltez)
    Input samples[:, 1] -> L (comprimento)
    
    Retorna: colunas [I, L] para manter compatibilidade com o resto do código.
    """
    n = samples.shape[0]
    out = np.zeros((n, 2), dtype=np.float64)

    # 1. Recuperar Lambda (L/h) e L dos ranges
    lam_min, lam_max = ranges["lambda"]
    l_min, l_max = ranges["L"]

    lam_vals = lam_min + samples[:, 0] * (lam_max - lam_min)
    L_vals   = l_min + samples[:, 1] * (l_max - l_min)

    # 2. Calcular a altura h baseada na esbeltez
    # lambda = L / h  =>  h = L / lambda
    h_vals = L_vals / lam_vals

    # 3. Calcular Inércia (I) baseada na geometria da seção
    # AQUI VOCÊ DEFINE A FORMA.
    # Exemplo atual: Seção Quadrada Maciça (b = h)
    # I = (b * h^3) / 12 = h^4 / 12
    I_vals = (h_vals ** 4) / 12.0
    
    # Se fosse um perfil I de aço aproximado, poderia ser algo como:
    # I_vals = 0.5 * (h_vals**3) * 0.01 (apenas exemplo empírico)

    # Preenche o array de saída [I, L]
    out[:, 0] = I_vals
    out[:, 1] = L_vals

    return out


def compute_rho(E: float, I: np.ndarray, L: np.ndarray, q: float) -> np.ndarray:
    """Calcula rho = q*L^3/(8*E*I) vetorizado (Para Viga Engastada-Livre)."""
    # Nota: O coeficiente '8' no denominador é para a flecha máxima no balanço (qL^4/8EI)
    # Se a fórmula original usava '8' para rho, mantemos. 
    # (Rho geralmente é w_max/L, então w_max = qL^4/8EI => w/L = qL^3/8EI). Correto.
    return q * (L**3) / (8.0 * E * I)


def generate_valid_samples(
    n_target: int, ranges: dict, rho_max: float,
    E_fixed: float, q_fixed: float, block_size: int,
    max_iterations: int, seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    
    sampler = Sobol(d=2, scramble=True, seed=seed)
    valid_samples: list[np.ndarray] = []
    total_generated = 0
    total_rejected = 0
    iteration = 0

    while len(valid_samples) < n_target and iteration < max_iterations:
        raw_block = sampler.random(block_size)
        total_generated += block_size

        # --- AQUI É A MUDANÇA PRINCIPAL NA CHAMADA ---
        # Transforma os números aleatórios em I e L físicos usando a nova lógica
        il = transform_params(raw_block, ranges)
        I = il[:, 0]
        L = il[:, 1]
        # ---------------------------------------------

        rho = compute_rho(E_fixed, I, L, q_fixed)
        mask = rho <= rho_max
        
        rejected_in_block = int((~mask).sum())
        total_rejected += rejected_in_block

        if mask.any():
            I_valid = I[mask]
            L_valid = L[mask]
            for i in range(len(I_valid)):
                if len(valid_samples) < n_target:
                    sample = np.array([E_fixed, I_valid[i], L_valid[i], q_fixed])
                    valid_samples.append(sample)
        
        iteration += 1

    stats = {
        "target": n_target,
        "generated": total_generated,
        "rejected": total_rejected,
        "acceptance_rate": ((total_generated - total_rejected) / total_generated) if total_generated > 0 else 0.0,
        "iterations": iteration
    }
    return np.array(valid_samples), stats

# =============================================================================
# MANTER O RESTO IGUAL (save_samples e main)
# =============================================================================

def save_samples(samples: np.ndarray, output_dir: Path, n_samples: int) -> None:
    ensure_dir(output_dir)
    all_cases = {}
    for i, (E, I, L, q) in enumerate(samples):
        case_key = f"case_{i:06d}"
        all_cases[case_key] = {"L": float(L), "q": float(q), "E": float(E), "I": float(I)}

    filepath = output_dir / f"params_{n_samples}_samples.json"
    with open(filepath, "w") as f:
        json.dump(all_cases, f, indent=2)

def main():
    print(f"Gerando {N_TARGET} amostras (Baseado em Esbeltez L/h)...")
    print(f"Ranges: {PARAM_RANGES}")
    
    samples, stats = generate_valid_samples(
        n_target=N_TARGET, ranges=PARAM_RANGES, rho_max=RHO_MAX,
        E_fixed=E_FIXED, q_fixed=Q_FIXED, block_size=BLOCK_SIZE,
        max_iterations=MAX_ITERATIONS, seed=SEED,
    )

    print(f"Taxa de aceitação: {stats['acceptance_rate']:.1%}")
    print(f"Amostras geradas: {len(samples)}")

    # Ajuste o caminho de saída para indicar que é o método novo
    output_dir = paths.data.raw / "Sobol" / "params" / "slenderness_method"
    print(f"Salvando em {output_dir}")
    save_samples(samples, output_dir, len(samples))

if __name__ == "__main__":
    main()