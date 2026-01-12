"""
Teste de gradientes: verificar se os gradientes fluem corretamente.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import build_edge_index
from src.modeling.gnn import BeamGNN


def test_gradients():
    print("=" * 60)
    print("TESTE: Fluxo de Gradientes")
    print("=" * 60)

    device = torch.device("mps")
    n_nodes = 105
    n_features = 12

    # Criar dados
    torch.manual_seed(42)
    x = torch.randn(n_nodes, n_features, device=device)
    y = torch.randn(n_nodes, device=device)
    edge_index = build_edge_index(n_nodes).to(device)

    # Criar modelo
    model = BeamGNN(
        input_dim=n_features,
        hidden_dim=64,
        output_dim=1,
        num_layers=4,
        dropout=0.0,
    ).to(device)

    criterion = nn.MSELoss()

    # Forward pass
    model.train()
    out = model(x, edge_index).squeeze()
    loss = criterion(out, y)

    print(f"\nLoss: {loss.item():.6f}")
    print(f"Output shape: {out.shape}")
    print(f"Output min: {out.min().item():.4f}")
    print(f"Output max: {out.max().item():.4f}")

    # Backward pass
    loss.backward()

    # Verificar gradientes de cada camada
    print("\n--- Gradientes por camada ---")

    all_grads_ok = True

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            grad_norm = grad.norm().item()
            grad_mean = grad.mean().item()
            grad_max = grad.abs().max().item()
            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()
            is_zero = grad_norm < 1e-10

            status = "OK"
            if has_nan:
                status = "NaN!"
                all_grads_ok = False
            elif has_inf:
                status = "Inf!"
                all_grads_ok = False
            elif is_zero:
                status = "ZERO!"
                all_grads_ok = False

            print(
                f"  {name:40s} | norm={grad_norm:.2e} | max={grad_max:.2e} | {status}"
            )
        else:
            print(f"  {name:40s} | grad=None | SEM GRADIENTE!")
            all_grads_ok = False

    print("\n" + "=" * 60)
    if all_grads_ok:
        print("RESULTADO: PASSOU")
        print("Gradientes estão fluindo corretamente.")
    else:
        print("RESULTADO: FALHOU")
        print("Há problemas com os gradientes!")
    print("=" * 60)

    return all_grads_ok


if __name__ == "__main__":
    test_gradients()
