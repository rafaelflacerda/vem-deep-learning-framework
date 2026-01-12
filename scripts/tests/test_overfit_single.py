"""
Teste de sanidade: verificar se a GNN consegue memorizar uma única amostra.

Se este teste falhar, há um bug no código.
Se passar, o problema provavelmente está nos hiperparâmetros.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import build_edge_index
from src.modeling.gnn import BeamGNN


def test_overfit_single_sample():
    print("=" * 60)
    print("TESTE: Overfitting em uma única amostra")
    print("=" * 60)

    # Configuração
    device = torch.device("mps")
    n_nodes = 105
    n_features = 12

    # Criar dados sintéticos simples (mais fácil de memorizar)
    torch.manual_seed(42)
    x = torch.randn(n_nodes, n_features)
    y = torch.randn(n_nodes)  # Target aleatório
    edge_index = build_edge_index(n_nodes)

    # Mover para device
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)

    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Device: {device}")

    # Criar modelo
    model = BeamGNN(
        input_dim=n_features,
        hidden_dim=64,
        output_dim=1,
        num_layers=4,
        dropout=0.0,  # Sem dropout para este teste
    ).to(device)

    # Optimizer com learning rate alto (queremos memorizar rápido)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print("\nTreinando por 500 iterações...")
    print("-" * 40)

    model.train()
    losses = []

    for i in range(500):
        optimizer.zero_grad()
        out = model(x, edge_index).squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (i + 1) % 100 == 0:
            print(f"Iteração {i+1:4d} | Loss: {loss.item():.6f}")

    # Avaliação final
    model.eval()
    with torch.no_grad():
        final_pred = model(x, edge_index).squeeze()
        final_loss = criterion(final_pred, y).item()

    print("-" * 40)
    print(f"Loss inicial: {losses[0]:.6f}")
    print(f"Loss final:   {final_loss:.6f}")
    print(f"Redução:      {(1 - final_loss/losses[0])*100:.1f}%")

    # Verificar se conseguiu memorizar
    print("\n" + "=" * 60)
    if final_loss < 0.01:
        print("RESULTADO: PASSOU")
        print("A rede conseguiu memorizar a amostra.")
        print("O código provavelmente está correto.")
        print("Problema provável: hiperparâmetros.")
    elif final_loss < 0.1:
        print("RESULTADO: PARCIAL")
        print("A rede reduziu a loss mas não memorizou completamente.")
        print("Pode ser problema de capacidade ou learning rate.")
    else:
        print("RESULTADO: FALHOU")
        print("A rede NÃO conseguiu memorizar uma única amostra.")
        print("Há provavelmente um BUG no código.")
    print("=" * 60)

    return final_loss < 0.01


if __name__ == "__main__":
    test_overfit_single_sample()
