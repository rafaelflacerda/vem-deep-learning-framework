"""
Teste com target sintético: verificar se a rede consegue aprender
uma relação conhecida entre features e target.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import build_edge_index
from src.modeling.gnn import BeamGNN
from src.paths import paths


def test_synthetic_target():
    print("=" * 60)
    print("TESTE: Aprendizado de Target Sintético")
    print("=" * 60)

    device = torch.device("mps")

    # Carregar dataset real (para usar as features reais)
    dataset_path = paths.data.processed / "dataset_1000.pt"
    data = torch.load(dataset_path, weights_only=False)

    features = data["features"]  # (n_samples, n_nodes, n_features)
    n_samples, n_nodes, n_features = features.shape

    print(f"\nDataset: {n_samples} amostras, {n_nodes} nós, {n_features} features")

    # Criar target sintético: combinação linear simples das features
    # y = sum(features) / n_features
    # Isso é algo que a rede DEVERIA conseguir aprender facilmente
    synthetic_targets = features.mean(dim=-1)  # (n_samples, n_nodes)

    # Normalizar
    features_mean = features.mean(dim=(0, 1), keepdim=True)
    features_std = features.std(dim=(0, 1), keepdim=True) + 1e-8
    features_norm = (features - features_mean) / features_std

    targets_mean = synthetic_targets.mean()
    targets_std = synthetic_targets.std() + 1e-8
    targets_norm = (synthetic_targets - targets_mean) / targets_std

    print(
        f"Features normalizadas: mean={features_norm.mean():.4f}, std={features_norm.std():.4f}"
    )
    print(
        f"Targets normalizados: mean={targets_norm.mean():.4f}, std={targets_norm.std():.4f}"
    )

    # Criar dataset PyTorch Geometric
    edge_index = build_edge_index(n_nodes)

    dataset = []
    for i in range(n_samples):
        dataset.append(
            Data(
                x=features_norm[i],
                edge_index=edge_index,
                y=targets_norm[i],
            )
        )

    # Split
    train_size = int(0.8 * n_samples)
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"\nTreino: {len(train_dataset)}, Validação: {len(val_dataset)}")

    # Modelo
    model = BeamGNN(
        input_dim=n_features,
        hidden_dim=64,
        output_dim=1,
        num_layers=4,
        dropout=0.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Treinar
    print("\nTreinando...")
    print("-" * 40)

    for epoch in range(50):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index).squeeze()
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validação
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index).squeeze()
                loss = criterion(out, batch.y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Época {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    # Calcular R²
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index).squeeze()
            all_preds.append(out.cpu())
            all_targets.append(batch.y.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    print("-" * 40)
    print(f"R² final: {r2:.4f}")

    print("\n" + "=" * 60)
    if r2 > 0.9:
        print("RESULTADO: PASSOU")
        print("A rede consegue aprender relações simples.")
        print("Problema provável: a relação features→deslocamento é mais complexa")
        print(
            "que a rede atual consegue capturar, ou os hiperparâmetros precisam ajuste."
        )
    elif r2 > 0.5:
        print("RESULTADO: PARCIAL")
        print("A rede aprende parcialmente.")
        print("Pode haver problemas sutis no código ou nos hiperparâmetros.")
    else:
        print("RESULTADO: FALHOU")
        print("A rede NÃO consegue aprender nem relações simples.")
        print("Há provavelmente um BUG no código.")
    print("=" * 60)

    return r2 > 0.9


if __name__ == "__main__":
    test_synthetic_target()
