"""
Teste do pipeline de dados: verificar se os dados estão corretos.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import BeamGraphDataset
from src.paths import paths


def test_data_pipeline():
    print("=" * 60)
    print("TESTE: Pipeline de Dados")
    print("=" * 60)

    # Carregar dataset pequeno para teste
    dataset_path = paths.data.processed / "dataset_1000.pt"

    if not dataset_path.exists():
        print(f"ERRO: Dataset não encontrado em {dataset_path}")
        return False

    print(f"\nCarregando: {dataset_path}")

    # Carregar sem scaling para ver valores originais
    print("\n--- Dados SEM scaling ---")
    dataset_raw = BeamGraphDataset(
        pt_path=str(dataset_path),
        feature_scaler="none",
        target_scaler="none",
    )

    sample = dataset_raw.get(0)
    print(f"Features shape: {sample.x.shape}")
    print(f"Targets shape: {sample.y.shape}")
    print(f"Edge index shape: {sample.edge_index.shape}")

    print("\nFeatures - estatísticas por coluna:")
    for i, name in enumerate(dataset_raw.metadata["feature_names"]["all"]):
        col = dataset_raw.features[:, :, i]
        print(
            f"  {name:20s}: min={col.min():.2e}, max={col.max():.2e}, mean={col.mean():.2e}"
        )

    print("\nTargets (deslocamentos):")
    print(f"  min:  {dataset_raw.targets.min():.6f}")
    print(f"  max:  {dataset_raw.targets.max():.6f}")
    print(f"  mean: {dataset_raw.targets.mean():.6f}")

    # Verificar se há NaN ou Inf
    print("\n--- Verificação de NaN/Inf ---")
    has_nan_features = torch.isnan(dataset_raw.features).any()
    has_nan_targets = torch.isnan(dataset_raw.targets).any()
    has_inf_features = torch.isinf(dataset_raw.features).any()
    has_inf_targets = torch.isinf(dataset_raw.targets).any()

    print(f"NaN em features: {has_nan_features}")
    print(f"NaN em targets:  {has_nan_targets}")
    print(f"Inf em features: {has_inf_features}")
    print(f"Inf em targets:  {has_inf_targets}")

    if has_nan_features or has_nan_targets or has_inf_features or has_inf_targets:
        print("\nERRO: Dados contêm NaN ou Inf!")
        return False

    # Carregar COM scaling
    print("\n--- Dados COM scaling (StandardScaler) ---")
    dataset_scaled = BeamGraphDataset(
        pt_path=str(dataset_path),
        feature_scaler="standard",
        target_scaler="standard",
    )

    print("\nFeatures escaladas - estatísticas gerais:")
    print(f"  min:  {dataset_scaled.features_scaled.min():.4f}")
    print(f"  max:  {dataset_scaled.features_scaled.max():.4f}")
    print(f"  mean: {dataset_scaled.features_scaled.mean():.4f}")
    print(f"  std:  {dataset_scaled.features_scaled.std():.4f}")

    print("\nTargets escalados:")
    print(f"  min:  {dataset_scaled.targets_scaled.min():.4f}")
    print(f"  max:  {dataset_scaled.targets_scaled.max():.4f}")
    print(f"  mean: {dataset_scaled.targets_scaled.mean():.4f}")
    print(f"  std:  {dataset_scaled.targets_scaled.std():.4f}")

    # Verificar edge_index
    print("\n--- Verificação de Edge Index ---")
    sample = dataset_scaled.get(0)
    n_nodes = sample.x.shape[0]
    n_edges = sample.edge_index.shape[1]

    print(f"Número de nós: {n_nodes}")
    print(f"Número de arestas: {n_edges}")
    print(f"Arestas esperadas (grafo linear bidirecional): {2 * (n_nodes - 1)}")

    # Verificar se edge_index está correto
    edge_min = sample.edge_index.min().item()
    edge_max = sample.edge_index.max().item()
    print(f"Índices de edge_index: min={edge_min}, max={edge_max}")

    if edge_max >= n_nodes:
        print("ERRO: edge_index contém índices fora do range!")
        return False

    # Teste de inverse transform
    print("\n--- Teste de Inverse Transform ---")
    original_targets = dataset_raw.targets[0]
    scaled_targets = dataset_scaled.targets_scaled[0]
    recovered_targets = dataset_scaled.inverse_transform_targets(scaled_targets)

    diff = (original_targets - recovered_targets).abs().max()
    print(f"Diferença máxima após inverse transform: {diff:.2e}")

    if diff > 1e-5:
        print("AVISO: Inverse transform tem erro significativo!")

    print("\n" + "=" * 60)
    print("RESULTADO: Pipeline de dados parece correto")
    print("=" * 60)

    return True


if __name__ == "__main__":
    test_data_pipeline()
