"""
Pré-processa o dataset aplicando scaling e salvando em cache.

Uso:
    python scripts/prepare_scaled_dataset.py \
        --dataset data/processed/Sobol/slenderness_method/dataset_100000.pt \
        --scaler minmax
"""

import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.scalers import get_scaler


def main():
    parser = argparse.ArgumentParser(description="Pré-processa dataset com scaling")
    parser.add_argument("--dataset", type=str, required=True, help="Caminho do .pt original")
    parser.add_argument("--scaler", type=str, default="minmax", help="Tipo de scaler (minmax, standard, robust)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Erro: arquivo não encontrado: {dataset_path}")
        sys.exit(1)

    # Caminho de saída
    output_path = dataset_path.parent / f"{dataset_path.stem}_scaled_{args.scaler}.pt"

    print(f"Dataset original: {dataset_path}")
    print(f"Scaler: {args.scaler}")
    print(f"Saída: {output_path}")
    print()

    # Carregar dados originais
    print("Carregando dataset original...")
    data = torch.load(dataset_path, weights_only=False)
    data_list = data["data_list"]
    metadata = data["metadata"]
    print(f"Carregado: {len(data_list)} grafos")

    # Criar scalers
    feature_scaler = get_scaler(args.scaler)
    target_scaler = get_scaler(args.scaler)

    # Fitar scalers
    print("Fitando scalers...")
    all_features = torch.cat([d.x for d in data_list], dim=0)
    all_targets = torch.cat([d.y for d in data_list], dim=0)

    feature_scaler.fit(all_features.unsqueeze(0))
    target_scaler.fit(all_targets.unsqueeze(0).unsqueeze(-1))
    print("Scalers fitados")

    # Aplicar scaling
    print("Aplicando scaling em todos os grafos...")
    data_list_scaled = []
    for i, d in enumerate(data_list):
        x_scaled = feature_scaler.transform(d.x.unsqueeze(0)).squeeze(0)
        y_scaled = target_scaler.transform(d.y.unsqueeze(0).unsqueeze(-1)).squeeze(0).squeeze(-1)

        scaled_data = Data(
            x=x_scaled,
            edge_index=d.edge_index,
            y=y_scaled,
            n_elements=d.n_elements,
        )
        data_list_scaled.append(scaled_data)

        if (i + 1) % 20000 == 0:
            print(f"  Processado: {i + 1}/{len(data_list)}")

    print(f"Scaling completo: {len(data_list_scaled)} grafos")

    # Salvar cache
    print("Salvando cache...")
    cache = {
        "data_list_scaled": data_list_scaled,
        "metadata": metadata,
        "feature_scaler_state": feature_scaler.state_dict(),
        "target_scaler_state": target_scaler.state_dict(),
        "scaler_type": args.scaler,
    }
    torch.save(cache, output_path)

    # Tamanho do arquivo
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Cache salvo: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()