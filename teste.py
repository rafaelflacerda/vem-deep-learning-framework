"""
Script simples para verificar o número de features em cada dataset .pt
"""

from pathlib import Path
import torch

# Diretório onde estão os datasets
data_dir = Path("/Users/rafaelflacerda/00-projects/vem-deep-learning-framework/data/processed/Sobol/rho_0.010_E_fixed_102_elements/")

# Listar todos os arquivos .pt no diretório
pt_files = sorted(data_dir.glob("dataset_*.pt"))

if not pt_files:
    print(f"Nenhum arquivo .pt encontrado em {data_dir}")
else:
    print(f"Encontrados {len(pt_files)} arquivos .pt\n")
    
    for pt_file in pt_files:
        # Carregar o dataset
        data = torch.load(pt_file, weights_only=False)
        
        # Extrair número de features
        # Features estão em data["features"] com shape (n_samples, n_nodes, n_features)
        n_samples, n_nodes, n_features = data["features"].shape
        
        print(f"{pt_file.name}:")
        print(f"  Amostras: {n_samples}")
        print(f"  Nós: {n_nodes}")
        print(f"  Features: {n_features}")
        print()