"""
Divide um arquivo .pt em múltiplas partes para upload.

Uso:
    python scripts/split_dataset.py <arquivo.pt> --parts 5

Exemplo:
    python scripts/split_dataset.py data/processed/Sobol/rho_0.010_E_fixed_102_elements/dataset_50000.pt --parts 5

Gera:
    - dataset_50000.pt.part_1
    - dataset_50000.pt.part_2
    - ...
    - dataset_50000.pt.manifest (JSON com checksum e metadados)
"""

import argparse
import hashlib
import json
from pathlib import Path


def calculate_sha256(filepath: Path) -> str:
    """Calcula SHA256 de um arquivo."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def split_file(filepath: Path, n_parts: int) -> None:
    """
    Divide arquivo em n_parts partes binárias.

    Args:
        filepath: Caminho do arquivo .pt
        n_parts: Número de partes para dividir
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    if n_parts < 2:
        raise ValueError("Número de partes deve ser >= 2")

    file_size = filepath.stat().st_size
    print(f"Arquivo: {filepath}")
    print(f"Tamanho: {file_size / (1024 * 1024):.2f} MB")
    print(f"Partes: {n_parts}")

    # Calcular checksum antes de dividir
    print("Calculando checksum...")
    checksum = calculate_sha256(filepath)
    print(f"SHA256: {checksum}")

    # Calcular tamanho de cada parte
    part_size = file_size // n_parts
    remainder = file_size % n_parts

    # Dividir arquivo
    print("Dividindo arquivo...")
    part_files = []

    with open(filepath, "rb") as f:
        for i in range(1, n_parts + 1):
            # Última parte pode ser ligeiramente maior (pega o remainder)
            current_size = part_size + (remainder if i == n_parts else 0)

            part_path = filepath.parent / f"{filepath.name}.part_{i}"
            part_files.append(part_path.name)

            with open(part_path, "wb") as part_f:
                part_f.write(f.read(current_size))

            part_actual_size = part_path.stat().st_size
            print(f"  {part_path.name}: {part_actual_size / (1024 * 1024):.2f} MB")

    # Criar manifest
    manifest = {
        "original_filename": filepath.name,
        "original_size": file_size,
        "sha256": checksum,
        "n_parts": n_parts,
        "part_files": part_files,
    }

    manifest_path = filepath.parent / f"{filepath.name}.manifest"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest: {manifest_path.name}")
    print("Divisão concluída.")


def main():
    parser = argparse.ArgumentParser(description="Divide arquivo .pt em partes")
    parser.add_argument("filepath", type=Path, help="Caminho do arquivo .pt")
    parser.add_argument(
        "--parts", type=int, required=True, help="Número de partes para dividir"
    )

    args = parser.parse_args()
    split_file(args.filepath, args.parts)


if __name__ == "__main__":
    main()