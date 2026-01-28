"""
Junta partes de um arquivo .pt e verifica integridade.

Uso:
    python scripts/merge_dataset.py <arquivo.pt.manifest>

Exemplo:
    python scripts/merge_dataset.py data/processed/Sobol/rho_0.010_E_fixed_102_elements/dataset_50000.pt.manifest

Lê o manifest, junta as partes na ordem, e verifica o checksum.
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


def merge_file(manifest_path: Path) -> None:
    """
    Junta partes de arquivo usando informações do manifest.

    Args:
        manifest_path: Caminho do arquivo .manifest
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest não encontrado: {manifest_path}")

    # Ler manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    original_filename = manifest["original_filename"]
    original_size = manifest["original_size"]
    expected_checksum = manifest["sha256"]
    n_parts = manifest["n_parts"]
    part_files = manifest["part_files"]

    print(f"Arquivo original: {original_filename}")
    print(f"Tamanho esperado: {original_size / (1024 * 1024):.2f} MB")
    print(f"Partes: {n_parts}")

    # Verificar se todas as partes existem
    parent_dir = manifest_path.parent
    for part_name in part_files:
        part_path = parent_dir / part_name
        if not part_path.exists():
            raise FileNotFoundError(f"Parte não encontrada: {part_path}")

    # Juntar partes
    output_path = parent_dir / original_filename
    print("Juntando partes...")

    with open(output_path, "wb") as out_f:
        for part_name in part_files:
            part_path = parent_dir / part_name
            with open(part_path, "rb") as part_f:
                out_f.write(part_f.read())
            print(f"  {part_name}")

    # Verificar tamanho
    actual_size = output_path.stat().st_size
    if actual_size != original_size:
        print(f"ERRO: Tamanho incorreto!")
        print(f"  Esperado: {original_size} bytes")
        print(f"  Obtido:   {actual_size} bytes")
        return

    # Verificar checksum
    print("Verificando checksum...")
    actual_checksum = calculate_sha256(output_path)

    if actual_checksum != expected_checksum:
        print("ERRO: Checksum não confere!")
        print(f"  Esperado: {expected_checksum}")
        print(f"  Obtido:   {actual_checksum}")
        return

    print(f"SHA256: {actual_checksum}")
    print("Verificação OK - arquivo íntegro.")
    print(f"Arquivo reconstruído: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Junta partes de arquivo .pt")
    parser.add_argument("manifest", type=Path, help="Caminho do arquivo .manifest")

    args = parser.parse_args()
    merge_file(args.manifest)


if __name__ == "__main__":
    main()