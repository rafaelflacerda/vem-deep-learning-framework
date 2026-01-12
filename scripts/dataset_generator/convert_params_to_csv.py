"""Converte todos os arquivos JSON de parâmetros para CSV."""

from src.data.io import params_json_to_csv
from src.paths import paths

PARAMS_DIR = paths.data.raw / "Sobol" / "params" / "rho_0.010"


def main():
    json_files = sorted(PARAMS_DIR.glob("*.json"))

    print(f"Encontrados {len(json_files)} arquivos JSON")

    for json_file in json_files:
        csv_file = params_json_to_csv(json_file)
        print(f"  {json_file.name} -> {csv_file.name}")

    print("Concluído.")


if __name__ == "__main__":
    main()
