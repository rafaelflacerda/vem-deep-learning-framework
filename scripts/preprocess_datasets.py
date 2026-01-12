"""
Script para preprocessar datasets VEM (JSON → .pt).

Converte JSONs raw em arquivos .pt otimizados para treino com PyTorch.
Configuração via YAML: configs/preprocessing/features.yaml

Uso:
    python scripts/preprocess_datasets.py
"""

from loguru import logger

from src.data.preprocessing import preprocess_all_datasets
from src.paths import paths
from src.utils.logger import configure_logger


def main():
    """Função principal do script de preprocessamento."""

    # Configurar logging
    configure_logger(
        category="preprocessing", experiment_name="json_to_pt", level="INFO"
    )

    logger.info("Iniciando preprocessamento de datasets VEM")

    # Path da configuração
    config_path = paths.configs.root / "preprocessing" / "features.yaml"

    if not config_path.exists():
        logger.error("Arquivo de configuração não encontrado: {}", config_path)
        logger.error("Crie o arquivo configs/preprocessing/features.yaml")
        return

    try:
        # Executar preprocessamento
        output_paths = preprocess_all_datasets(config_path)

        logger.info("")
        logger.info("=" * 70)
        logger.info("PREPROCESSAMENTO CONCLUÍDO COM SUCESSO")
        logger.info("=" * 70)
        logger.info("Datasets criados:")
        for path in output_paths:
            try:
                rel_path = path.relative_to(paths.root)
            except ValueError:
                rel_path = path
            file_size_mb = path.stat().st_size / (1024 * 1024)
            logger.info("  - {} ({:.2f} MB)", rel_path, file_size_mb)
        logger.info("")
        logger.info("Os arquivos .pt estão prontos para uso em treino.")
        logger.info("Localização: {}", paths.data.processed)
        logger.info("=" * 70)

    except Exception:
        logger.exception("Erro durante preprocessamento")
        raise


if __name__ == "__main__":
    main()
