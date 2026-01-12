"""
Módulo de logging centralizado usando loguru.

Este módulo fornece configuração robusta de logging com suporte para:
- Detecção automática e robusta do nome do script (várias fontes)
- Modo não-interativo (padrão): escolhe automaticamente subpasta e nome do arquivo
- Modo interativo (opt-in): permite escolher subpasta e sobrescrever nomes
- Cabeçalho com experiment_name no início do log
- Segurança para multiprocessing (enqueue=True)
- Compatibilidade Python 3.9+

Uso básico (não-interativo, recomendado para pipelines):
    from src.utils.logger import configure_logger
    from loguru import logger

    configure_logger(category="training")
    logger.info("Experimento iniciado")

Uso com interatividade (desenvolvimento/debugging):
    configure_logger(
        category="experiments",
        interactive=True,
        experiment_name=None  # Será pedido no input
    )
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.paths import paths

# ==============================================================================
# FUNÇÕES AUXILIARES INTERNAS
# ==============================================================================


def _detect_script_name() -> str:
    """
    Detecta o nome do script atual de forma robusta, cobrindo múltiplos cenários.

    Tenta múltiplas fontes na seguinte ordem:
    1. sys.argv[0] - se for um arquivo real e válido (não "-c", vazio, ou genérico)
    2. __main__.__spec__ - nome do módulo em execução ('python -m ...')
    3. __main__.__file__ - arquivo principal Python
    4. __main__.__package__ - pacote em execução
    5. fallback seguro: "unknown_script"

    Por que isso importa? Python pode ser invocado de muitas formas:
    - 'python scripts/train.py' → sys.argv[0] funciona bem
    - 'python -m src.scripts.train' → precisa de __spec__ ou __package__
    - 'python -c "code"' → sys.argv[0] é "-c", não usar
    - console_scripts (entrypoint) → sys.argv[0] pode ser genérico
    - notebooks → nenhuma fonte confiável

    Retorna sempre um nome "limpo" (stem sem caminho, sem extensão).

    Returns:
        Nome do script "limpo", sem extensão e sem path, nunca vazio.
    """

    # Tentativa 1: sys.argv[0] - mais literal, mas validar que é um arquivo real
    try:
        if sys.argv and len(sys.argv) > 0:
            argv0 = sys.argv[0].strip()

            # Descartar casos especiais
            if argv0 and argv0 not in ("-c", "-m", "-"):
                script_path = Path(argv0)
                stem = script_path.stem

                # Validar que o stem é sensato (não é um caminho genérico de sistema)
                # Exemplos de inválidos: "", "python", "bash", "sh"
                if stem and stem not in ("python", "python3", "bash", "sh"):
                    return stem
    except (IndexError, ValueError, TypeError, AttributeError):
        pass

    # Tentativa 2: __main__.__spec__ - bom para 'python -m pacote.modulo'
    try:
        import __main__

        if hasattr(__main__, "__spec__") and __main__.__spec__:
            spec = __main__.__spec__
            if hasattr(spec, "name") and spec.name:
                # Extrai o último componente do nome do módulo
                # Ex: "src.scripts.train" → "train"
                module_parts = spec.name.split(".")
                if module_parts:
                    return module_parts[-1]
    except (AttributeError, ValueError, TypeError):
        pass

    # Tentativa 3: __main__.__file__ - funciona bem em execução direta simples
    try:
        import __main__

        if hasattr(__main__, "__file__") and __main__.__file__:
            script_path = Path(__main__.__file__)
            stem = script_path.stem
            if stem:
                return stem
    except (AttributeError, ValueError, TypeError):
        pass

    # Tentativa 4: __main__.__package__ - para 'python -m ...'
    try:
        import __main__

        if hasattr(__main__, "__package__") and __main__.__package__:
            # Extrai o último componente do pacote
            package_parts = __main__.__package__.split(".")
            if package_parts and package_parts[-1]:
                return package_parts[-1]
    except (AttributeError, ValueError, TypeError):
        pass

    # Fallback seguro: arquivo desconhecido
    return "unknown_script"


def _sanitize_name(name: str) -> str:
    """
    Remove caracteres inválidos em nomes de arquivo (especialmente Windows).

    Caracteres problemáticos: : \\ / * ? " < > |
    Esses são proibidos no Windows ou causam problemas em vários filesystems.

    Args:
        name: Nome a ser sanitizado.

    Returns:
        Nome sanitizado, com caracteres inválidos substituídos por underscores.
    """
    invalid_chars = r':<>"/\|?*'
    sanitized = name
    for char in invalid_chars:
        sanitized = sanitized.replace(char, "_")
    return sanitized


def _get_subfolders(log_base_dir: Path) -> list:
    """
    Lista subpastas existentes em um diretório de logs.

    Args:
        log_base_dir: Diretório raiz de logs (ex: results/logs).

    Returns:
        Lista de nomes de subpastas (apenas os nomes, não paths completos).
        Ordenada alfabeticamente. Vazio se não houver subpastas.
    """
    if not log_base_dir.exists():
        return []

    subfolders = []
    for item in log_base_dir.iterdir():
        if item.is_dir():
            subfolders.append(item.name)

    return sorted(subfolders)


def _prompt_choose_subfolder(log_base_dir: Path) -> str:
    """
    Exibe menu interativo para escolher ou criar subpasta de logs.

    Mostra as subpastas existentes com números, permite escolher uma,
    ou criar uma nova digitando o nome.

    Args:
        log_base_dir: Diretório raiz de logs.

    Returns:
        Nome da subpasta escolhida (sem path completo).
    """
    subfolders = _get_subfolders(log_base_dir)

    print("\n" + "=" * 60)
    print("SELEÇÃO DE SUBPASTA DE LOGS")
    print("=" * 60)

    if subfolders:
        print("\nSubpastas existentes:")
        for i, folder in enumerate(subfolders, 1):
            print(f"  {i}. {folder}")
        print(f"  {len(subfolders) + 1}. Criar nova subpasta")
    else:
        print("\nNenhuma subpasta encontrada em:", log_base_dir)
        print("  1. Criar nova subpasta")

    while True:
        try:
            if subfolders:
                choice = input(f"\nEscolha (1-{len(subfolders) + 1}): ").strip()
                choice_num = int(choice)

                if 1 <= choice_num <= len(subfolders):
                    return subfolders[choice_num - 1]
                elif choice_num == len(subfolders) + 1:
                    break  # Criar nova
                else:
                    print(f"Opção inválida. Escolha entre 1 e {len(subfolders) + 1}.")
            else:
                choice = (
                    input("\nDeseja criar uma nova subpasta? (s/n): ").strip().lower()
                )
                if choice == "s":
                    break
                elif choice != "n":
                    print("Digite 's' ou 'n'.")
                else:
                    print("Usando a raiz de logs.")
                    return ""
        except ValueError:
            print("Entrada inválida. Digite um número.")

    # Criar nova subpasta
    while True:
        new_name = input("\nNome da nova subpasta (ex: training, validation): ").strip()
        if new_name:
            new_name = _sanitize_name(new_name)
            return new_name
        else:
            print("Nome não pode estar vazio.")


def _prompt_experiment_name() -> str:
    """
    Pede ao usuário o nome do experimento (interativo).

    Returns:
        Nome do experimento fornecido pelo usuário.
    """
    while True:
        exp_name = input(
            "\nNome do experimento (ex: baseline_mlp, teste_dropout): "
        ).strip()
        if exp_name:
            return _sanitize_name(exp_name)
        else:
            print("Nome do experimento não pode estar vazio.")


def _is_interactive_available() -> bool:
    """
    Verifica se o ambiente atual permite entrada interativa.

    Retorna False em ambientes não-interativos como:
    - Servidores/clusters sem TTY
    - Jobs de batch
    - Redirecionamento de stdin

    Isso evita que o programa trave esperando input quando não há
    possibilidade de o usuário digitar.

    Returns:
        True se há acesso a stdin interativo (TTY), False caso contrário.
    """
    return sys.stdin.isatty() and not os.environ.get("CI")


def _write_log_header(
    experiment_name: str,
    script_name: str,
    category: str | None,
    log_file: Path,
    log_subdir: Path,
) -> None:
    """
    Escreve um cabeçalho informativo no início do arquivo de log.

    O cabeçalho mostra claramente qual experimento/execução gerou os logs
    e onde o arquivo foi salvo, facilitando a identificação posterior.

    No modo interativo, mostrar o caminho do arquivo é crucial, pois o usuário
    pode ter escolhido uma subpasta que não corresponde ao parâmetro 'category'.

    Args:
        experiment_name: Nome do experimento.
        script_name: Nome do script que está sendo executado.
        category: Categoria/tipo de execução (ex: "training", "validation").
        log_file: Path completo do arquivo de log.
        log_subdir: Path da subpasta onde o arquivo foi salvo.
    """
    # Tentar exibir path relativo ao projeto, senão usar path absoluto
    try:
        log_file_display = log_file.relative_to(paths.root)
    except ValueError:
        log_file_display = log_file.absolute()

    try:
        log_subdir_display = log_subdir.relative_to(paths.results.logs)
    except ValueError:
        log_subdir_display = log_subdir.absolute()

    logger.info("=" * 70)
    logger.info("INÍCIO DA EXECUÇÃO")
    logger.info("=" * 70)
    logger.info("Experimento: {}", experiment_name)
    logger.info("Script: {}", script_name)
    if category:
        logger.info("Categoria (argumento): {}", category)
    logger.info("Subpasta efetiva: {}", log_subdir_display)
    logger.info("Arquivo de log: {}", log_file_display)
    logger.info("Timestamp: {}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("PID: {}", os.getpid())
    logger.info("=" * 70)
    logger.info("")


# ==============================================================================
# FUNÇÃO PRINCIPAL DE CONFIGURAÇÃO
# ==============================================================================


def configure_logger(
    category: str | None = None,
    log_dir: Path | None = None,
    level: str = "DEBUG",
    interactive: bool = False,
    experiment_name: str | None = None,
    log_name: str | None = None,
    rotation: str = "500 MB",
    retention: str = "10 days",
):
    # type: (Optional[str], Optional[Path], str, bool, Optional[str], Optional[str], str, str) -> None
    """
    Configura o logger loguru com suporte para modo interativo e não-interativo.

    Deve ser chamada uma única vez no início de qualquer script que precise
    de logging.

    MODO NÃO-INTERATIVO (padrão, interactive=False):
    - Não faz nenhuma pergunta ao usuário
    - Escolhe automaticamente subpasta baseado em 'category' ou 'script_name'
    - Nome do experimento usa 'script_name' como default se não informado
    - Ideal para pipelines, clusters, automação

    MODO INTERATIVO (interactive=True):
    - Mostra menu de subpastas existentes
    - Permite criar nova subpasta
    - Pede nome do experimento se não fornecido
    - Só funciona em ambientes com TTY (trava automaticamente para modo
      não-interativo se não houver TTY)

    Args:
        category: Categoria/tipo de execução (ex: "training", "validation", "preprocessing").
                  Se None, usa o nome do script. Define qual subpasta em logs/ será usada.
        log_dir: Diretório raiz de logs. Se None, usa paths.results.logs.
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: DEBUG.
        interactive: Se True, permite input interativo para escolher subpasta e experiment_name.
                    Se False (padrão), tudo é automático. Se interactive=True mas não há TTY,
                    cai automaticamente para modo não-interativo.
        experiment_name: Nome do experimento/execução. Se None e interactive=True, pede input.
                        Se None e interactive=False, usa script_name como default.
        log_name: Nome base do arquivo de log (sem timestamp/extensão).
                  Se None, usa script_name. O arquivo final será: {log_name}_{timestamp}.log
        rotation: Rotação de arquivos de log (ex: "500 MB", "00:00", "7 days").
                  Cria novo arquivo quando limite é atingido.
        retention: Retenção de logs antigos (ex: "10 days"). Logs mais antigos são deletados.

    Exemplo de uso (não-interativo):
        configure_logger(category="training", experiment_name="baseline_mlp")

    Exemplo de uso (interativo):
        configure_logger(interactive=True)  # Pedirá escolhas se houver TTY
    """

    # ========================================================================
    # ETAPA 1: Detectar script_name e validar diretório de logs
    # ========================================================================

    script_name = _detect_script_name()

    if log_dir is None:
        log_dir = paths.results.logs
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # ETAPA 2: Determinar subpasta de logs (automática ou interativa)
    # ========================================================================

    # Verificar se entrada interativa é realmente possível
    can_interact = interactive and _is_interactive_available()

    if can_interact:
        # Modo interativo: mostrar menu
        subfolder_name = _prompt_choose_subfolder(log_dir)
        if subfolder_name:
            log_subdir = log_dir / subfolder_name
        else:
            log_subdir = log_dir
    else:
        # Modo não-interativo: escolher automaticamente
        if category:
            subfolder_name = _sanitize_name(category)
            log_subdir = log_dir / subfolder_name
        else:
            subfolder_name = _sanitize_name(script_name)
            log_subdir = log_dir / subfolder_name

    # Criar subpasta se necessário
    log_subdir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # ETAPA 3: Determinar experiment_name (automática ou interativa)
    # ========================================================================

    if experiment_name is None:
        if can_interact:
            # Modo interativo: pedir ao usuário
            experiment_name = _prompt_experiment_name()
        else:
            # Modo não-interativo: usar script_name como default
            experiment_name = script_name
    else:
        # Sanitizar se foi fornecido
        experiment_name = _sanitize_name(experiment_name)

    # ========================================================================
    # ETAPA 4: Construir nome do arquivo de log com timestamp único
    # ========================================================================

    if log_name is None:
        log_name_base = script_name
    else:
        log_name_base = _sanitize_name(log_name)

    # Timestamp com microsegundos e PID para garantir unicidade absoluta
    # Em Ray Tune, múltiplos trials podem iniciar no mesmo segundo.
    # Incluir microsegundos (6 dígitos) + PID torna o nome garantidamente único.
    # Formato: {log_name}_{YYYY_MM_DD_HHMMSS}_{microseconds:06d}_{pid}.log
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H%M%S")
    microseconds = now.microsecond
    pid = os.getpid()
    log_filename = f"{log_name_base}_{timestamp}_{microseconds:06d}_{pid}.log"
    log_file = log_subdir / log_filename

    # ========================================================================
    # ETAPA 5: Configurar handlers do loguru
    # ========================================================================

    logger.remove()  # Remove handler padrão

    # Handler para arquivo (detalhado, thread-safe com enqueue)
    #
    # enqueue=True: usa uma fila interna para evitar que múltiplos processos
    # escrevam simultaneamente no mesmo arquivo, o que poderia truncar linhas.
    # Isso é importante em cenários com Ray Tune ou DataLoaders com workers.
    logger.add(
        str(log_file),
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=rotation,
        retention=retention,
        enqueue=True,  # Thread-safe para multiprocessing
    )

    # Handler para console (menos verbose, apenas INFO e acima)
    # logger.add(
    #     sys.stdout,
    #     level="INFO",
    #     format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    # )

    # ========================================================================
    # ETAPA 6: Escrever cabeçalho informativo
    # ========================================================================

    _write_log_header(
        experiment_name=experiment_name,
        script_name=script_name,
        category=category if category else None,
        log_file=log_file,
        log_subdir=log_subdir,
    )
