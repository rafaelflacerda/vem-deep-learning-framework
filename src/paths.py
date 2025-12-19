"""
Centralização de paths do projeto.

Este módulo é a ÚNICA fonte de verdade para todos os caminhos do projeto.
Nunca hardcode paths em outros arquivos — sempre importe daqui.

Uso básico:
    from src.paths import paths
    
    # Acessar diretórios
    dados = paths.data.raw / "meu_arquivo.npz"
    
    # Criar diretório para um novo experimento
    exp_dir = paths.create_experiment_dir("teste_mlp")
    
    # Garantir que um diretório existe
    paths.ensure_dir(paths.results.figures)

Uso alternativo (variáveis diretas):
    from src.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

def _find_project_root() -> Path:
    """
    Encontra a raiz do projeto subindo na árvore de diretórios.
    
    A raiz é identificada pela presença do arquivo pyproject.toml.
    Isso funciona independente de onde o código é executado, seja:
    - De dentro do pacote src/
    - De um notebook em notebooks/
    - De um script em scripts/
    - Do diretório raiz
    
    Returns:
        Path para o diretório raiz do projeto.
        
    Raises:
        RuntimeError: Se não encontrar pyproject.toml em nenhum diretório pai.
    """
    
    # Começa do diretório onde este arquivo (paths.py) está localizado
    current = Path(__file__).resolve().parent
    
    # Sobe na árvore de diretórios até encontrar pyproject.toml
    # O limite current != current.parent evita loop infinito na raiz do sistema
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    
    # Se chegou aqui, não encontrou o arquivo em lugar nenhum
    raise RuntimeError(
        "Não foi possível encontrar a raiz do projeto. "
        "Certifique-se de que pyproject.toml existe no diretório raiz."
    )

PROJECT_ROOT = _find_project_root()

# Estas classes organizam os paths de forma hierárquica, permitindo acesso intuitivo como paths.data.raw, paths.results.figures, etc.
# O uso de @dataclass com frozen=True garante que os paths não sejam modificados acidentalmente durante a execução.

@dataclass(frozen=True)
class DataPaths:
    """Paths relacionados a dados."""
    
    root: Path
    """Diretório raiz de dados: data/"""
    
    raw: Path
    """Dados brutos, como gerados pelo solver: data/raw/"""
    
    interim: Path
    """Dados em processamento intermediário: data/interim/"""
    
    processed: Path
    """Dados prontos para treinamento: data/processed/"""
    
    external: Path
    """Dados externos, como dados de outros projetos: data/external/"""

@dataclass(frozen=True)
class ResultsPaths:
    """Paths relacionados a resultados."""
    
    root: Path
    """Diretório raiz de resultados: results/"""
    
    checkpoints: Path
    """Checkpoints de modelos salvos: results/checkpoints/"""
    
    experiments: Path
    """Resultados de experimentos: results/experiments/"""
    
    figures: Path
    """Figuras e visualizações: results/figures/"""
    
    logs: Path
    """Logs de execução: results/logs/"""


@dataclass(frozen=True)
class ConfigsPaths:
    """Paths relacionados a configurações."""
    
    root: Path
    """Diretório raiz de configs: configs/"""
    
    environments: Path
    """Configurações de ambiente: configs/environments/"""
    
    experiments: Path
    """Configurações de experimentos: configs/experiments/"""


@dataclass(frozen=True)
class DocsPaths:
    """Paths relacionados a documentação."""
    
    root: Path
    """Diretório raiz de docs: docs/"""
    
    articles: Path
    """Artigos e papers: docs/articles/"""
    
    images: Path
    """Imagens para documentação: docs/images/"""
    
    mkdocs: Path
    """Documentação MkDocs: docs/mkdocs/"""
    
    notes: Path
    """Notas pessoais: docs/notes/"""
    
    thesis: Path
    """Texto do TCC: docs/thesis/"""


@dataclass(frozen=True)
class ProjectPaths:
    """
    Estrutura completa de paths do projeto.
    
    Esta é a classe principal que agrega todos os paths organizados
    hierarquicamente. Use através da instância global `paths`.
    
    Exemplo:
        from src.paths import paths
        
        # Acessar subdiretórios
        raw_data = paths.data.raw
        figures = paths.results.figures
        
        # Construir paths para arquivos
        dataset = paths.data.processed / "train.npz"
        config = paths.configs.experiments / "baseline.yaml"
    """
    
    root: Path
    """Diretório raiz do projeto (onde está pyproject.toml)."""
    
    data: DataPaths
    """Paths de dados."""
    
    results: ResultsPaths
    """Paths de resultados."""
    
    configs: ConfigsPaths
    """Paths de configurações."""
    
    docs: DocsPaths
    """Paths de documentação."""
    
    src: Path
    """Diretório do código fonte: src/"""
    
    tests: Path
    """Diretório de testes: tests/"""
    
    notebooks: Path
    """Diretório de notebooks: notebooks/"""
    
    scripts: Path
    """Diretório de scripts: scripts/"""
    
    solver_vem: Path
    """Diretório do solver VEM (C++): solver_vem/"""
    
    latex: Path
    """Diretório de arquivos LaTeX: latex/"""

@dataclass(frozen=True)
class SamplePaths:
    """
    Paths para um conjunto específico de samples.
    
    Cada conjunto de samples tem uma estrutura:
        data/raw/{n}_samples/
        ├── params/    <- arquivos JSON com parâmetros de entrada
        └── results/   <- arquivos com resultados do solver VEM
    """
    
    root: Path
    """Diretório raiz do conjunto de samples: data/raw/{n}_samples/"""
    
    params: Path
    """Diretório de parâmetros: data/raw/{n}_samples/params/"""
    
    results: Path
    """Diretório de resultados: data/raw/{n}_samples/results/"""
    
    n_samples: int
    """Número de samples neste conjunto."""

def _create_project_paths() -> ProjectPaths:
    """
    Cria a estrutura completa de paths do projeto.
    
    Esta função é chamada uma única vez quando o módulo é importado.
    Ela constrói toda a hierarquia de paths a partir do PROJECT_ROOT.
    """
    root = PROJECT_ROOT
    
    # Diretório de dados e subdiretórios
    data_root = root / "data"
    data = DataPaths(
        root=data_root,
        raw=data_root / "raw",
        interim=data_root / "interim",
        processed=data_root / "processed",
        external=data_root / "external",
    )
    
    # Diretório de resultados e subdiretórios
    results_root = root / "results"
    results = ResultsPaths(
        root=results_root,
        checkpoints=results_root / "checkpoints",
        experiments=results_root / "experiments",
        figures=results_root / "figures",
        logs=results_root / "logs",
    )
    
    # Diretório de configs e subdiretórios
    configs_root = root / "configs"
    configs = ConfigsPaths(
        root=configs_root,
        environments=configs_root / "environments",
        experiments=configs_root / "experiments",
    )
    
    # Diretório de docs e subdiretórios
    docs_root = root / "docs"
    docs = DocsPaths(
        root=docs_root,
        articles=docs_root / "articles",
        images=docs_root / "images",
        mkdocs=docs_root / "mkdocs",
        notes=docs_root / "notes",
        thesis=docs_root / "thesis",
    )
    
    return ProjectPaths(
        root=root,
        data=data,
        results=results,
        configs=configs,
        docs=docs,
        src=root / "src",
        tests=root / "tests",
        notebooks=root / "notebooks",
        scripts=root / "scripts",
        solver_vem=root / "solver_vem",
        latex=root / "latex",
    )


# ==============================================================================
# INSTÂNCIA GLOBAL DE PATHS
# ==============================================================================
# Esta é a instância que você deve usar em todo o projeto.
# Ela é criada uma única vez quando o módulo é importado.
# ==============================================================================

paths = _create_project_paths()


# ==============================================================================
# VARIÁVEIS DE CONVENIÊNCIA (para imports diretos)
# ==============================================================================
# Estas variáveis permitem imports mais curtos para os paths mais usados:
#   from src.paths import DATA_DIR, RESULTS_DIR
# ==============================================================================

# Diretórios principais
DATA_DIR = paths.data.root
RESULTS_DIR = paths.results.root
CONFIGS_DIR = paths.configs.root
DOCS_DIR = paths.docs.root

# Subdiretórios de dados
RAW_DATA_DIR = paths.data.raw
INTERIM_DATA_DIR = paths.data.interim
PROCESSED_DATA_DIR = paths.data.processed
EXTERNAL_DATA_DIR = paths.data.external
# Subdiretórios de resultados
CHECKPOINTS_DIR = paths.results.checkpoints
EXPERIMENTS_DIR = paths.results.experiments
FIGURES_DIR = paths.results.figures
LOGS_DIR = paths.results.logs

# Subdiretórios de configs
ENVIRONMENTS_DIR = paths.configs.environments
EXPERIMENTS_CONFIG_DIR = paths.configs.experiments


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def ensure_dir(path: Path) -> Path:
    """
    Garante que um diretório existe, criando-o se necessário.
    
    Esta função é idempotente: pode ser chamada múltiplas vezes
    sem efeitos colaterais se o diretório já existir.
    
    Args:
        path: Caminho do diretório a ser criado.
        
    Returns:
        O mesmo path passado como argumento (permite encadeamento).
        
    Exemplo:
        from src.paths import paths, ensure_dir
        
        # Garante que o diretório existe e já usa
        output = ensure_dir(paths.results.figures) / "grafico.png"
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_experiment_dir(name: str, timestamp: bool = True) -> Path:
    """
    Cria um diretório para um novo experimento.
    
    Os experimentos são organizados em results/experiments/ com nomes
    que incluem timestamp para evitar conflitos e facilitar ordenação
    cronológica.
    
    Args:
        name: Nome descritivo do experimento (ex: "baseline_mlp", "teste_lr").
        timestamp: Se True, adiciona timestamp ao nome. Default True.
        
    Returns:
        Path do diretório criado.
        
    Exemplo:
        from src.paths import create_experiment_dir
        
        # Cria: results/experiments/2025-01-15_143022_baseline_mlp/
        exp_dir = create_experiment_dir("baseline_mlp")
        
        # Salvar arquivos do experimento
        config_file = exp_dir / "config.yaml"
        metrics_file = exp_dir / "metrics.json"
        model_file = exp_dir / "model.pt"
    """
    if timestamp:
        # Formato: YYYY-MM-DD_HHMMSS_nome
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        dir_name = f"{ts}_{name}"
    else:
        dir_name = name
    
    exp_dir = paths.results.experiments / dir_name
    ensure_dir(exp_dir)
    
    return exp_dir


def create_run_dir(experiment_dir: Path, run_name: str | None = None) -> Path:
    """
    Cria um diretório para uma execução específica dentro de um experimento.
    
    Útil quando você quer rodar o mesmo experimento múltiplas vezes
    (por exemplo, com seeds diferentes) e manter os resultados separados.
    
    Args:
        experiment_dir: Diretório do experimento pai.
        run_name: Nome opcional da execução. Se None, usa timestamp.
        
    Returns:
        Path do diretório criado.
        
    Exemplo:
        exp_dir = create_experiment_dir("sweep_learning_rate")
        
        for lr in [0.001, 0.01, 0.1]:
            run_dir = create_run_dir(exp_dir, f"lr_{lr}")
            # Treinar e salvar resultados em run_dir
    """
    if run_name is None:
        run_name = datetime.now().strftime("%H%M%S")
    
    run_dir = experiment_dir / run_name
    ensure_dir(run_dir)
    
    return run_dir


def get_latest_experiment(pattern: str = "*") -> Path | None:
    """
    Retorna o diretório do experimento mais recente.
    
    Útil para continuar trabalho ou analisar resultados do último
    experimento executado.
    
    Args:
        pattern: Padrão glob para filtrar experimentos. Default "*" (todos).
        
    Returns:
        Path do experimento mais recente, ou None se não houver nenhum.
        
    Exemplo:
        from src.paths import get_latest_experiment
        
        # Último experimento de qualquer tipo
        latest = get_latest_experiment()
        
        # Último experimento que começa com "baseline"
        latest_baseline = get_latest_experiment("*baseline*")
    """
    experiments = sorted(paths.results.experiments.glob(pattern))
    return experiments[-1] if experiments else None

def get_sample_paths(n_samples: int) -> SamplePaths:
    """
    Retorna os paths para um conjunto específico de samples.
    
    Args:
        n_samples: Número de samples (ex: 10, 100, 1000, 10000).
        
    Returns:
        SamplePaths com acesso a root, params e results.
        
    Raises:
        ValueError: Se o diretório para esse número de samples não existir.
        
    Exemplo:
        from src.paths import get_sample_paths
        
        sample = get_sample_paths(1000)
        
        # Acessar diretórios
        params_dir = sample.params
        results_dir = sample.results
        
        # Construir path para arquivo específico
        param_file = sample.params / "case_042.json"
        result_file = sample.results / "case_042.npz"
        
        # Iterar sobre todos os arquivos de parâmetros
        for param_file in sample.params.glob("*.json"):
            print(param_file.name)
    """
    sample_dir = paths.data.raw / f"{n_samples}_samples"
    
    if not sample_dir.exists():
        available = list_available_samples()
        raise ValueError(
            f"Diretório para {n_samples} samples não encontrado: {sample_dir}\n"
            f"Conjuntos disponíveis: {available}"
        )
    
    return SamplePaths(
        root=sample_dir,
        params=sample_dir / "params",
        results=sample_dir / "results",
        n_samples=n_samples,
    )


def list_available_samples() -> list[int]:
    """
    Lista os conjuntos de samples disponíveis em data/raw/.
    
    Procura por diretórios com o padrão {n}_samples e retorna
    os valores de n encontrados, ordenados.
    
    Returns:
        Lista de inteiros representando os tamanhos de samples disponíveis.
        
    Exemplo:
        from src.paths import list_available_samples
        
        available = list_available_samples()
        # [10, 100, 250, 1000, 2500, 10000, 25000]
        
        # Processar todos os conjuntos
        for n in available:
            sample = get_sample_paths(n)
            # fazer algo com sample.params, sample.results
    """
    sample_dirs = paths.data.raw.glob("*_samples")
    sizes = []
    
    for dir_path in sample_dirs:
        # Extrai o número do nome da pasta (ex: "1000_samples" -> 1000)
        name = dir_path.name
        if name.endswith("_samples"):
            try:
                n = int(name.replace("_samples", ""))
                sizes.append(n)
            except ValueError:
                # Ignora pastas que não seguem o padrão esperado
                pass
    
    return sorted(sizes)


def get_all_sample_paths() -> dict[int, SamplePaths]:
    """
    Retorna um dicionário com todos os conjuntos de samples disponíveis.
    
    Útil quando você quer processar todos os conjuntos de uma vez.
    
    Returns:
        Dicionário mapeando n_samples -> SamplePaths.
        
    Exemplo:
        from src.paths import get_all_sample_paths
        
        all_samples = get_all_sample_paths()
        
        for n, sample in all_samples.items():
            print(f"{n} samples: {sample.root}")
            n_params = len(list(sample.params.glob("*.json")))
            print(f"  {n_params} arquivos de parâmetros")
    """
    return {n: get_sample_paths(n) for n in list_available_samples()}


# ==============================================================================
# VALIDAÇÃO (executada quando o módulo é importado)
# ==============================================================================

def _validate_project_structure() -> None:
    """
    Valida que a estrutura básica do projeto existe.
    
    Esta função emite warnings (não erros) se diretórios importantes
    não existirem. Isso ajuda a identificar problemas de configuração
    sem quebrar a execução.
    """
    import warnings
    
    critical_dirs = [
        paths.src,
        paths.data.root,
        paths.results.root,
        paths.configs.root,
    ]
    
    for dir_path in critical_dirs:
        if not dir_path.exists():
            warnings.warn(
                f"Diretório não encontrado: {dir_path}. "
                f"Execute 'mkdir -p {dir_path}' para criar.",
                stacklevel=2,
            )


# Descomente a linha abaixo se quiser validação automática ao importar
# _validate_project_structure()


# ==============================================================================
# INFORMAÇÕES DE DEBUG
# ==============================================================================

if __name__ == "__main__":
    # Se executar este arquivo diretamente, mostra informações úteis
    print("=" * 60)
    print("PATHS DO PROJETO")
    print("=" * 60)
    print(f"Raiz do projeto: {PROJECT_ROOT}")
    print()
    print("Estrutura de diretórios:")
    print(f"  src/           : {paths.src}")
    print(f"  data/          : {paths.data.root}")
    print(f"    raw/         : {paths.data.raw}")
    print(f"    interim/     : {paths.data.interim}")
    print(f"    processed/   : {paths.data.processed}")
    print(f"  results/       : {paths.results.root}")
    print(f"    checkpoints/ : {paths.results.checkpoints}")
    print(f"    experiments/ : {paths.results.experiments}")
    print(f"    figures/     : {paths.results.figures}")
    print(f"    logs/        : {paths.results.logs}")
    print(f"  configs/       : {paths.configs.root}")
    print(f"    environments/: {paths.configs.environments}")
    print(f"    experiments/ : {paths.configs.experiments}")
    print(f"  tests/         : {paths.tests}")
    print(f"  notebooks/     : {paths.notebooks}")
    print(f"  scripts/       : {paths.scripts}")
    print()
    print("Verificação de existência:")
    for name, path in [
        ("src", paths.src),
        ("data", paths.data.root),
        ("results", paths.results.root),
        ("configs", paths.configs.root),
    ]:
        status = "✓ existe" if path.exists() else "✗ não existe"
        print(f"  {name}: {status}")