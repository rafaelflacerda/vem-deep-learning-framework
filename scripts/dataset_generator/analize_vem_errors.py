"""
Script para análise de erro entre soluções VEM e analíticas
Compara deslocamentos e rotações nodais com solução de viga em balanço
"""

from pathlib import Path

import numpy as np
import pandas as pd


class VEMErrorAnalyzer:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "data/raw/Sobol/results"
        self.params_dir = self.base_dir / "data/raw/Sobol/params"

    def parse_vector(self, vector_str: str) -> np.ndarray:
        """Converte string de vetor (valores separados por ;) em array numpy"""
        values = vector_str.strip().split(";")
        return np.array([float(v) for v in values if v.strip()])

    def analytical_displacement(
        self, x: np.ndarray, q: float, L: float, E: float, I: float
    ) -> np.ndarray:
        """Calcula deslocamento analítico em pontos x"""
        return -q * (x**2) / (24 * E * I) * (6 * L**2 - 4 * L * x + x**2)

    def analytical_slope(
        self, x: np.ndarray, q: float, L: float, E: float, I: float
    ) -> np.ndarray:
        """Calcula rotação analítica em pontos x"""
        return -q * x / (6 * E * I) * (3 * L**2 - 3 * L * x + x**2)

    def load_dataset(self, n_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega resultados e parâmetros para um dataset específico"""
        results_file = self.results_dir / f"results_{n_samples}_samples.csv"
        params_file = self.params_dir / f"params_{n_samples}_samples.csv"

        if not results_file.exists() or not params_file.exists():
            raise FileNotFoundError(
                f"Arquivos não encontrados para {n_samples} samples"
            )

        results_df = pd.read_csv(results_file)
        params_df = pd.read_csv(params_file)

        return results_df, params_df

    def compute_errors_for_case(
        self, case_row: pd.Series, params_row: pd.Series
    ) -> dict:
        """Calcula erros para um caso específico"""
        # Parse dos vetores VEM
        displacements_vem = self.parse_vector(case_row["displacements"])
        rotations_vem = self.parse_vector(case_row["rotations"])

        # Parâmetros do problema
        n_elements = int(case_row["n_elements"])
        L = float(params_row["L"])
        q = float(params_row["q"])
        E = float(params_row["E"])
        I = float(params_row["I"])

        # Gerar coordenadas nodais (malha uniforme)
        n_nodes = n_elements + 1
        x = np.linspace(0, L, n_nodes)

        # Solução analítica
        displacements_analytical = self.analytical_displacement(x, q, L, E, I)
        rotations_analytical = self.analytical_slope(x, q, L, E, I)

        # Verificar dimensões
        if len(displacements_vem) != n_nodes or len(rotations_vem) != n_nodes:
            return None

        # Erros absolutos
        error_disp_abs = np.abs(displacements_vem - displacements_analytical)
        error_rot_abs = np.abs(rotations_vem - rotations_analytical)

        # Erros relativos (evitando divisão por zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            error_disp_rel = (
                np.abs(
                    (displacements_vem - displacements_analytical)
                    / (np.abs(displacements_analytical) + 1e-12)
                )
                * 100
            )
            error_rot_rel = (
                np.abs(
                    (rotations_vem - rotations_analytical)
                    / (np.abs(rotations_analytical) + 1e-12)
                )
                * 100
            )

        # Normas L2
        l2_disp = np.sqrt(np.mean(error_disp_abs**2))
        l2_rot = np.sqrt(np.mean(error_rot_abs**2))

        # Normas infinito (erro máximo)
        linf_disp = np.max(error_disp_abs)
        linf_rot = np.max(error_rot_abs)

        return {
            "case_id": case_row["case_id"],
            "n_elements": n_elements,
            "L": L,
            "q": q,
            "E": E,
            "I": I,
            # Erros absolutos
            "error_disp_mean": np.mean(error_disp_abs),
            "error_disp_max": np.max(error_disp_abs),
            "error_disp_std": np.std(error_disp_abs),
            "error_rot_mean": np.mean(error_rot_abs),
            "error_rot_max": np.max(error_rot_abs),
            "error_rot_std": np.std(error_rot_abs),
            # Erros relativos
            "error_disp_rel_mean": np.mean(error_disp_rel[np.isfinite(error_disp_rel)]),
            "error_disp_rel_max": np.max(error_disp_rel[np.isfinite(error_disp_rel)]),
            "error_rot_rel_mean": np.mean(error_rot_rel[np.isfinite(error_rot_rel)]),
            "error_rot_rel_max": np.max(error_rot_rel[np.isfinite(error_rot_rel)]),
            # Normas
            "l2_disp": l2_disp,
            "l2_rot": l2_rot,
            "linf_disp": linf_disp,
            "linf_rot": linf_rot,
            # Arrays completos para análise detalhada
            "error_disp_nodal": error_disp_abs,
            "error_rot_nodal": error_rot_abs,
        }

    def analyze_dataset(self, n_samples: int) -> pd.DataFrame:
        """Analisa um dataset completo"""
        print(f"\n{'='*70}")
        print(f"Analisando dataset: {n_samples} samples")
        print(f"{'='*70}")

        results_df, params_df = self.load_dataset(n_samples)

        # Merge por case_id
        merged_df = results_df.merge(params_df, on="case_id", how="inner")

        print(f"Total de casos: {len(merged_df)}")

        errors_list = []
        failed_cases = []

        for idx, row in merged_df.iterrows():
            try:
                # Buscar parâmetros correspondentes
                error_dict = self.compute_errors_for_case(row, row)
                if error_dict is not None:
                    errors_list.append(error_dict)
            except Exception as e:
                failed_cases.append((row["case_id"], str(e)))

        if failed_cases:
            print(f"\n⚠️  Casos com falha: {len(failed_cases)}")
            for case_id, error in failed_cases[:5]:
                print(f"  Case {case_id}: {error}")

        # Converter para DataFrame (sem arrays nodais)
        errors_summary = []
        for error_dict in errors_list:
            summary = {
                k: v for k, v in error_dict.items() if not isinstance(v, np.ndarray)
            }
            errors_summary.append(summary)

        errors_df = pd.DataFrame(errors_summary)

        # Estatísticas globais
        self._print_statistics(errors_df)

        # Detectar outliers
        self._detect_outliers(errors_df)

        return errors_df

    def _print_statistics(self, errors_df: pd.DataFrame):
        """Imprime estatísticas globais do dataset"""
        print(f"\n{'─'*70}")
        print("ESTATÍSTICAS GLOBAIS")
        print(f"{'─'*70}")

        print("\nDeslocamentos:")
        print(f"  Erro absoluto médio:  {errors_df['error_disp_mean'].mean():.6e}")
        print(f"  Erro absoluto máximo: {errors_df['error_disp_max'].max():.6e}")
        print(f"  Erro relativo médio:  {errors_df['error_disp_rel_mean'].mean():.2f}%")
        print(f"  L2 norm médio:        {errors_df['l2_disp'].mean():.6e}")
        print(f"  L∞ norm máximo:       {errors_df['linf_disp'].max():.6e}")

        print("\nRotações:")
        print(f"  Erro absoluto médio:  {errors_df['error_rot_mean'].mean():.6e}")
        print(f"  Erro absoluto máximo: {errors_df['error_rot_max'].max():.6e}")
        print(f"  Erro relativo médio:  {errors_df['error_rot_rel_mean'].mean():.2f}%")
        print(f"  L2 norm médio:        {errors_df['l2_rot'].mean():.6e}")
        print(f"  L∞ norm máximo:       {errors_df['linf_rot'].max():.6e}")

    def _detect_outliers(
        self, errors_df: pd.DataFrame, threshold_percentile: float = 95
    ):
        """Detecta e reporta casos com erros anormalmente altos"""
        print(f"\n{'─'*70}")
        print(f"DETECÇÃO DE OUTLIERS (percentil {threshold_percentile})")
        print(f"{'─'*70}")

        # Thresholds
        disp_threshold = np.percentile(
            errors_df["error_disp_max"], threshold_percentile
        )
        rot_threshold = np.percentile(errors_df["error_rot_max"], threshold_percentile)

        outliers_disp = errors_df[errors_df["error_disp_max"] > disp_threshold]
        outliers_rot = errors_df[errors_df["error_rot_max"] > rot_threshold]

        print(
            f"\nCasos com erro de deslocamento > {disp_threshold:.6e}: {len(outliers_disp)}"
        )
        if len(outliers_disp) > 0:
            print("\nTop 5 piores casos (deslocamento):")
            worst = outliers_disp.nlargest(5, "error_disp_max")
            for _, row in worst.iterrows():
                print(
                    f"  Case {row['case_id']}: erro máx = {row['error_disp_max']:.6e}, "
                    f"n_elem = {row['n_elements']}, L = {row['L']:.3f}"
                )

        print(f"\nCasos com erro de rotação > {rot_threshold:.6e}: {len(outliers_rot)}")
        if len(outliers_rot) > 0:
            print("\nTop 5 piores casos (rotação):")
            worst = outliers_rot.nlargest(5, "error_rot_max")
            for _, row in worst.iterrows():
                print(
                    f"  Case {row['case_id']}: erro máx = {row['error_rot_max']:.6e}, "
                    f"n_elem = {row['n_elements']}, L = {row['L']:.3f}"
                )

    def analyze_all_datasets(self) -> dict[int, pd.DataFrame]:
        """Analisa todos os datasets disponíveis"""
        # Buscar todos os arquivos de resultados
        result_files = sorted(self.results_dir.glob("results_*_samples.csv"))

        if not result_files:
            raise FileNotFoundError(f"Nenhum dataset encontrado em {self.results_dir}")

        datasets = {}
        for result_file in result_files:
            # Extrair número de samples do nome do arquivo
            n_samples = int(result_file.stem.split("_")[1])
            errors_df = self.analyze_dataset(n_samples)
            datasets[n_samples] = errors_df

        return datasets

    def save_analysis(self, datasets: dict[int, pd.DataFrame], output_dir: Path):
        """Salva resultados da análise"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for n_samples, errors_df in datasets.items():
            output_file = output_dir / f"error_analysis_{n_samples}_samples.csv"
            errors_df.to_csv(output_file, index=False)
            print(f"\n✓ Análise salva: {output_file}")


def main():
    # Configuração
    base_dir = Path("/Users/rafaelflacerda/00-projects/vem-deep-learning-framework")
    output_dir = base_dir / "data/processed/error_analysis"

    # Criar analisador
    analyzer = VEMErrorAnalyzer(base_dir)

    # Analisar todos os datasets
    try:
        datasets = analyzer.analyze_all_datasets()

        # Salvar resultados
        analyzer.save_analysis(datasets, output_dir)

        print(f"\n{'='*70}")
        print("✓ Análise concluída com sucesso!")
        print(f"{'='*70}")

    except Exception as e:
        print(f"\n❌ Erro durante análise: {e}")
        raise


if __name__ == "__main__":
    main()
