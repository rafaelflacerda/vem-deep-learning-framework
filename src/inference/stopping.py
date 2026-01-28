"""
Critérios de parada para refinamento adaptativo.

Este módulo define uma interface abstrata para critérios de parada e
implementações concretas. Novos critérios podem ser adicionados facilmente
seguindo a interface StoppingCriterion.
"""

from abc import ABC, abstractmethod

import numpy as np


class StoppingCriterion(ABC):
    """Interface abstrata para critérios de parada."""

    @abstractmethod
    def should_stop(
        self,
        iteration: int,
        n_nodes: int,
        y_std: np.ndarray,
        y_pred: np.ndarray,
        positions: np.ndarray,
    ) -> tuple[bool, str]:
        """
        Verifica se o refinamento deve parar.

        Args:
            iteration: Número da iteração atual (começando em 1).
            n_nodes: Número atual de nós na malha.
            y_std: Array de incertezas nos nós.
            y_pred: Array de predições nos nós.
            positions: Array de posições dos nós.

        Returns:
            Tupla (should_stop, reason) onde should_stop é True se deve parar,
            e reason é uma string descrevendo o motivo.
        """
        pass


class MaxUncertaintyCriterion(StoppingCriterion):
    """
    Para quando a incerteza máxima fica abaixo de um threshold.

    Este é o critério principal: o refinamento continua enquanto houver
    algum nó com incerteza acima do limite aceitável.
    """

    def __init__(self, threshold: float):
        """
        Args:
            threshold: Limite de incerteza máxima aceitável [m].
        """
        self.threshold = threshold

    def should_stop(
        self,
        iteration: int,
        n_nodes: int,
        y_std: np.ndarray,
        y_pred: np.ndarray,
        positions: np.ndarray,
    ) -> tuple[bool, str]:
        max_uncertainty = y_std.max()

        if max_uncertainty <= self.threshold:
            return True, f"Incerteza máxima ({max_uncertainty:.2e}) <= threshold ({self.threshold:.2e})"

        return False, ""


class MaxNodesCriterion(StoppingCriterion):
    """Para quando o número de nós atinge um limite máximo."""

    def __init__(self, max_nodes: int):
        self.max_nodes = max_nodes

    def should_stop(
        self,
        iteration: int,
        n_nodes: int,
        y_std: np.ndarray,
        y_pred: np.ndarray,
        positions: np.ndarray,
    ) -> tuple[bool, str]:
        if n_nodes >= self.max_nodes:
            return True, f"Número máximo de nós atingido ({n_nodes} >= {self.max_nodes})"

        return False, ""


class MaxIterationsCriterion(StoppingCriterion):
    """Para quando o número de iterações atinge um limite máximo."""

    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def should_stop(
        self,
        iteration: int,
        n_nodes: int,
        y_std: np.ndarray,
        y_pred: np.ndarray,
        positions: np.ndarray,
    ) -> tuple[bool, str]:
        if iteration >= self.max_iterations:
            return True, f"Número máximo de iterações atingido ({iteration} >= {self.max_iterations})"

        return False, ""


class CompositeCriterion(StoppingCriterion):
    """
    Combina múltiplos critérios de parada.

    Para quando QUALQUER um dos critérios é satisfeito.
    """

    def __init__(self, criteria: list[StoppingCriterion]):
        self.criteria = criteria

    def should_stop(
        self,
        iteration: int,
        n_nodes: int,
        y_std: np.ndarray,
        y_pred: np.ndarray,
        positions: np.ndarray,
    ) -> tuple[bool, str]:
        for criterion in self.criteria:
            should_stop, reason = criterion.should_stop(
                iteration, n_nodes, y_std, y_pred, positions
            )
            if should_stop:
                return True, reason

        return False, ""


def create_stopping_criterion(
    uncertainty_threshold: float,
    max_nodes: int,
    max_iterations: int,
) -> StoppingCriterion:
    """
    Factory function para criar o critério de parada composto padrão.

    Args:
        uncertainty_threshold: Limite de incerteza máxima [m].
        max_nodes: Número máximo de nós.
        max_iterations: Número máximo de iterações.

    Returns:
        Critério de parada composto.
    """
    criteria = [
        MaxUncertaintyCriterion(uncertainty_threshold),
        MaxNodesCriterion(max_nodes),
        MaxIterationsCriterion(max_iterations),
    ]
    return CompositeCriterion(criteria)