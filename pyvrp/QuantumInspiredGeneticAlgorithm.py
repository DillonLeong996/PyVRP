from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Collection

from pyvrp.ProgressPrinter import ProgressPrinter
from pyvrp.Result import Result
from pyvrp.Statistics import Statistics

if TYPE_CHECKING:
    from pyvrp.PenaltyManager import PenaltyManager
    from pyvrp.PopulationQCI import PopulationQCI
    from pyvrp._pyvrp import (
        CostEvaluator,
        ProblemData,
        RandomNumberGenerator,
        Solution,
    )
    from pyvrp.search.SearchMethod import SearchMethod
    from pyvrp.stop.StoppingCriterion import StoppingCriterion


@dataclass
class QuantumInspiredGeneticAlgorithmParams:
    """
    Parameters for the quantum-inspired genetic algorithm.

    Parameters
    ----------
    repair_probability
        Probability (in :math:`[0, 1]`) of repairing an infeasible solution.
        If the reparation makes the solution feasible, it is also added to
        the population in the same iteration.
    num_iters_no_improvement
        Number of iterations without any improvement needed before a restart
        occurs.

    Attributes
    ----------
    repair_probability
        Probability of repairing an infeasible solution.
    num_iters_no_improvement
        Number of iterations without improvement before a restart occurs.

    Raises
    ------
    ValueError
        When ``repair_probability`` is not in :math:`[0, 1]`, or
        ``num_iters_no_improvement`` is negative.
    """

    repair_probability: float = 0.80
    num_iters_no_improvement: int = 20_000
    num_of_qubits: int = 3
    crossover_rate = 0.8
    mutation_rate = 0.2
    rotation_angle = np.pi * 0.25

    def __post_init__(self):
        if not 0 <= self.repair_probability <= 1:
            raise ValueError("repair_probability must be in [0, 1].")

        if self.num_iters_no_improvement < 0:
            raise ValueError("num_iters_no_improvement < 0 not understood.")
        
        if self.num_of_qubits < 1:
            raise ValueError("num_of_qubits < 1 not understood.")
        


class QuantumInspiredGeneticAlgorithm:
    """
    Creates a Quantum-inspired GeneticAlgorithm instance.

    Parameters
    ----------
    data
        Data object describing the problem to be solved.
    penalty_manager
        Penalty manager to use.
    rng
        Random number generator.
    population
        Population to use.
    search_method
        Search method to use.
    crossover_op
        Crossover operator to use for generating offspring.
    initial_solutions
        Initial solutions to use to initialise the population.
    params
        Genetic algorithm parameters. If not provided, a default will be used.

    Raises
    ------
    ValueError
        When the population is empty.
    """

    def __init__(
        self,
        data: ProblemData,
        penalty_manager: PenaltyManager,
        rng: RandomNumberGenerator,
        population: PopulationQCI,
        search_method: SearchMethod,
        crossover_op: Callable[
            [
                tuple[Solution, Solution],
                ProblemData,
                CostEvaluator,
                RandomNumberGenerator,
            ],
            Solution,
        ],
        initial_solutions: Collection[Solution],
        params: QuantumInspiredGeneticAlgorithmParams = QuantumInspiredGeneticAlgorithmParams(),
    ):
        if len(initial_solutions) == 0:
            raise ValueError("Expected at least one initial solution.")

        self._data = data
        self._pm = penalty_manager
        self._rng = rng
        self._pop = population
        self._search = search_method
        self._crossover = crossover_op
        self._initial_solutions = initial_solutions
        self._params = params
        self._sols = np.zeros(self._params.num_of_qubits, dtype = object)

        # Find best feasible initial solution if any exist, else set a random
        # infeasible solution (with infinite cost) as the initial best.
        self._best = min(initial_solutions, key=self._cost_evaluator.cost)

    @property
    def _cost_evaluator(self) -> CostEvaluator:
        return self._pm.cost_evaluator()
    
    def _addQubits(self, *solutions):
        """
        Adds solutions into the current population.
        """
        m = self._params.num_of_qubits

        sols = np.zeros((m, 2), dtype = object)
        costs = np.zeros((m, 2), dtype = object)
        angles = np.zeros((m, 2))
        for sol in solutions:
            for c_col in range(m):
                theta_angle = self._rng.rand()*np.pi
                angles[c_col][0] = np.cos(theta_angle)
                angles[c_col][1] = np.sin(theta_angle)
                for c_row in range(2):
                    sols[c_col][c_row] = sol
                    costs[c_col][c_row] = self._cost_evaluator
            #self._pop.addqci(sols, costs)
            self._pop.addqci(sols, costs, angles, self._rng)


    def run(
        self,
        stop: StoppingCriterion,
        collect_stats: bool = True,
        display: bool = False,
        display_interval: float = 5.0,
    ):
        """
        Runs the quantum-inspired genetic algorithm with the provided stopping criterion.

        Parameters
        ----------
        stop
            Stopping criterion to use. The algorithm runs until the first time
            the stopping criterion returns ``True``.
        collect_stats
            Whether to collect statistics about the solver's progress. Default
            ``True``.
        display
            Whether to display information about the solver progress. Default
            ``False``. Progress information is only available when
            ``collect_stats`` is also set.
        display_interval
            Time (in seconds) between iteration logs. Defaults to 5s.

        Returns
        -------
        Result
            A Result object, containing statistics (if collected) and the best
            found solution.
        """
        print_progress = ProgressPrinter(display, display_interval)
        print_progress.start(self._data)

        start = time.perf_counter()
        stats = Statistics(collect_stats=collect_stats)
        iters = 0
        iters_no_improvement = 1

        self._addQubits(*self._initial_solutions)

        while not stop(self._cost_evaluator.cost(self._best)):
            iters += 1

            if iters_no_improvement == self._params.num_iters_no_improvement:
                print_progress.restart()

                iters_no_improvement = 1
                self._pop.clear()

                #for sol in self._initial_solutions:
                #    self._pop.addqci(sol, self._cost_evaluator)
                self._addQubits(*self._initial_solutions)

            curr_best = self._cost_evaluator.cost(self._best)

            if self._rng.rand() < self._params.crossover_rate :
                parents = self._pop.selectqci(self._rng, self._cost_evaluator)
                first, second = parents
                quantum_rot_gate = np.array([[np.cos(self._params.rotation_angle), -np.sin(self._params.rotation_angle)], [np.sin(self._params.rotation_angle), np.cos(self._params.rotation_angle)]])
                for c in range(self._params.num_of_qubits):
                    for r in range(2):
                        first[2][c][r] = np.dot(quantum_rot_gate[r],first[2][c][r])
                        second[2][c][r] = np.dot(quantum_rot_gate[r],second[2][c][r])

                for c in range(self._params.num_of_qubits):
                    if self._rng.rand() < self._params.mutation_rate and c + 1 < self._params.num_of_qubits:
                        if first[2][c][1][0] > 0.5:
                            first[2][c+1] = np.array([[first[2][c+1][1][0]], [first[2][c+1][0][0]]])
                        if second[2][c][1][0] > 0.5:
                            second[2][c+1] = np.array([[second[2][c+1][1][0]], [second[2][c+1][0][0]]])


                #offspring = self._crossover(
                #    parents, self._data, self._cost_evaluator, self._rng
                #)
                self._improve_offspring(first)
                self._improve_offspring(second)

            new_best = self._cost_evaluator.cost(self._best)

            if new_best < curr_best:
                iters_no_improvement = 1
            else:
                iters_no_improvement += 1

            stats.collect_from_qci(self._pop, self._cost_evaluator)
            print_progress.iteration(stats)

        end = time.perf_counter() - start
        res = Result(self._best, stats, iters, end)

        print_progress.end(res)

        return res

    def _improve_offspring(self, sol: Solution):
        def is_new_best(sol):
            for c in range(self._params.num_of_qubits):
                for r in range(2):
                    cost = self._cost_evaluator.cost(sol[0][c][r])
                    best_cost = self._cost_evaluator.cost(self._best)
                    return cost < best_cost

        for c in range(self._params.num_of_qubits):
            for r in range(2):
                sol[0][c][r] = self._search(sol[0][c][r], self._cost_evaluator)
        #self._pop.addqci(sol, self._cost_evaluator)
        self._addQubits(sol)

        for c in range(self._params.num_of_qubits):
            for r in range(2):
                self._pm.register(sol[0][c][r])

                if is_new_best(sol[0][c][r]):
                    self._best = sol[0][c][r]

                # Possibly repair if current solution is infeasible. In that case, we
                # penalise infeasibility more using a penalty booster.
                if (
                    not sol[0][c][r].is_feasible()
                    and self._rng.rand() < self._params.repair_probability
                ):
                    sol[0][c][r] = self._search(sol[0][c][r], self._pm.booster_cost_evaluator())

                    if sol[0][c][r].is_feasible():
                        self._pop.addqci(sol[0][c][r], self._cost_evaluator)
                        self._pm.register(sol[0][c][r])

                    if is_new_best(sol[0][c][r]):
                        self._best = sol[0][c][r]
