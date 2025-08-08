from __future__ import annotations

import math
import numpy as np

from typing import TYPE_CHECKING, Callable, Generator

from pyvrp._pyvrp import PopulationParams, SubPopulation
#from pyvrp.QuantumInspiredGeneticAlgorithm import QuantumInspiredGeneticAlgorithmParams

if TYPE_CHECKING:
    from pyvrp._pyvrp import CostEvaluator, RandomNumberGenerator, Solution


class PopulationQCI:
    """
    Creates a Population instance for Qubits.

    Parameters
    ----------
    diversity_op
        Operator to use to determine pairwise diversity between solutions. Have
        a look at :mod:`pyvrp.diversity` for available operators.
    params
        Population parameters. If not provided, a default will be used.
    """

    def __init__(
        self,
        diversity_op: Callable[[Solution, Solution], float],
        params: PopulationParams | None = None,
        #algo_params: QuantumInspiredGeneticAlgorithmParams = QuantumInspiredGeneticAlgorithmParams(),
    ):
        self._op = diversity_op
        self._params = params if params is not None else PopulationParams()

        self._cols = 3 #algo_params.num_of_qubits

        self._tourn_idx = 0
        self._first_idx = 0
        self._second_idx = 0
        count = self._params.min_pop_size + self._params.generation_size
        count = math.ceil(count/self._cols)
        self._individual = np.zeros((count, 3, self._cols, 2), dtype = object)
        self._count = 0
        self._limit = count
        self._obsfeas = np.zeros(self._cols, dtype = object)
        self._obsinfeas = np.zeros(self._cols, dtype = object)
        self._feas = np.zeros((self._cols, 2), dtype = object)
        self._infeas = np.zeros((self._cols, 2), dtype = object)
        self._angles = np.zeros((self._cols, 2), dtype = object)

        #self._sols = SubPopulation(self._op, self._params)

        for c in range(self._cols):
            self._obsfeas[c] = SubPopulation(self._op, self._params)
            self._obsinfeas[c] = SubPopulation(self._op, self._params)
            for r in range(2):
                self._feas[c][r] = SubPopulation(self._op, self._params)
                self._infeas[c][r] = SubPopulation(self._op, self._params)

    def __iter__(self) -> Generator[Solution, None, None]:
        """
        Iterates over the solutions contained in this population.
        """
        for c in range(self._cols):
            for r in range(2):
                for item in self._feas[c][r]:
                    yield item.solution

        for c in range(self._cols):
            for r in range(2):
                for item in self._infeas[c][r]:
                    yield item.solution
    
    def __len__(self) -> int:
        """
        Returns the current population size.
        """
        thelen = 0
        for c in range(self._cols):
            for r in range(2):
                thelen+= len(self._feas[c][r]) + len(self._infeas[c][r])
        return thelen

    def _update_fitnessqci(self, cost_evaluator: CostEvaluator):
        """
        Updates the biased fitness values for the subpopulations.

        Parameters
        ----------
        cost_evaluator
            CostEvaluator to use for computing the fitness.
        """
        for c in range(self._cols):
            self._obsfeas[c].update_fitness(cost_evaluator)
            self._obsinfeas[c].update_fitness(cost_evaluator)
            for r in range(2):
                self._feas[c][r].update_fitness(cost_evaluator)
                self._infeas[c][r].update_fitness(cost_evaluator)

    def num_feasible(self) -> int:
        """
        Returns the number of feasible solutions in the population.
        """
        thelen = 0
        for c in range(self._cols):
            for r in range(2):
                thelen+= len(self._feas[c][r])
        return thelen

    def num_infeasible(self) -> int:
        """
        Returns the number of infeasible solutions in the population.
        """
        thelen = 0
        for c in range(self._cols):
            for r in range(2):
                thelen+= len(self._infeas[c][r])
        return thelen

    def addqci(self, solutions: Solution, cost_evaluators: CostEvaluator, q_angle, rng):
        """
        Inserts the given solution in the appropriate feasible or infeasible
        (sub)population.

        .. note::

           Survivor selection is automatically triggered when the subpopulation
           reaches its maximum size, given by
           :attr:`~pyvrp.Population.PopulationParams.max_pop_size`.

        Parameters
        ----------
        solution
            Solution to add to the population.
        cost_evaluator
            CostEvaluator to use to compute the cost.
        q_angle
            The angle for each qubits
        """

        # Note: the CostEvaluator is required here since adding a solution
        # may trigger a purge which needs to compute the biased fitness which
        # requires computing the cost.
        for c in range(self._cols):
            self.observe(solutions[c],cost_evaluators[c],q_angle[c],c,rng)
            for r in range(2):
                #self._sols.add(solutions[c][r], cost_evaluators[c][r])
                self._individual[self._count][0][c][r] = solutions[c][r]
                self._individual[self._count][1][c][r] = cost_evaluators[c][r]
                self._individual[self._count][2][c][r] = q_angle[c][r]
                if solutions[c][r].is_feasible():
                    # Note: the feasible subpopulation actually does not depend
                    # on the penalty values but we use the same implementation.
                    self._feas[c][r].add(solutions[c][r], cost_evaluators[c][r])
                else:
                    self._infeas[c][r].add(solutions[c][r], cost_evaluators[c][r])
                self._angles[c][r] = q_angle[c][r]
        
        self._count+=1
        if self._count>=self._limit:
            self._count = 0
    
    def observe(self, solutions: Solution, cost_evaluators: CostEvaluator, q_angle, arrnum, rng):
        """
        Selects the inidividual from the qubit based on the quantum gate
        """
        idx = rng.randint(1)
        if idx == 0 :
            if q_angle[0] < q_angle[1]:
                idx = 1
        else:
            if q_angle[1] < q_angle[0]:
                idx = 0

        if solutions[idx].is_feasible():
            self._obsfeas[arrnum].add(solutions[idx], cost_evaluators[idx])
        else:
            self._obsinfeas[arrnum].add(solutions[idx], cost_evaluators[idx])

    def clear(self):
        """
        Clears the population by removing all solutions currently in the
        population.
        """
        for c in range(self._cols):
            self._obsfeas[c] = SubPopulation(self._op, self._params)
            self._obsinfeas[c] = SubPopulation(self._op, self._params)
            for r in range(2):
                self._feas[c][r] = SubPopulation(self._op, self._params)
                self._infeas[c][r] = SubPopulation(self._op, self._params)

    def selectqci(
        self,
        rng: RandomNumberGenerator,
        cost_evaluator: CostEvaluator,
        k: int = 2,
    ): #-> tuple[Solution, Solution]:
        """
        Selects two (if possible non-identical) parents by tournament, subject
        to a diversity restriction.

        Parameters
        ----------
        rng
            Random number generator.
        cost_evaluator
            Cost evaluator to use when computing the fitness.
        k
            The number of solutions to draw for the tournament. Defaults to
            two, which results in a binary tournament.

        Returns
        -------
        tuple
            A solution pair (parents).
        """
        self._update_fitnessqci(cost_evaluator)

        #first = self._tournamentqci(rng, k)
        #self._first_idx = self._tourn_idx
        #second = self._tournamentqci(rng, k)
        #self._second_idx = self._tourn_idx

        #diversity = self._op(first, second)
        #lb = self._params.lb_diversity
        #ub = self._params.ub_diversity

        #tries = 1
        #while not (lb <= diversity <= ub) and tries <= 10:
        #    tries += 1
        #    second = self._tournamentqci(rng, k)
        #    self._second_idx = self._tourn_idx
        #    diversity = self._op(first, second)

        #first_num = math.ceil(self._first_idx/self._cols)
        #second_num = math.ceil(self._second_idx/self._cols)
        first_num = rng.randint(self._limit)
        while len(self._individual[first_num]) == 0 :
            first_num = rng.randint(self._limit)
        second_num = rng.randint(self._limit)
        while first_num == second_num or len(self._individual[second_num]) == 0:
            second_num = rng.randint(self._limit)
        return self._individual[first_num], self._individual[second_num]

    def tournamentqci(
        self,
        rng: RandomNumberGenerator,
        cost_evaluator: CostEvaluator,
        k: int = 2,
    ) -> Solution:
        """
        Selects a solution from this population by k-ary tournament, based
        on the (internal) fitness values of the selected solutions.

        Parameters
        ----------
        rng
            Random number generator.
        cost_evaluator
            Cost evaluator to use when computing the fitness.
        k
            The number of solutions to draw for the tournament. Defaults to
            two, which results in a binary tournament.

        Returns
        -------
        Solution
            The selected solution.
        """
        self._update_fitnessqci(cost_evaluator)
        return self._tournamentqci(rng, k)

    def _tournamentqci(self, rng: RandomNumberGenerator, k: int) -> Solution:
        if k <= 0:
            raise ValueError(f"Expected k > 0; got k = {k}.")

        def select():
            num_feas = 0
            for c in range(self._cols):
                for r in range(2):
                    num_feas+= len(self._feas[c][r])
                    
            #idx = rng.randint(len(self))
            idx = rng.randint(self.__len__())

            if idx < num_feas:
                curr_num = 0
                for c in range(self._cols):
                    for r in range(2):
                        if curr_num >= idx:
                            self._tourn_idx = idx
                            return self._feas[c][r][curr_num - idx]
                        curr_num+= len(self._feas[c][r])
                #return self._feas[idx]

            else :
                curr_innum = 0
                for c in range(self._cols):
                    for r in range(2):
                        if curr_innum >= idx - num_feas:
                            self._tourn_idx = idx - num_feas
                            return self._infeas[c][r][curr_innum - self._tourn_idx]
                        curr_innum += len(self._infeas[c][r])
            #return self._infeas[idx - num_feas]

        items = [select() for _ in range(k)]
        fittest = min(items, key=lambda item: item.fitness)
        return fittest.solution