"""
MaxSAT with bit-vector optimization based the following paper:
- SAT 2018: A. Nadel. Solving MaxSAT with bit-vector optimization. ("Mrs. Beaver")
https://www.researchgate.net/profile/Alexander-Nadel/publication/325970660_Solving_MaxSAT_with_Bit-Vector_Optimization/links/5b3a08fb4585150d23ee95df/Solving-MaxSAT-with-Bit-Vector-Optimization.pdf

Related
- FMCAD 19: Anytime Weighted MaxSAT with Improved Polarity Selection and Bit-Vector Optimization

FIXME: to validate (this file is generated by LLM)
"""

from typing import List, Optional, Tuple
import time
from pysat.solvers import Solver
from pysat.formula import CNF


class AnytimeMaxSAT:
    """Anytime MaxSAT solver using bit-vector optimization"""

    def __init__(self, hard: List[List[int]], soft: List[List[int]],
                 weights: Optional[List[int]] = None,
                 solver_name: str = 'glucose4'):
        """Initialize the solver with hard and soft clauses"""
        self.hard = hard
        self.soft = soft
        self.weights = weights if weights else [1] * len(soft)
        self.sat_engine_name = solver_name
        self.best_cost = sum(self.weights)
        self.best_model = None

    def _create_assumption_lits(self, bits: List[int], value: int) -> List[int]:
        """Create assumption literals for binary search"""
        assumptions = []
        for i, bit in enumerate(bits):
            if (value >> i) & 1:
                assumptions.append(bit)
            else:
                assumptions.append(-bit)
        return assumptions

    def solve(self, timeout: int = 300) -> Tuple[bool, Optional[List[int]], int]:
        """Solve MaxSAT problem with binary search
        
        Returns:
            (success, model, cost)
        """
        start_time = time.time()

        # Create SAT solver instance
        sat_oracle = Solver(name=self.sat_engine_name, bootstrap_with=self.hard)

        # Add soft clauses with selector variables
        selector_vars = []
        max_var = max(abs(lit) for clause in self.hard + self.soft
                      for lit in clause)
        next_var = max_var + 1

        for clause in self.soft:
            sel = next_var
            next_var += 1
            selector_vars.append(sel)
            sat_oracle.add_clause(clause + [-sel])

        # Binary search for optimal solution
        lb, ub = 0, sum(self.weights)
        best_model = None

        while lb < ub and (time.time() - start_time) < timeout:
            target = (lb + ub) // 2
            assumptions = self._create_assumption_lits(selector_vars, target)

            if sat_oracle.solve(assumptions=assumptions):
                # Found satisfying assignment with cost <= target
                model = sat_oracle.get_model()
                ub = target
                best_model = model
                self.best_model = model
                self.best_cost = target
            else:
                # No solution with cost <= target exists
                lb = target + 1

        success = best_model is not None
        return success, best_model, self.best_cost

    def get_solution(self) -> Tuple[Optional[List[int]], int]:
        """Get the best solution found so far"""
        return self.best_model, self.best_cost


def solve_maxsat(hard: List[List[int]], soft: List[List[int]],
                 weights: Optional[List[int]] = None,
                 timeout: int = 300) -> Tuple[bool, Optional[List[int]], int]:
    """Convenience function to solve MaxSAT problems"""
    solver = AnytimeMaxSAT(hard, soft, weights)
    return solver.solve(timeout)
