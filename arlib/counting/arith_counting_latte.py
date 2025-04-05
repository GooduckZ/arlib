"""
Model counting for linear integer arithmetic formulas using LattE.
LattE (Lattice point Enumeration) is a tool for counting integer points in polytopes.
See https://www.math.ucdavis.edu/~latte/

FIXME: to validate (this file is generated by LLM)
"""

import os
import subprocess
import tempfile
from typing import List, Optional
import z3
from pathlib import Path

from arlib.config import check_library
from arlib.utils.z3_expr_utils import get_variables


class LatteCounter:
    """Counter for linear integer arithmetic formulas using LattE"""

    def __init__(self, latte_path: Optional[str] = None):
        """Initialize the counter with optional path to LattE executable"""
        self.latte_path = latte_path or self._find_latte()
        if not self.latte_path:
            raise RuntimeError("LattE executable not found")

    def _find_latte(self) -> Optional[str]:
        """Find LattE executable in system path"""
        candidates = ['count', 'latte-count', 'latte-int']
        for cmd in candidates:
            path = subprocess.check_output(['which', cmd],
                                           stderr=subprocess.DEVNULL).strip()
            if path:
                return path.decode()
        return None

    def _formula_to_polytope(self, formula: z3.ExprRef) -> str:
        """FIXME: Convert Z3 formula to LattE polytope format"""
        # Extract variables and constraints
        vars = get_variables(formula)
        s = z3.Solver()
        s.add(formula)

        # Convert to polytope format
        # Format: 
        # dim
        # number_of_constraints
        # b -a1 -a2 ... -an
        # (for each constraint: a1x1 + a2x2 + ... + anxn <= b)

        polytope = []
        polytope.append(str(len(vars)))
        # TODO: extract linear constraints and convert to matrix form

        return "\n".join(polytope)

    def count_models(self, formula: z3.ExprRef) -> int:
        """Count models of a linear integer arithmetic formula"""
        # Check if formula is in the supported fragment
        if not self._is_lia_formula(formula):
            raise ValueError("Formula must be in linear integer arithmetic")

        # Create temporary file for polytope
        with tempfile.NamedTemporaryFile(mode='w', suffix='.hrep') as f:
            polytope = self._formula_to_polytope(formula)
            f.write(polytope)
            f.flush()

            # Run LattE
            try:
                output = subprocess.check_output(
                    [self.latte_path, f.name],
                    stderr=subprocess.PIPE
                )
                # Parse output to get count
                count = int(output.decode().strip())
                return count
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"LattE error: {e.stderr.decode()}")

    def _is_lia_formula(self, formula: z3.ExprRef) -> bool:
        """Check if formula is in linear integer arithmetic"""
        # TODO: implement proper check for LIA fragment
        return True


def count_lia_models(formula: z3.ExprRef) -> int:
    """Convenience function to count models of LIA formula"""
    counter = LatteCounter()
    return counter.count_models(formula)


if __name__ == "__main__":
    # Example usage
    x, y = z3.Ints('x y')
    formula = z3.And(x > 0, y < 10, x + y == 5)
    count = count_lia_models(formula)
    print(f"Number of models: {count}")
