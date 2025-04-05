"""
FIXME: generated by LLM (to check)
"""
import z3
from z3 import is_true, BoolVal, Bool, And, Or, Not
import functools
from typing import List, Tuple, Optional, Dict, Union
import re

foldr = lambda func, acc, xs: functools.reduce(lambda x, y: func(y, x), xs[::-1], acc)


def z3_val_to_int(z3_val):
    return 1 if is_true(z3_val) else 0


def int_vec_to_z3(int_vec):
    return [BoolVal(val == 1) for val in int_vec]


q_to_z3 = {1: z3.ForAll, -1: z3.Exists}


class QDIMACSParser:
    def __init__(self):
        self.vars: Dict[int, Bool] = {}
        self.num_vars = 0
        self.num_clauses = 0

    def get_var(self, var_id: int) -> Bool:
        """Get or create a Z3 Bool variable for a given DIMACS variable ID"""
        abs_id = abs(var_id)
        if abs_id not in self.vars:
            self.vars[abs_id] = Bool(f"x_{abs_id}")
        return self.vars[abs_id]

    def parse_qdimacs(self, qdimacs_str: str) -> 'QBF':
        lines = [line.strip() for line in qdimacs_str.splitlines()]
        lines = [line for line in lines if line and not line.startswith('c')]

        # Parse header
        header = lines[0].split()
        assert header[0] == 'p' and header[1] == 'cnf'
        self.num_vars = int(header[2])
        self.num_clauses = int(header[3])

        # Parse quantifiers and clauses
        q_list = []
        clauses = []

        current_line = 1
        # Parse quantifier lines
        while current_line < len(lines) and (
                lines[current_line].startswith('a') or lines[current_line].startswith('e')):
            line = lines[current_line].split()
            quant_type = 1 if line[0] == 'a' else -1  # 'a' for ForAll, 'e' for Exists
            # Convert variable IDs to Z3 Bool variables, stopping at 0
            vars = []
            for var_id in line[1:]:
                if var_id == '0':
                    break
                vars.append(self.get_var(int(var_id)))
            if vars:  # Only add if there are variables
                q_list.append((quant_type, vars))
            current_line += 1

        # Parse clauses
        while current_line < len(lines):
            clause = []
            literals = lines[current_line].split()
            for lit in literals:
                lit_val = int(lit)
                if lit_val == 0:
                    continue
                var = self.get_var(abs(lit_val))
                clause.append(Not(var) if lit_val < 0 else var)
            if clause:  # Only add non-empty clauses
                clauses.append(Or(clause))
            current_line += 1

        # Create the final formula (conjunction of all clauses)
        prop_formula = And(clauses)
        return QBF(prop_formula, q_list)


class QBF:

    def __init__(self, prop_formula, q_list=None):
        super(QBF, self).__init__()
        if q_list is None:
            q_list = []
        self._q_list = q_list
        self._prop = prop_formula

    def get_prop(self):
        return self._prop

    def get_q_list(self):
        return self._q_list

    def to_z3(self):
        return foldr(lambda q_v, f: q_to_z3[q_v[0]](q_v[1], f), self._prop, self._q_list)

    def negate(self):
        new_q_list = [(-_q, _v) for (_q, _v) in self._q_list]
        return QBF(self._prop.children()[0] if z3.is_not(self._prop) else z3.Not(self._prop), new_q_list)

    def well_named(self):
        q_list = self.get_q_list()
        appeared = set()
        for _, var_vec in q_list:
            for _v in var_vec:
                if str(_v) in appeared:
                    return False
                appeared.add(str(_v))
        return True

    def solve(self) -> Optional[Dict[str, bool]]:
        """
        Solve the QBF formula and return a model.
        """
        solver = z3.Solver()
        formula = self.to_z3()
        solver.add(formula)

        if solver.check() == z3.sat:
            model = solver.model()
            result = {}

            # Extract values for all variables
            for _, vars in self._q_list:
                for v in vars:
                    val = model.eval(v, model_completion=True)
                    result[str(v)] = bool(is_true(val))

            return result
        return None

    def solve_with_skolem(self) -> Optional[Dict[str, bool]]:
        """
        Alternative solving method that tries to find Skolem functions.
        """
        solver = Solver()
        universal_vars = []
        existential_vars = []

        # Separate universal and existential variables
        for quant, vars in self._q_list:
            if quant == 1:  # Universal
                universal_vars.extend(vars)
            else:  # Existential
                existential_vars.extend(vars)

        # Try to find values for existential variables
        result = {}

        # Add the formula
        solver.add(self._prop)

        if solver.check() == sat:
            model = solver.model()

            # Get values for existential variables
            for v in existential_vars:
                val = model.eval(v, model_completion=True)
                result[str(v)] = bool(is_true(val))

            # For universal variables, we can assign arbitrary values
            for v in universal_vars:
                val = model.eval(v, model_completion=True)
                result[str(v)] = bool(is_true(val))

            return result
        return None

    def __eq__(self, o):
        return self._prop.eq(o.get_prop()) and self._q_list == o.get_q_list()

    def __ne__(self, o):
        return not self == o

    def __hash__(self):
        return hash((hash(self._prop), hash(tuple(self._q_list))))


def demo():
    # Test QDIMACS parsing
    qdimacs_str = """
    c Example QDIMACS file
    p cnf 4 2
    a 1 2 0
    e 3 4 0
    1 2 3 0
    -1 -2 4 0
    """

    parser = QDIMACSParser()
    qbf = parser.parse_qdimacs(qdimacs_str)

    # Test solving
    result = qbf.solve()
    print(f"QBF Solution: {result}")


if __name__ == "__main__":
    demo()
