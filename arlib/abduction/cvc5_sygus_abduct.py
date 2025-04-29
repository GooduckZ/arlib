"""CVC5 abduction interface.

- IJCAR20: Scalable algorithms for abduction via enumerative syntax-guided synthesis. Reynolds, A., Barbosa, H., Larraz, D., Tinelli, C.: 
- https://homepage.divms.uiowa.edu/~ajreynol/pres-sygus2023.pdf

Consider the following example:

(set-option :produce-abducts true)
(set-logic QF_LRA)
(declare-const x Real)
(declare-const y Real)
(declare-const z Real)
(assert (and (>= x 0.0) (< y 7.0)))
(get-abduct A (>= y 5.0))

在这个例子中，归纳推理试图找到一个约束A，使得：
- A与当前的断言相容（可满足）
- A与现有约束一起能够推出y ≥ 5.0

It seems that we can also specify a grammar for the abduct, e.g.:
(get-abduct <conj> <grammar>)
"""

import z3
import tempfile
import subprocess
import os
from typing import Tuple, List
from arlib.abduction.utils import get_variables


def z3_to_smtlib2_abduction(formula: z3.ExprRef, target_var: str) -> str:
    """Convert Z3 formula to SMT-LIB2 format with abduction syntax."""
    base_smt = formula.sexpr()
    logic = "QF_LIA"
    
    # Extract variable declarations
    var_decls = []
    for var_name, var_type in get_variables(formula).items():
        var_decls.append(f"(declare-fun {var_name} () {var_type})")

    return f"""(set-logic {logic})
{chr(10).join(var_decls)}
(assert true)
(get-abduct {target_var} {base_smt})
"""


def solve_abduction(formula: z3.ExprRef) -> Tuple[bool, str]:
    """Solve abduction problem using CVC5."""
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as tmp_file:
        smt2_content = z3_to_smtlib2_abduction(formula, "A")
        tmp_file.write(smt2_content)
        tmp_path = tmp_file.name
    
    try:
        # Get CVC5 path
        try:
            from arlib.global_params.paths import global_config
            cvc5_path = global_config.get_solver_path("cvc5")
        except (ImportError, AttributeError):
            cvc5_path = "cvc5" # try system cvc5
            if subprocess.run(["which", "cvc5"], capture_output=True).returncode != 0:
                return False, "CVC5 not found in PATH"
        
        # Run CVC5
        cmd = [cvc5_path, '--produce-abducts', '--sygus-inference', tmp_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Handle result
        if result.returncode != 0:
            return False, f"CVC5 error (code {result.returncode})"
        
        output = result.stdout.strip()
        if not output or 'unsat' in output.lower():
            return False, "No solution found"
        return True, output

    except Exception as e:
        return False, f"Error: {str(e)}"
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass


# Example usage
if __name__ == "__main__":
    x = z3.Int('x')
    y = z3.Int('y')
    goal = z3.And(x > 0, y > x)
    
    print("Finding formula A such that A implies (x > 0 and y > x)")
    success, result = solve_abduction(goal)
    
    if success:
        print("Found abduction solution:", result)
    else:
        print("Failed:", result)
        