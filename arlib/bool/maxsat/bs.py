"""
This is a binary search algorithm of bit-vector optimization.
The OBV-BS algorithm from TACAS 16
     Bit-Vector Optimization, Alex Navel.
"""

from pysat.solvers import Solver


def obv_bs(clauses, literals):
    """
    This is a binary search algorithm of bit-vector optimization.
    Args:
        clauses: the given constraints
        literals: literals listed in priority

    Returns: the maximum assignment of literals

    """
    result = []
    s = Solver(bootstrap_with=clauses)
    if s.solve():
        m = s.get_model()
        # print(m)
    else:
        print('UNSAT')
        return result
    l = len(m)
    for lit in literals:
        if lit > l:
            '''If 'lit' is not in m, 'lit' can be assigned 0 or 1, to maximum the result, 'lit' is assigned 1.'''
            result.append(lit)
        else:
            # if lit is true in m, then lit is assigned 1
            if m[lit - 1] > 0:
                result.append(lit)
            else:
                # if lit is false in m, then lit is assigned 0
                result.append(lit)
                # update the assumptions and solve again
                if s.solve(assumptions=result):
                    m = s.get_model()
                else:
                    result.pop()
                    result.append(-lit)
    # print(result)
    return result
