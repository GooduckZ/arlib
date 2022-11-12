"""
Solving Exits-Forall Problem (currently focus on bit-vec?)
https://github.com/pysmt/pysmt/blob/97088bf3b0d64137c3099ef79a4e153b10ccfda7/examples/efsmt.py
Possible extensions:
- better generalizaton for esolver
- better generalizaton for fsolver
- uniform sampling for processing multiple models each round?
- use unsat core??
However, the counterexample may not be general enough to exclude a large class of invalid expressions,
which will lead to the repetition of several loop iterations. We believe our sampling technique could
be a good enhancement to CEGIS. By generating several diverse counterexamples, the verifier can
provide more information to the learner so that it can make more progress on its own,
limiting the number of calls to the verifier
"""

import logging
import time
from arlib.efsmt.efbv.efbv_forall_solver import ForAllSolver
from arlib.efsmt.efbv.efbv_exists_solver import ExistsSolver
from arlib.efsmt.efbv.efbv_utils import EFBVResult
from arlib.utils.exceptions import ExitsSolverSuccess, ExitsSolverUnknown, ForAllSolverSuccess, ForAllSolverUnknown

import z3
from z3.z3util import get_vars
from z3 import *

logger = logging.getLogger(__name__)


def bv_efsmt_with_uniform_sampling(exists_vars, forall_vars, phi, maxloops=None):
    """
    Solves exists x. forall y. phi(x, y)
    FIXME: inconsistent with efsmt
    """
    # x = [item for item in get_vars(phi) if item not in y]

    esolver = ExistsSolver(exists_vars, z3.BoolVal(True))
    fsolver = ForAllSolver()
    fsolver.vars = forall_vars
    fsolver.phi = phi

    loops = 0
    result = EFBVResult.UNKNOWN
    try:
        while maxloops is None or loops <= maxloops:
            logger.debug("  Round: {}".format(loops))
            loops += 1
            # TODO: need to make the fist and the subsequent iteration different???
            # TODO: in the uniform sampler, I always call the solver once before xx...
            e_models = esolver.get_models(5)

            if len(e_models) == 0:
                logger.debug("  Success with UNSAT")
                result = EFBVResult.UNSAT  # esolver tells unsat
                break
            else:
                # sub_phis = []
                reverse_sub_phis = []
                print("e models: ", e_models)
                for emodel in e_models:
                    sub_phi = z3.substitute(phi, [(x, emodel.eval(x, True)) for x in exists_vars])
                    # sub_phis.append(sub_phi)
                    reverse_sub_phis.append(z3.Not(sub_phi))
                blocking_fml = fsolver.get_blocking_fml(reverse_sub_phis)
                if z3.is_false(blocking_fml):  # At least one Not(sub_phi) is UNSAT
                    logger.debug("  Success with SAT")
                    result = EFBVResult.SAT  # fsolver tells sat
                    break
                # block all CEX?
                esolver.fmls.append(blocking_fml)
    except ForAllSolverSuccess as ex:
        # print(ex)
        logger.debug("  Forall solver SAT")
        result = EFBVResult.SAT
    except ForAllSolverUnknown as ex:
        logger.debug("  Forall solver UNKNOWN")
        result = EFBVResult.UNKNOWN
    except ExitsSolverSuccess as ex:
        logger.debug("  Exists solver UNSAT")
        result = EFBVResult.UNSAT
    except ExitsSolverUnknown as ex:
        logger.debug("  Exists solver UNKNOWN")
        result = EFBVResult.UNKNOWN
    except Exception as ex:
        print("XX")

    return result


def test_efsmt():
    x, y, z = z3.BitVecs("x y z", 16)
    fmla = z3.Implies(z3.And(y > 0, y < 10), y - 2 * x < 7)
    # '''
    start = time.time()
    print(bv_efsmt_with_uniform_sampling([x], [y], fmla, 100))
    print(time.time() - start)


def test():
    """
    FIXME: crash in parallel mode
    """
    w, x, y = z3.BitVecs("w x y", 3)
    fml = And(Or(7 >= x, 5 > y, And(3 < w, 5 == y), 5 >= x),
              Or(3 < w, 5 > y, 5 == y, 5 >= x),
              Or(3 <= x, 5 < 7, Xor(5 >= x, 5 < 7)), 5 < w,
              Or(Xor(5 >= x, 5 < 7), 3 < w, 7 >= x, 3 <= x, 5 == y, And(3 < w, 5 == y)),
              Or(And(3 < w, 5 == y), 3 <= x, 5 <= 7, 3 < w, Xor(5 >= x, 5 < 7), 5 > y, 5 == y),
              Or(5 < w, And(3 < w, 5 == y), 5 <= 7, 5 < 7, 5 > y, 7 >= x, 5 == y),
              Or(5 > y, 3 <= x, 5 < w, 3 < w, 5 <= 7),
              Or(5 > y, 5 <= 7, 5 == y, 3 < w, And(3 < w, 5 == y), Xor(5 >= x, 5 < 7), 5 >= x),
              Or(5 > y, 5 < w, 3 < w, 5 < 7, Xor(5 >= x, 5 < 7), And(3 < w, 5 == y), 7 >= x, 5 == y),
              Or(5 < 7, 5 == y, 5 > y, 3 <= x),
              Or(5 <= 7, 3 <= x, 5 < 7, And(3 < w, 5 == y), 5 > y, 3 < w, 5 >= x),
              Or(3 <= x, 5 <= 7, 5 < w),
              Or(And(3 < w, 5 == y), Xor(5 >= x, 5 < 7)),
              Or(3 <= x, Xor(5 >= x, 5 < 7), 5 > y, 5 < w, 5 < 7, 5 == y, 7 >= x, 5 >= x),
              Or(Xor(5 >= x, 5 < 7), 5 < 7, 3 < w, 5 <= 7, And(3 < w, 5 == y), 7 >= x),
              Or(5 < w, 5 <= 7, 5 < 7, 5 == y, 5 > y, 3 <= x),
              Or(5 < w, 5 > y, And(3 < w, 5 == y), 7 >= x, 5 < 7, 3 <= x, 5 >= x),
              Or(5 >= x, 5 <= 7, And(3 < w, 5 == y), 7 >= x))

    print(bv_efsmt_with_uniform_sampling([w, y], [x], fml, 100))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # test_efsmt()
    test()
