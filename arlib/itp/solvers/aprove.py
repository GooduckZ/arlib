import arlib.itp.smt as smt
import arlib.itp.solvers

"""
AProVE is a termination prover for term rewrite systems.
https://aprove.informatik.rwth-aachen.de/

"""


def run_aprove(vs: list[smt.ExprRef], eqs: list[smt.BoolRef], timeout=1):
    def pp(e):
        return arlib.itp.solvers.expr_to_tptp(e, format="fof", theories=False)

    with open("/tmp/aprove.trs", "w") as f:
        f.write(f"(VARS {' '.join(pp(v) for v in vs)})\n")
        f.write("(RULES\n")
        for eq in eqs:
            assert smt.is_eq(eq)
            f.write(f"{pp(eq.arg(0))} -> {pp(eq.arg(1))}\n")
        f.write(")\n")
        f.flush()
    import subprocess

    res = subprocess.run(
        [
            "java",
            "-ea",
            "-jar",
            arlib.itp.solvers.binpath("aprove.jar"),
            "-m",
            "wst",
            "-t",
            str(timeout),
            "-p",
            "plain",
            "/tmp/aprove.trs",
        ],
        check=True,
    )
    return res
