import arlib.itp.smt as smt
import arlib.itp as itp

# https://en.wikipedia.org/wiki/Extended_real_number_line
# https://isabelle.in.tum.de/library/HOL/HOL-Library/Extended_Real.html

EReal = smt.Datatype("EReal")
EReal.declare("Real", ("val", smt.RealSort()))
EReal.declare("Inf")
EReal.declare("NegInf")
# EReal.declare("NaN")
EReal = EReal.create()

x, y, z = smt.Consts("x y z", EReal)

le = itp.notation.le.define(
    [x, y],
    itp.cond(
        (x == y, smt.BoolVal(True)),
        (x.is_NegInf, smt.BoolVal(True)),
        (y.is_Inf, smt.BoolVal(True)),
        (y.is_NegInf, smt.BoolVal(False)),
        (x.is_Inf, smt.BoolVal(False)),
        (smt.And(x.is_Real, y.is_Real), x.val <= y.val),
    ),
)

le_refl = itp.prove(itp.QForAll([x], x <= x), by=[le.defn])
le_trans = itp.prove(itp.QForAll([x, y, z], x <= y, y <= z, x <= z), by=[le.defn])
le_total = itp.prove(smt.ForAll([x, y], smt.Or(x <= y, y <= x)), by=[le.defn])

add_undef = smt.Function("add_undef", EReal, EReal, EReal)
add = itp.notation.add.define(
    [x, y],
    itp.cond(
        (smt.And(x.is_Real, y.is_Real), EReal.Real(x.val + y.val)),
        (smt.And(x.is_Inf, smt.Not(y.is_NegInf)), EReal.Inf),
        (smt.And(smt.Not(x.is_NegInf), y.is_Inf), EReal.Inf),
        (smt.And(x.is_NegInf, smt.Not(y.is_Inf)), EReal.NegInf),
        (smt.And(smt.Not(x.is_Inf), y.is_NegInf), EReal.NegInf),
        default=add_undef(x, y),
    ),
)
add_defined = itp.define(
    "add_defined",
    [x, y],
    smt.Or(
        smt.And(x.is_Real, y.is_Real),
        smt.And(x.is_Inf, smt.Not(y.is_NegInf)),
        smt.And(smt.Not(x.is_NegInf), y.is_Inf),
        smt.And(x.is_NegInf, smt.Not(y.is_Inf)),
        smt.And(smt.Not(x.is_Inf), y.is_NegInf),
    ),
)

defined_undef = itp.prove(
    itp.QForAll([x, y], smt.Not(add_defined(x, y)), x + y == add_undef(x, y)),
    by=[add_defined.defn, add.defn],
)

add_comm = itp.prove(
    itp.QForAll([x, y], add_defined(x, y), x + y == y + x),
    by=[add.defn, add_defined.defn],
)

add_comm1 = itp.prove(
    itp.QForAll([x, y], add_undef(x, y) == add_undef(y, x), x + y == y + x),
    by=[add.defn],
)

EPosReal = smt.Datatype("EPosReal")
EPosReal.declare("real", ("val", smt.RealSort()))
EPosReal.declare("inf")
EPosReal = EPosReal.create()
x_p = smt.Const("x", EPosReal)
itp.notation.wf.define([x_p], smt.Implies(x_p.is_real, x_p.val >= 0))
