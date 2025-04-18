import arlib.itp as itp
import arlib.itp.smt as smt
import arlib.itp.theories.set as set_
import functools

# https://plato.stanford.edu/Entries/logic-intuitionistic/#FormSystMathMath
# https://en.wikipedia.org/wiki/Kripke_semantics#Semantics_of_intuitionistic_logic
# def modus(A : smt.BoolRef, AB : smt.BoolRef) -> itp.Proof:
#    return itp.axiom(smt.Implies(A, smt.Implies(AB, A)))

"""
A different approach. Direct axiomatization of an uninterpreted sort.
Prop = smt.DeclareSort("Prop")
A, B = smt.Consts("A B", Prop)
Implies = smt.Function("Implies", Prop, Prop, Prop)
And = smt.Function("And", Prop, Prop, Prop)
Or = smt.Function("Or", Prop, Prop, Prop)
Not = smt.Function("Not", Prop, Prop)
modus = itp.axiom(itp.QForAll([A, B], Implies(A, Implies(A, B), B)))

Another approach might be to make a datatype of intuitionistic syntax trees.

"""

World = smt.DeclareSort("World")
w, u, v = smt.Consts("w u v", World)
acc = smt.Function("acc", World, World, smt.BoolSort())
# acc0 = smt.Function("acc0", World, smt.BoolSort())
# accplus = smt.TransitiveClosure(acc0)
# acc = smt.Lambda([w,u], smt.Or(w == u, accplus(w,u)))
acc_refl = itp.axiom(smt.ForAll([w], acc(w, w)))
acc_trans = itp.axiom(itp.QForAll([w, u, v], acc(w, u), acc(u, v), acc(w, v)))

Prop = itp.NewType(
    "Prop",
    smt.ArraySort(World, smt.BoolSort()),
    pred=lambda p: itp.QForAll([w, u], acc(w, u), p.val[w], p.val[u]),
)
"""
A proposition is a world valuation function. Propositions become monotonically more true as we move to more accessible worlds.
Note that Prop ~ Sort(Unit)
"""


def And(*ps: smt.DatatypeRef) -> smt.DatatypeRef:
    """
    w |= (A /\\ B)[e] if and only if w |= A[e] and w |= B[e]

    >>> p, q = smt.Consts("p q", Prop)
    >>> And(p,q)
    Prop(Lambda(w, And(val(p)[w], val(q)[w])))
    """
    return Prop(smt.Lambda([w], smt.And(*[p.val[w] for p in ps])))


def Or(*ps: smt.DatatypeRef) -> smt.DatatypeRef:
    """
    w |= (A \\/ B)[e] if and only if w |= A[e] or w |= B[e]

    >>> p, q = smt.Consts("p q", Prop)
    >>> Or(p,q)
    Prop(Lambda(w, Or(val(p)[w], val(q)[w])))
    """
    return Prop(smt.Lambda([w], smt.Or(*[p.val[w] for p in ps])))


def Implies(p: smt.DatatypeRef, q: smt.DatatypeRef) -> smt.DatatypeRef:
    return Prop(
        smt.Lambda([w], itp.QForAll([u], acc(w, u), smt.Implies(p.val[u], q.val[u])))
    )


TRUE = Prop(smt.K(World, smt.BoolVal(True)))
FALSE = Prop(smt.K(World, smt.BoolVal(False)))


def Not(p: smt.DatatypeRef) -> smt.DatatypeRef:
    return Implies(p, FALSE)


def Valid(p: smt.DatatypeRef) -> smt.BoolRef:
    return smt.ForAll([w], p.val[w])


@functools.cache
def Sort(sort: smt.SortRef):
    return itp.NewType(
        f"I_{sort}",
        smt.ArraySort(World, set_.Set(sort)),
        pred=lambda x: itp.QForAll([w, u], acc(w, u), x.val[w] <= x.val[u]),
    )


def Const(name: str, sort: smt.SortRef) -> smt.DatatypeRef:
    raise NotImplementedError


# def Exists(xs, body):

a, b, c = smt.Consts("a b c", Prop)
and_ = itp.define("iand", [a, b], And(a, b))
or_ = itp.define("ior", [a, b], Or(a, b))
impl_ = itp.define("iimpl", [a, b], Implies(a, b))
not_ = itp.define("inot", [a], Not(a))
valid = itp.define("valid", [a], Valid(a))

itp.notation.and_.register(Prop, and_)
itp.notation.or_.register(Prop, or_)
itp.notation.invert.register(Prop, not_)

impl_aba = itp.prove(itp.QForAll([a, b], valid(impl_(a, impl_(b, a)))), unfold=1)

impl_aba = itp.prove(itp.QForAll([a, b], Valid(Implies(a, Implies(b, a)))))
and_elim1 = itp.prove(itp.QForAll([a, b], Valid(Implies(And(a, b), a))))
and_elim2 = itp.prove(itp.QForAll([a, b], Valid(Implies(And(a, b), b))))
or_intro1 = itp.prove(itp.QForAll([a, b], Valid(Implies(a, Or(a, b)))))
or_intro2 = itp.prove(itp.QForAll([a, b], Valid(Implies(b, Or(a, b)))))
# fails dne = itp.prove(itp.QForAll([a], Valid(Implies(Not(Not(a)), a))))

# Non theorems. Raise errors. See Tests

# Mmm. Maybe this isn't enough to show a non provability?
# excluded_middle = itp.prove(
#    smt.Not(itp.QForAll([a], Valid(Or(a, Not(a))))), by=[acc_refl, acc_trans]
# )
# dne = itp.prove(itp.QForAll([a], Valid(Implies(Not(Not(a)), a))))

"""
Finite model property + 
"""
