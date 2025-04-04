"""
Theory of regular expressions
"""

import arlib.itp as itp
import arlib.itp.smt as smt
import functools

"""
Some of the z3 builtins on Regular Expressions:

ReStr = smt.ReSort(smt.SeqSort(smt.IntSort()))
r1 = smt.Full(ReStr)
smt.AllChar(ReStr)
smt.Complement(r1)
smt.Concat(r1, r1)
smt.Diff(r1, r1)
smt.Empty(ReStr)
re = smt.Union(smt.Re("a"),smt.Re("b"))
smt.simplify(smt.InRe("a", re))
smt.Intersect(re, re)
smt.Loop(re, 0, 10)
smt.Option(re)
smt.Plus(re)
smt.Range("a", "z")
smt.Re("abc") # literal match
smt.Re(smt.Unit(smt.BoolVal(True)))
smt.Star(re)
"""


@functools.cache
def ReSort(S: smt.SortRef) -> smt.ReSortRef:
    """
    Regular expression sort. Sort parameter needs to be a sequence sort.
    https://en.wikipedia.org/wiki/Kleene_algebra

    >>> T = ReSort(smt.SeqSort(smt.IntSort()))
    >>> x,y,z = smt.Consts("x y z", T)
    >>> x + y
    Union(x, y)
    >>> x | y
    Union(x, y)
    >>> x * y
    re.++(x, y)
    >>> x & y
    Intersect(x, y)
    >>> ~x
    Complement(x)
    >>> x[smt.Unit(smt.IntVal(0))] # A regular expression is something like a predicate on sequences
    InRe(Unit(0), x)
    >>> x - y
    re.diff(x, y)
    """
    T = smt.ReSort(S)
    empty = smt.Empty(T)
    zero = empty
    eps = smt.Re(smt.Empty(S))  # empty string acceptor
    one = eps
    T.empty = empty
    T.full = smt.Full(T)
    # ReRef already defines + to be Union
    itp.notation.mul.register(T, lambda x, y: smt.Concat(x, y))
    itp.notation.or_.register(T, lambda x, y: smt.Union(x, y))
    itp.notation.and_.register(T, lambda x, y: smt.Intersect(x, y))
    itp.notation.invert.register(T, lambda x: smt.Complement(x))
    itp.notation.getitem.register(T, lambda x, i: smt.InRe(i, x))
    itp.notation.sub.register(T, lambda x, y: smt.Diff(x, y))

    x, y, z = smt.Consts("x y z", T)
    T.add_comm = itp.prove(smt.ForAll([x, y], x + y == y + x))
    T.add_assoc = itp.prove(smt.ForAll([x, y, z], (x + y) + z == x + (y + z)))
    # TODO: failing. Wrong or needs to be axiomatized?
    # T.concat_assoc = itp.prove(smt.ForAll([x, y, z], (x * y) * z == x * (y * z)))
    T.add_zero = itp.prove(smt.ForAll([x], x + zero == x))
    T.add_zero_left = itp.prove(smt.ForAll([x], zero + x == x))
    T.add_idem = itp.prove(smt.ForAll([x], x + x == x))
    T.mul_zero = itp.prove(smt.ForAll([x], x * zero == zero))
    # T.distrib = itp.prove(smt.ForAll([x, y, z], x * (y + z) == x * y + x * z))

    T.concat_one = itp.prove(smt.ForAll([x], x * one == x))

    T.zero_star = itp.prove(smt.Star(zero) == one)
    T.one_star = itp.prove(smt.Star(one) == one)
    T.star_concat = itp.prove(smt.ForAll([x], smt.Star(x) * smt.Star(x) == smt.Star(x)))
    T.star_star = itp.prove(smt.ForAll([x], smt.Star(smt.Star(x)) == smt.Star(x)))
    T.star_unfold = itp.prove(smt.ForAll([x], one + x * smt.Star(x) == smt.Star(x)))
    T.star_unfold2 = itp.prove(smt.ForAll([x], one + smt.Star(x) * x == smt.Star(x)))

    T.option_defn = itp.prove(smt.ForAll([x], smt.Option(x) == one + x))

    T.le = itp.notation.le.define([x, y], y + x == y)
    T.le_refl = itp.prove(smt.ForAll([x], x <= x), by=[T.le.defn])

    # T.le_trans = itp.prove(
    #    itp.QForAll([x, y, z], x <= y, y <= z, x <= z),
    #    by=[T.le.defn],
    # )

    a = smt.Const("a", S)
    T.to_set = itp.define("to_set", [x], smt.Lambda([a], x[a]))

    # T.union_all_char = itp.prove(smt.ForAll([a], smt.AllChar(T) ==
    # smt.

    return T
