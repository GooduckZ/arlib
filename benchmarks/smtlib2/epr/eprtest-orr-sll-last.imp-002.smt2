(set-option :print-success false)
(set-option :produce-proofs false)

(set-logic UF)
(set-info :source |Second push block from orr/sll-last.imp.smt2|)
(set-info :smt-lib-version 2.0)
(set-info :category "crafted")
(set-info :status unsat)

(declare-sort V 0)
(declare-fun i () V)
(declare-fun h () V)
(declare-fun n* (V V) Bool)
(declare-fun null () V)
(declare-fun j () V)

(declare-fun EQ (V V) Bool)
(assert (forall ((x V)) (EQ x x)))
(assert (forall ((x V) (y V)) (=> (EQ x y) (EQ y x))))
(assert (forall ((x V) (y V) (z V)) (=> (and (EQ x y) (EQ y z)) (EQ x z))))
(assert (forall ((x0 V) (y0 V) (x1 V) (y1 V)) (=> (and (EQ x0 y0) (EQ x1 y1)) (=> (n* x0 x1) (n* y0 y1)))))

(assert (forall ((u$1$1 V)) (n* u$1$1 u$1$1)))
(assert (forall ((u$2$1 V) (v$1$1 V) (w$1$1 V)) (=> (and (n* u$2$1 v$1$1) (n* v$1$1 w$1$1)) (n* u$2$1 w$1$1))))
(assert (forall ((u$3$1 V) (v$2$1 V) (w$2$1 V)) (=> (and (n* u$3$1 v$2$1) (n* u$3$1 w$2$1)) (or (n* v$2$1 w$2$1) (n* w$2$1 v$2$1)))))
(assert (forall ((u$4$1 V) (v$3$1 V)) (=> (n* u$4$1 v$3$1) (=> (n* v$3$1 u$4$1) (EQ u$4$1 v$3$1)))))
(assert (forall ((v$4$1 V)) (=> (or (n* null v$4$1) (n* v$4$1 null)) (EQ null v$4$1))))

(assert (not (and (=> (not (EQ h null)) (n* h h)) (ite (EQ null null) (EQ h h) (and (n* h null) (or (and (n* null h) (not (EQ null h)) (forall ((w$11$1 V)) (=> (and (n* null w$11$1) (not (EQ null w$11$1))) (n* h w$11$1)))) (and (EQ h null) (forall ((w$12$1 V)) (not (and (n* null w$12$1) (not (EQ null w$12$1))))))))) true)))(check-sat)

(exit)
