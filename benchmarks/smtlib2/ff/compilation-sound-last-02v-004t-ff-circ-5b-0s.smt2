; from Yices2
(set-info :smt-lib-version 2.6)
(set-info :category "crafted")
(set-info :status 'sat')
(set-logic QF_FFA)
(define-sort FF0 () (_ FiniteField 17))
(declare-fun a () Bool)
(declare-fun b () Bool)
(declare-fun return_n0 () FF0)
(declare-fun mul_n4 () FF0)
(declare-fun a_n2 () FF0)
(declare-fun b_n1 () FF0)
(declare-fun mul_n3 () FF0)
(assert 
  (let ((let0 a))
  (let ((let1 b))
  (let ((let2 (=> let1 let0)))
  (let ((let3 (not let2)))
  (let ((let4 (ite let2 let1 let0)))
  (let ((let5 (and let4 let3)))
  (let ((let6 return_n0))
  (let ((let7 (as ff1 FF0)))
  (let ((let8 (= let7 let6)))
  (let ((let9 (= let8 let5)))
  (let ((let10 (as ff0 FF0)))
  (let ((let11 (= let10 let6)))
  (let ((let12 (or let8 let11)))
  (let ((let13 (and let12 let9)))
  (let ((let14 mul_n4))
  (let ((let15 (as ff16 FF0)))
  (let ((let16 a_n2))
  (let ((let17 (ff.mul let16 let15)))
  (let ((let18 b_n1))
  (let ((let19 (ff.add let18 let17)))
  (let ((let20 mul_n3))
  (let ((let21 (ff.mul let20 let15)))
  (let ((let22 (ff.add let21 let7)))
  (let ((let23 (ff.mul let22 let19)))
  (let ((let24 (= let23 let14)))
  (let ((let25 (ff.add let17 let7)))
  (let ((let26 (ff.mul let18 let25)))
  (let ((let27 (= let26 let20)))
  (let ((let28 (and let27 let24)))
  (let ((let29 (ite let0 let7 let10)))
  (let ((let30 (= let16 let29)))
  (let ((let31 (ite let1 let7 let10)))
  (let ((let32 (= let18 let31)))
  (let ((let33 (and let32 let30)))
  (let ((let34 (and let33 let28)))
  (let ((let35 (=> let34 let13)))
  (let ((let36 (not let35)))
  let36
)))))))))))))))))))))))))))))))))))))
)
(check-sat)
