import random as rand
from timeit import default_timer as timer

n = 5

## this is the example for matrix ordering given in the docs
#m = matrix(2, [2,3,0,1])
#T = TermOrder(m)
#P = PolynomialRing(QQ, 'x', 2, order=T)
#P.inject_variables()
#
#print(x0^3 > x1^2)
#print(x0 < x1)
#
## now if we try doing it with n vars and a random matrix it dies?
#m = matrix(n, [rand.randint(0,1000) for _ in range(n*n)]) # nxn matrix, random entries in [0,1]
#T = TermOrder(m)
#P = PolynomialRing(QQ, 'x', n, order=T)
#P.inject_variables()
## now these two expressions will ALWAYS evaluate to True?????
#print(x0^3 > x1^2)
#print(x0 > x1)

data = {}
for i in range(1000):
    A = matrix(n, [rand.randint(0, 100000) for _ in range(n*n)], immutable=True)
    T = TermOrder(A)
    var(['x'+str(i) for i in range(n)])
    R = PolynomialRing(QQ, 'x', n, order=T)
    # print(T)
    # print(x0 > x1)
    # print(R)
    # print(TermOrder(A))
    # Define the polynomial ring with 8 variables over the rationals
    R = PolynomialRing(QQ, names=['x0','x1','x2','x3','x4','x5','x6','x7'])
    x0, x1, x2, x3, x4, x5, x6, x7 = R.gens()

    # Define the generators of the Cyclic(8) ideal
    f1 = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
    f2 = x0*x1 + x1*x2 + x2*x3 + x3*x4 + x4*x5 + x5*x6 + x6*x7 + x7*x0
    f3 = x0*x1*x2 + x1*x2*x3 + x2*x3*x4 + x3*x4*x5 + x4*x5*x6 + x5*x6*x7 + x6*x7*x0 + x7*x0*x1
    f4 = x0*x1*x2*x3 + x1*x2*x3*x4 + x2*x3*x4*x5 + x3*x4*x5*x6 + x4*x5*x6*x7 + x5*x6*x7*x0 + x6*x7*x0*x1 + x7*x0*x1*x2
    f5 = x0*x1*x2*x3*x4 + x1*x2*x3*x4*x5 + x2*x3*x4*x5*x6 + x3*x4*x5*x6*x7 + x4*x5*x6*x7*x0 + x5*x6*x7*x0*x1 + x6*x7*x0*x1*x2 + x7*x0*x1*x2*x3
    f6 = x0*x1*x2*x3*x4*x5 + x1*x2*x3*x4*x5*x6 + x2*x3*x4*x5*x6*x7 + x3*x4*x5*x6*x7*x0 + x4*x5*x6*x7*x0*x1 + x5*x6*x7*x0*x1*x2 + x6*x7*x0*x1*x2*x3 + x7*x0*x1*x2*x3*x4
    f7 = x0*x1*x2*x3*x4*x5*x6 + x1*x2*x3*x4*x5*x6*x7 + x2*x3*x4*x5*x6*x7*x0 + x3*x4*x5*x6*x7*x0*x1 + x4*x5*x6*x7*x0*x1*x2 + x5*x6*x7*x0*x1*x2*x3 + x6*x7*x0*x1*x2*x3*x4 + x7*x0*x1*x2*x3*x4*x5
    f8 = x0*x1*x2*x3*x4*x5*x6*x7 - 1

    # Define the ideal
    I = R.ideal(f1, f2, f3, f4, f5, f6, f7, f8)

    start = timer()
    G = I.groebner_basis()
    end = timer()

    data[A] = end - start
    if i % 1 == 0:
        print(A, data[A])
        print(G)
        print()

print(data)
