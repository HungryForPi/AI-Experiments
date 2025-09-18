#!/usr/bin/env sage

import sys
from timeit import default_timer as timer
import multiprocessing as mp

n = 6
var(['x' + str(i) for i in range(n)])
# Define the generators of the Cyclic(6) ideal
f1 = x0 + x1 + x2 + x3 + x4 + x5
f2 = x0*x1 + x1*x2 + x2*x3 + x3*x4 + x4*x5 + x5*x0
f3 = x0*x1*x2 + x1*x2*x3 + x2*x3*x4 + x3*x4*x5 + x4*x5*x0 + x5*x0*x1
f4 = x0*x1*x2*x3 + x1*x2*x3*x4 + x2*x3*x4*x5 + x3*x4*x5*x0 + x4*x5*x0*x1 + x5*x0*x1*x2
f5 = x0*x1*x2*x3*x4 + x1*x2*x3*x4*x5 + x2*x3*x4*x5*x0 + x3*x4*x5*x0*x1 + x4*x5*x0*x1*x2 + x5*x0*x1*x2*x3
f6 = x0*x1*x2*x3*x4*x5 - 1

def gb_stopwatch(I, time):
    st = timer()
    I.groebner_basis()
    en = timer()
    time.value = en - st

def test_case(A, time_limit):
    T = TermOrder(A)
    R = PolynomialRing(QQ, 'x', n, order=T)
    # print(T)
    # print(x0 > x1)
    # print(R)
    # print(TermOrder(A))
    I = R.ideal([f1, f2, f3, f4, f5, f6])

    time = mp.Value('d', time_limit)
    process = mp.Process(target=gb_stopwatch, args=(I, time, ))
    process.start()
    process.join(timeout=time_limit)

    if process.is_alive():
        process.terminate()
        process.join()

    return time.value

def main():
    A = [int(sys.argv[i]) for i in range(1,n*n+1)]
    TIME_LIMIT = int(sys.argv[n*n+1])
    result = test_case(A, TIME_LIMIT)
    return result

if __name__ == "__main__":
    print(main())
