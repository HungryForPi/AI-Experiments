#!/usr/bin/env sage

import sys
from timeit import default_timer as timer
import multiprocessing as mp

n = 3
var(['x' + str(i) for i in range(n)])

def gb_stopwatch(I, time):
    st = timer()
    I.groebner_basis()
    en = timer()
    time.value = en - st

def test_case(polys, A, time_limit):
    # polys is a list of polynomials
    T = TermOrder(A)
    R = PolynomialRing(QQ, 'x', n, order=T)
    # print(T)
    # print(x0 > x1)
    # print(R)
    # print(TermOrder(A))
    I = R.ideal(polys)

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
