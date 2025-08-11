import random as rand
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
    s = timer()
    totaltime = 0
    TIME_LIMIT = 1000
    NUM_SAMPLES = 200
    MATRIX_ENTRY_SIZE = 100
    with open('data.txt', 'w') as f:
        for i in range(NUM_SAMPLES):
            # too-large matrix coefs may cause overflow issues or speed issues
            A = matrix(n, [rand.randint(1, MATRIX_ENTRY_SIZE) for _ in range(n*n)],
                       immutable=True)
            result = test_case(A, TIME_LIMIT)
            totaltime += result
            print(f'run number {i}: The current runtime is {totaltime} seconds.', end=' ')
            if result == TIME_LIMIT:
                print('The previous computation timed out.')
            else:
                print(f'The previous computation took {result} seconds.')
            print(timer() - s)
            print()
            to_write = ""
            for B in A:
                for j in B:
                    to_write += str(j) + " "
            to_write += str(result)
            to_write += "\n"
            f.write(to_write)
    
    print(f'The total runtime was {totaltime} seconds.')

if __name__ == "__main__":
    main()
