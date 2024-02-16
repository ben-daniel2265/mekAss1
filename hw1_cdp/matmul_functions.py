import numpy as np
from numba import njit, cuda, prange
import timeit


def matmul_transpose_trivial(X):
    C = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            sum = 0
            for k in range(X.shape[1]):
                sum += (X[i][k] * X[j][k])

            C[i][j] = sum

    return C



@njit(parallel=True)
def matmul_transpose_numba(X):
    C = np.zeros((X.shape[0], X.shape[0]))
    for i in prange(X.shape[0]):
        for j in prange(X.shape[0]):
            sum = 0
            for k in prange(X.shape[1]):
                sum += (X[i][k] * X[j][k])

            C[i][j] = sum

    return C

def matmul_transpose_gpu(X):
    threadsperblock = 1024
    blockspergrid = 1

    C = np.zeros((X.shape[0], X.shape[0]))
    matmul_kernel[blockspergrid, threadsperblock](X, C)
    return C

@cuda.jit
def matmul_kernel(A, C):
    # Thread id in a 1D block
    i = cuda.threadIdx.x
    # Block id in a 1D grid
    j = cuda.blockIdx.x

    size = ((A.shape[0]) ** 2 + 1023) // 1024

    current = i * size

    for k in range(current, min(current + size, (A.shape[0]) ** 2)):
        row = k % A.shape[0]
        column = k // A.shape[0]

        sum = 0
        for n in range(A.shape[1]):
            sum += A[row][n] * A[column][n]

        C[row][column] = sum



def verify_solution():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    if not np.allclose(matmul_transpose_trivial(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_trivial failed')
        exit(0)
    else:
        print('[+] matmul_transpose_trivial passed')

    if not np.allclose(matmul_transpose_numba(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_numba failed')
        exit(0)
    else:
        print('[+] matmul_transpose_numba passed')

    if not np.allclose(matmul_transpose_gpu(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_gpu failed')
        exit(0)
    else:
        print('[+] matmul_transpose_gpu passed')

    print('[+] All tests passed\n')


# this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X, Xt)).repeat(3, 100))

    # print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    verify_solution()
    matmul_comparison()
