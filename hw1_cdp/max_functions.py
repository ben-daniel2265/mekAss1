import numpy as np
from numba import cuda, njit, prange, float32
import timeit


def max_cpu(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    C = np.zeros((1000, 1000))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i, j] = max(A[i, j], B[i, j])

    return C


@njit(parallel=True)
def max_numba(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    C = np.zeros((1000, 1000))
    for i in prange(A.shape[0]):
        for j in prange(A.shape[1]):
            C[i, j] = max(A[i, j], B[i, j])
    return C


def max_gpu(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    threadsperblock = 1000
    blockspergrid = 1000
    C = np.zeros((1000, 1000))
    max_kernel[blockspergrid, threadsperblock](A, B, C)
    return C


@cuda.jit
def max_kernel(A, B, C):
    # Thread id in a 1D block
    i = cuda.threadIdx.x
    # Block id in a 1D grid
    j = cuda.blockIdx.x
    C[i, j] = max(A[i, j], B[i, j])


def verify_solution():
    A = np.random.randint(0, 256, (1000, 1000))
    B = np.random.randint(0, 256, (1000, 1000))

    if not np.all(max_cpu(A, B) == np.maximum(A, B)):
        print('[-] max_cpu failed')
        exit(0)
    else:
        print('[+] max_cpu passed')

    if not np.all(max_numba(A, B) == np.maximum(A, B)):
        print('[-] max_numba failed')
        exit(0)
    else:
        print('[+] max_numba passed')

    if not np.all(max_gpu(A, B) == np.maximum(A, B)):
        print('[-] max_gpu failed')
        exit(0)
    else:
        print('[+] max_gpu passed')

    print('[+] All tests passed\n')


# this is the comparison function - keep it as it is.
def max_comparison():
    A = np.random.randint(0, 256, (1000, 1000))
    B = np.random.randint(0, 256, (1000, 1000))

    def timer(f):
        return min(timeit.Timer(lambda: f(A, B)).repeat(3, 20))

    print('[*] CPU:', timer(max_cpu))
    print('[*] Numba:', timer(max_numba))
    print('[*] CUDA:', timer(max_gpu))



if __name__ == '__main__':
    verify_solution()
    max_comparison()

A = np.random.randint(0, 256, (1000, 1000))
B = np.random.randint(0, 256, (1000, 1000))
C = max_cpu(A, B)
print(C)
