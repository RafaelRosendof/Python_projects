from __future__ import division
from numba import cuda
import time
import math , numpy as np

print(cuda.gpus)

@cuda.jit
def kernel(io_array):
    #Thread em 1D block
    tx = cuda.threadIdx.x
    
    #Block em 1D grid
    ty = cuda.blockIdx.x

    bw = cuda.blockDim.x

    #Computando flatten index dentro do array
    pos = tx + ty * bw

    if pos < io_array.size:
        io_array[pos] *= 2
        print(io_array[pos])


@cuda.jit
def matmul(A , B , C):
    """Perform matrix multiplication of C = A * B
    """
    row , col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row , k] * B[k , col]
        C[row , col] = tmp
    

@cuda.jit(device=True)
def f(x):
    return math.log(x ** 2 * math.sqrt(x ** 2 + 1) + x ** 2 + 1)

@cuda.jit
def trapezio(a , b , n , result):
    idx = cuda.grid(1)
    h = (b - a) / n

    if idx < n:
        x_i = a + idx * h
        x_i1 = x_i + h

        result[idx] = (f(x_i) + f(x_i1)) * h / 2

    
    
def main():
    ''' 
    A = np.full((24000 , 1220) , 27 , np.float64) #Matrix de 24x12
    B = np.full((1220 , 22000) , 64 , np.float64) #Matrix de 12x22

    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)

    #Alocando memória para o resultado
    C_global_mem = cuda.device_array((24000 , 22000))

    #COnfigurtando os blocos 
    threadsperblock = (32 , 32)
    blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x , blockspergrid_y)

    #iniciando o kernel


#    begin = time.time()
    
#    matmul[blockspergrid , threadsperblock](A_global_mem , B_global_mem , C_global_mem)

#    end = time.time()

#    C = C_global_mem.copy_to_host()

#    print(C)
#    print(f'Tempo de execução: {end - begin}')

    print('---------------------------------\n\n\n')
    '''
    a = 1.0
    b = 100.0
    n = 10000000

    result = np.zeros(n , dtype=np.float64)

    res_global_mem = cuda.to_device(result)

    thBlock = 32

    blocks2Grid = (n+ ( thBlock - 1) // thBlock)

    #executando o kernel
    inicio = time.time()
    trapezio[blocks2Grid , thBlock](a , b , n , res_global_mem)

    result = res_global_mem.copy_to_host()
    integral = result.sum()

    fim = time.time()

    total = fim - inicio

    print(f'Valor da integral: {integral} tempo de execução: {total}')

if __name__ == "__main__":
    main()