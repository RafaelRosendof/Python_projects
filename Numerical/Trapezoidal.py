import numpy as np
import time
from numba import njit , vectorize

#@njit
#def f(x):
#    return np.sin(x)

@vectorize
def f(x):
    return np.sin(x)

@njit
def teste2(a, b, n=10000000):
    h = (b - a) / n
    f_sum = 0

    for i in range(1, n):
        x = a + i * h
        f_sum += f(x)
    return 0.5 * h * (f(a) + f(b) + 2 * f_sum)

def lento(f, a, b, n=10000000):
    h = (b - a) / n
    f_sum = 0

    for i in range(1, n):
        x = a + i * h
        f_sum += f(x)
    return 0.5 * h * (f(a) + f(b) + 2 * f_sum)

def main():
    a = 0
    b = np.pi

    tempo1 = time.time()
    figas = lento(np.sin, a, b)  # Passando np.sin, que é uma função compatível com Numba
    tempo2 = time.time()

    total1 = tempo2 - tempo1

    print(f'Valor da integral: {figas}')
    print(f'Tempo de execução: {total1}')

    print('---------------------------------\n\n\n')

    tempo3 = time.time()
    figas2 = teste2(a, b)  # Apenas passe os limites
    tempo4 = time.time()

    total2 = tempo4 - tempo3

    print(f'Valor da integral: {figas2}')
    print(f'Tempo de execução: {total2}')

if __name__ == '__main__':
    main()