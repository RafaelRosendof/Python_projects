import numpy as np 

from concurrent.futures import ProcessPoolExecutor

def calculate(f , a , b ,n):
    h = (b-a)/n
    x = np.linspace(a+h , b - h , n-1)
    sum_res = sum(f(xi) for xi in x)
    return h * (0.5 * (f(a) + 0.5 * f(b)) + sum_res)

def trapezoidal(f , a , b, n ,tol):
    prev_res = 0
    
    current_res = calculate(f , a , b , n)

    while abs(current_res - prev_res) > tol:
        prev_res = current_res
        n *= 2
        current_res = calculate(f , a , b , n)

    return current_res

def parallel_trapezoidal(f , a , b , n , tol , n_workers):
    with ProcessPoolExecutor() as executor:
        future = executor.submit(trapezoidal , f , a , b , n , tol)
        return future.result()