import multiprocessing as mp
import numpy as np
import time
import os 

print(f"PID: {os.getpid()}")

print("\n\n " , os.cpu_count())


def fx(x):
    return np.exp(x)

listA = np.random.rand(80000000)

sum_seq = 0
time1 = time.time()
for i in listA:
    sum_seq += fx(i)
fim = time.time()
print("Tempo sequencial:  ", fim - time1)

def calcular_parcial(start_idx, end_idx, listA, result, index):
    soma_partial = sum(fx(listA[i]) for i in range(start_idx, end_idx))
    result[index] = soma_partial

num_processes = 4  
chunk_size = len(listA) // num_processes

processes = []
manager = mp.Manager()
result = manager.list([0] * num_processes)

time2 = time.time()
for i in range(num_processes):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size if i != num_processes - 1 else len(listA)
    p = mp.Process(target=calcular_parcial, args=(start_idx, end_idx, listA, result, i))
    processes.append(p)
    p.start()

for p in processes:
    p.join()

sum_par = sum(result)
fim2 = time.time()
print("Tempo paralelo:  ", fim2 - time2)
print(f"Soma calculada (paralela): {sum_par}")
