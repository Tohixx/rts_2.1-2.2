import numpy as np
import random
import cmath
import math
import time
import matplotlib.pyplot as plt
from multiprocessing import Process
import multiprocessing
import json
from ast import literal_eval

def create_signals(n, N, W):
    generated_signal = np.zeros(N)
    for i in range(n):
        fi = 2 * math.pi * random.random()
        A = 5 * random.random()
        w = W - i * W / (n)
        x = A * np.sin(np.arange(0, N, 1) * w + fi)
        generated_signal += x

    return generated_signal


def fft_thread(N,return_dict):
    start = time.time()
    n = 12
    omega = 1100
    signal = create_signals(n, N, omega)
    N = len(signal)
    spectre = np.zeros(N, dtype=np.complex64)
    for p in range(N // 2):
        E_m = np.dot(signal[0:N:2], np.cos(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1))) - 1j * np.dot(signal[0:N:2],
              np.sin(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1)))
        W_p = (np.cos(2 * math.pi * p / N) - 1j * np.sin(2 * math.pi * p / N))
        O_m = np.dot(signal[1:N:2], np.cos(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1))) - 1j * np.dot(signal[1:N:2],
              np.sin(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1)))
        spectre[p] = E_m + W_p * O_m
        spectre[p + N // 2] = E_m - W_p * O_m
    return_dict[N] = time.time() - start



def fft(N):
    start = time.time()
    n = 12
    omega = 1100
    signal = create_signals(n, N, omega)
    N = len(signal)
    spectre = np.zeros(N, dtype=np.complex64)
    for p in range(N // 2):
        E_m = np.dot(signal[0:N:2], np.cos(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1))) - 1j * np.dot(signal[0:N:2],
              np.sin(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1)))
        W_p = (np.cos(2 * math.pi * p / N) - 1j * np.sin(2 * math.pi * p / N))
        O_m = np.dot(signal[1:N:2], np.cos(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1))) - 1j * np.dot(signal[1:N:2],
              np.sin(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1)))
        spectre[p] = E_m + W_p * O_m
        spectre[p + N // 2] = E_m - W_p * O_m
    polar_spectr_fft = np.array(list(map(lambda x: cmath.polar(x), spectre)))[:, 0]
    return time.time() - start

result_fft = {}
def run_fft():
    for N in [256,512,1024,2048,4096,8192,16384]:
        res = fft(N)
        result_fft[N] = res
    return result_fft

if __name__ == "__main__":  # confirms that the code is under main function
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    N_s = [256,512,1024,2048,4096,8192,16384]
    procs = []
    proc = Process(target=fft_thread)
    procs.append(proc)
    proc.start()

    for N in N_s:
        proc = Process(target=fft_thread, args=(N,return_dict))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


    return_dict = {int(k): v for k,v in literal_eval(json.dumps(return_dict._getvalue())).items()}
    result_fft_thread = {}

    for key in sorted(return_dict):
        result_fft_thread[key] = return_dict[key]

    result_fft = run_fft()

    result = [result_fft[key] - result_fft_thread[key] for key in result_fft_thread]

    plt.plot(result_fft.keys(), result)
    plt.xlabel('Difference time')
    plt.grid()
    plt.show()



