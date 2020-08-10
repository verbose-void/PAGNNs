from PAGNN.pagnn import PAGNN
from copy import deepcopy
from time import time
import torch
import torch.autograd.profiler as profiler
import numpy as np


if __name__ == '__main__':
    seed = 666
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    num_neurons, in_neurons, out_neurons = 200, 1, 1
    cpu_pagnn = PAGNN(num_neurons, in_neurons, out_neurons)
    gpu_pagnn = PAGNN(num_neurons, in_neurons, out_neurons).cuda()

    batch_size = 200
    cpu_X = torch.rand((batch_size, 1))
    cpu_T = torch.rand((batch_size, 1))
    gpu_X = cpu_X.cuda()
    gpu_T = cpu_T.cuda()

    N = 100
    steps = 5

    with profiler.profile(record_shapes=True) as prof:
        gpu_dt = time()

        # with profiler.record_function('gpu_inference'):
        for _ in range(N):
            gpu_Y = gpu_pagnn(gpu_X, num_steps=steps)

        gpu_dt = (time() - gpu_dt) / N

        cpu_dt = time()

        # with profiler.record_function('cpu_inference'):
        for _ in range(N):
            cpu_Y = cpu_pagnn(cpu_X, num_steps=steps)

        cpu_dt = (time() - cpu_dt) / N

    print(cpu_Y)
    print(gpu_Y)

    # print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=10))
    print('cpu delta time', cpu_dt)
    print('gpu delta time', gpu_dt)
    print('cpu is %.2fx faster' % (gpu_dt/cpu_dt))
