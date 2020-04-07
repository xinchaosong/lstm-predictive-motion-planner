from multiprocessing import Process
import random
import numpy as np


def data_generator(task_id):
    data = []
    limit = 10
    step = 0.25 / limit

    for j in range(11000):
        x_start = random.uniform(-1, 1)
        direction = random.choice([-1, 1])
        x_1 = np.arange(x_start, direction, direction * step)
        x_2 = np.arange(x_1[-1] - direction * step, -direction, -direction * step)
        x_3 = np.arange(x_2[-1] + direction * step, direction, direction * step)
        x = np.concatenate((x_1, x_2, x_3))[:150]
        data.append(x)

    np.savetxt('../data/raw/raw_data_101_%s.csv' % task_id, data, delimiter=',')


if __name__ == '__main__':
    process_list = []
    num_process = 10

    for i in range(num_process):
        process_list.append(Process(target=data_generator, args=(i,)))

    for i in range(num_process):
        process_list[i].start()

    for i in range(num_process):
        process_list[i].join()
