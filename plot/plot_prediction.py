import matplotlib.pyplot as plt
import numpy as np
from task_management.task_list import tasks

plot_task = tasks[105]

print("Loading the data...")

with open('../data/npy/original_trajectories_' + plot_task.task_name + '.npy', 'rb') as f:
    original_trajectories = np.load(f)

with open('../data/npy/predictive_trajectories_' + plot_task.task_name + '.npy', 'rb') as f:
    predictive_trajectories = np.load(f)

print("Data loaded successfully. The dimensions are:")
print(original_trajectories.shape)
print(predictive_trajectories.shape)

print("Plotting...")

i = 1600

plt.style.use('ggplot')
plt.figure(figsize=(12, 8))
plt.title('Prediction VS Ground Truth for the Moving Obstacle')
plt.plot(list(original_trajectories[i, :, 0]), c='r')
plt.plot(list(predictive_trajectories[i, :, 0]), c='b')
plt.legend(["ground truth", "prediction"])
plt.xlabel('Time')
plt.ylabel('X')

plt.show()
plt.close()
