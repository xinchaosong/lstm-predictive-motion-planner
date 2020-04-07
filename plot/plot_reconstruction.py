import matplotlib.pyplot as plt
import numpy as np
from task_management.task_list import tasks

plot_task = tasks[105]

print("Loading the data...")

with open('../data/npy/original_trajectories_' + plot_task.task_name + '.npy', 'rb') as f:
    original_trajectories = np.load(f)

with open('../data/npy/decoded_output_' + plot_task.task_name + '.npy', 'rb') as f:
    reconstruction_trajectories = np.load(f)

print("Data loaded successfully. The dimensions are:")
print(original_trajectories.shape)
print(reconstruction_trajectories.shape)

print("Plotting...")

i = 1200

plt.style.use('ggplot')
plt.figure(figsize=(12, 8))
plt.title('Reconstruction VS Ground Truth for the Moving Obstacle')
plt.plot(list(original_trajectories[i, :50, 0]), c='r')
plt.plot(list(np.flip(reconstruction_trajectories[i, :, 0])), c='b')
plt.legend(["ground truth", "reconstruction"])
plt.xlabel('Time')
plt.ylabel('X')

plt.show()
plt.close()
