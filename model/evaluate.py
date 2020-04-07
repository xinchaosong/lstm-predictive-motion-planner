import torch
from torch import nn
import numpy as np

from model.composite_model import CompositeModel
from data_processor.data_handler import CsvDataset
from task_management.task_list import tasks


def evaluate_model(task, checkpoint_path):
    input_sequence_length = 50
    future_sequence_length = 100
    x_limit = 10.0
    model = CompositeModel(
        task=task,
        input_size=task.input_size,
        hidden_size=task.hidden_size,
        input_sequence_length=input_sequence_length,
        future_sequence_length=future_sequence_length,
        num_layers=1,
        batch_first=True
    )
    model.load_state_dict(torch.load(checkpoint_path), strict=True)

    data_file = '../data/data_set/' + task.data_set_csv + '_test.csv'
    data_set = CsvDataset(data_file=data_file)

    normalized_trajectories = data_set[:, :, :] / x_limit

    input_data = normalized_trajectories[:, :input_sequence_length, :]
    future_data = normalized_trajectories[:, input_sequence_length:, :]
    input_data_reverse = torch.flip(input_data, dims=[1])

    encoded_output, decoded_output, forward_output = model.forward(input_data)

    reconstruction_criteria = nn.MSELoss()
    reconstrct_loss = reconstruction_criteria(
        input=decoded_output,
        target=input_data_reverse
    )

    prediction_criteria = nn.MSELoss()
    prediction_loss = prediction_criteria(
        input=forward_output,
        target=future_data
    )

    with open("../data/npy/decoded_output_" + task.task_name + ".npy", "wb") as f:
        unnormalized_decoded_output = decoded_output.detach().numpy() * x_limit
        np.save(f, unnormalized_decoded_output)

    with open("../data/npy/original_trajectories_" + task.task_name + ".npy", "wb") as f:
        original_trajectories = data_set[:, :, :].detach().numpy()
        np.save(f, original_trajectories)

    with open("../data/npy/predictive_trajectories_" + task.task_name + ".npy", "wb") as f:
        predictive_trajectories = torch.cat((input_data, forward_output), dim=1).detach().numpy() * x_limit
        np.save(f, predictive_trajectories)

    print("Reconstruction loss: %.6f, Future loss: %.6f" % (reconstrct_loss.item(), prediction_loss.item()))


if __name__ == '__main__':
    task_test = tasks[105]

    evaluate_model(
        task=task_test,
        checkpoint_path='../data/checkpoints/checkpoint_' + task_test.task_name + '.pt'
    )
