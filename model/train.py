import datetime

import torch
from torch import nn
from torch import optim
import numpy as np
from torch.utils.data import DataLoader

import model.utils as utils
from model.composite_model import CompositeModel
from data_processor.data_handler import CsvDataset
from data_processor.data_recorder import DataRecorder
from task_management.task_list import tasks
from plot import plot_loss


def train_model(task, eval_freq, checkpoint_path):
    task.save()

    input_sequence_length = 50
    future_sequence_length = 100
    x_limit = 10.0
    lr = 1e-3
    weight_decay = 0
    sparsity_weight = 0
    device = utils.get_device(task.gpu_index)
    model = CompositeModel(
        task=task,
        input_size=task.input_size,
        hidden_size=task.hidden_size,
        input_sequence_length=input_sequence_length,
        future_sequence_length=future_sequence_length,
        num_layers=1,
        batch_first=True
    ).to(device)

    data_file_train = '../data/data_set/' + task.data_set_csv + '_train.csv'
    data_set_train = CsvDataset(data_file=data_file_train)
    data_loader_train = DataLoader(
        dataset=data_set_train,
        batch_size=4096,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    data_file_valid = '../data/data_set/' + task.data_set_csv + '_valid.csv'
    data_set_valid = CsvDataset(data_file=data_file_valid)
    data_loader_valid = DataLoader(
        dataset=data_set_valid,
        batch_size=len(data_set_valid),
        shuffle=False,
        num_workers=1,
        drop_last=False
    )

    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_recorder_column = ['epoch_id', 'train_reco_loss', 'train_pred_loss', 'train_loss',
                            'vali_reco_loss', 'vali_pred_loss']
    loss_recorder = DataRecorder(column=loss_recorder_column)
    output_path = "../data/loss/" + task.task_name + "_loss.csv"

    start_time = datetime.datetime.now()
    print("Training starts")

    for epoch_i in range(task.num_epochs):
        print("Task Name: %s, epoch ID: %d" % (task.task_name, epoch_i))

        train_reconstrct_losses = []
        train_prediction_losses = []
        train_losses = []
        vali_reconstrct_losses = []
        vali_prediction_losses = []

        for i_batch, curr_batch in enumerate(data_loader_train):
            normalized_trajectories = curr_batch / x_limit
            input_data = normalized_trajectories[:, :input_sequence_length, :].to(device)
            future_data = normalized_trajectories[:, input_sequence_length:, :].to(device)
            input_data_reverse = torch.flip(input_data, dims=[1]).to(device)

            encoded_output, decoded_output, forward_output = model.forward(input_data)

            reconstrct_criteria = nn.MSELoss()
            reconstrct_loss = reconstrct_criteria(
                input=decoded_output,
                target=input_data_reverse
            )
            prediction_criteria = nn.MSELoss()
            prediction_loss = prediction_criteria(
                input=forward_output,
                target=future_data
            )

            sparsity_loss = torch.mean(torch.pow(torch.norm(encoded_output, p=2, dim=-1), 2))
            loss = reconstrct_loss + prediction_loss + (sparsity_weight * sparsity_loss)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_reconstrct_losses.append(reconstrct_loss.item())
            train_prediction_losses.append(prediction_loss.item())
            train_losses.append(loss.item())

        train_reco_loss = np.mean(train_reconstrct_losses).item()
        train_pred_loss = np.mean(train_prediction_losses).item()
        train_loss = np.mean(train_losses).item()

        print("Train reconstruction loss: %.6f, train future loss: %.6f, train total loss: %.6f"
              % (train_reco_loss, train_pred_loss, train_loss))

        if epoch_i % 1 == 0:
            with open(checkpoint_path, 'wb') as f:
                torch.save(model.state_dict(), f)

        if epoch_i % eval_freq == 0:
            with torch.no_grad():
                for i_batch, curr_batch in enumerate(data_loader_valid):
                    normalized_trajectories = curr_batch / x_limit
                    input_data = normalized_trajectories[:, :input_sequence_length, :].to(device)
                    future_data = normalized_trajectories[:, input_sequence_length:, :].to(device)
                    input_data_reverse = torch.flip(input_data, dims=[1]).to(device)

                    encoded_output, decoded_output, forward_output = model.forward(input_data)

                    reconstrct_criteria = nn.MSELoss()
                    reconstrct_loss = reconstrct_criteria(
                        input=decoded_output,
                        target=input_data_reverse
                    )

                    prediction_criteria = nn.MSELoss()
                    prediction_loss = prediction_criteria(
                        input=forward_output,
                        target=future_data
                    )

                    vali_reconstrct_losses.append(reconstrct_loss.item())
                    vali_prediction_losses.append(prediction_loss.item())

                vali_reco_loss = np.mean(vali_reconstrct_losses).item()
                vali_pred_loss = np.mean(vali_prediction_losses).item()

                print("Validation reconstruction loss: %.6f, validation future loss: %.6f"
                      % (vali_reco_loss, vali_pred_loss))

        epoch_recorder_row = [int(epoch_i), train_reco_loss, train_pred_loss, train_loss,
                              vali_reco_loss, vali_pred_loss]
        loss_recorder.append(epoch_recorder_row)
        loss_recorder.save_to_csv(output_path, index=False)

        epochs_counter = epoch_i + 1

        if epochs_counter % 20 == 0 or epochs_counter == 1:
            current_time = datetime.datetime.now()
            left_time = (current_time - start_time).seconds / 60 / epochs_counter * (task.num_epochs - epochs_counter)

            if left_time > 100:
                print("\033[0;34m\t%s: estimate %1.f hour(s) %1.f min(s) left.\033[0m"
                      % (task.task_name, int(left_time / 60), left_time % 60))
            else:
                print("\033[0;34m\t%s: estimate %1.f min(s) left.\033[0m" % (task.task_name, left_time))

            plot_loss.plot(task_name=task.task_name, csv_path=output_path, task=task,
                           show_fig=False, save_fig=True)


if __name__ == '__main__':
    task_test = tasks[106]

    train_model(
        task=task_test,
        eval_freq=1,
        checkpoint_path='../data/checkpoints/checkpoint_' + task_test.task_name + '.pt',
    )
