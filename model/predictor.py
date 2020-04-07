import torch

from model.composite_model import CompositeModel
from task_management.task_list import tasks


class Predictor:
    def __init__(self, task, checkpoint_path):
        self.task = task
        self.checkpoint_path = checkpoint_path
        self.input_sequence_length = 50
        self.future_sequence_length = 100
        self.x_limit = 10
        self.model = CompositeModel(
            task=task,
            input_size=task.input_size,
            hidden_size=task.hidden_size,
            input_sequence_length=self.input_sequence_length,
            future_sequence_length=self.future_sequence_length,
            num_layers=1,
            batch_first=True
        )
        self.model.load_state_dict(torch.load(checkpoint_path), strict=True)

    def predict(self, x):
        trajectory = torch.tensor(x, dtype=torch.float).unsqueeze(0).unsqueeze(2)
        input_data = trajectory[:, :self.input_sequence_length, :] / self.x_limit
        _, decoded_output, forward_output = self.model.forward(input_data)

        return decoded_output * self.x_limit, forward_output * self.x_limit


if __name__ == '__main__':
    task_test = tasks[105]

    predictor = Predictor(
        task=task_test,
        checkpoint_path='../data/checkpoints/checkpoint_' + task_test.task_name + '.pt'
    )
