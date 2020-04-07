from torch import nn

from model.components import LSTMEncoder
from model.components import LSTMDecoder


class CompositeModel(nn.Module):

    def __init__(self, task, input_size, hidden_size, input_sequence_length, future_sequence_length,
                 num_layers, batch_first=True):
        super(CompositeModel, self).__init__()

        self.task = task
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_sequence_length = input_sequence_length
        self.future_sequence_length = future_sequence_length
        self.num_layers = num_layers
        self.batch_firs = batch_first

        self.encoder_ = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )

        self.decoder_ = LSTMDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=input_size,
            num_layers=num_layers,
            batch_first=batch_first,
            sequence_length=input_sequence_length
        )

        self.future_predictor_ = LSTMDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=input_size,
            num_layers=num_layers,
            batch_first=batch_first,
            sequence_length=future_sequence_length
        )

    def forward(self, x):
        encoded_output, hidden = self.encoder_(x)
        decoded_output, hidden = self.decoder_(encoded_output, hidden)
        forward_output, hidden = self.future_predictor_(encoded_output, hidden)

        return encoded_output, decoded_output, forward_output
