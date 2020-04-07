import torch
from torch import nn


class LSTMEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(LSTMEncoder, self).__init__()

        self.input_size_ = input_size
        self.hidden_size_ = hidden_size
        self.num_layers_ = num_layers

        self.encoder_ = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )

    def forward(self, x):
        encoded_output, hidden = self.encoder_(x)
        return encoded_output[:, [-1], :], hidden


class LSTMDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, sequence_length, num_layers, batch_first=True):
        super(LSTMDecoder, self).__init__()

        self.input_size_ = input_size
        self.hidden_size_ = hidden_size
        self.num_layers_ = num_layers
        self.sequence_length_ = sequence_length

        self.decoder_ = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )

        self.fc_layer_ = nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )

    def forward(self, x, hidden):
        decoded_output = []
        output = x

        for idx in range(self.sequence_length_):
            output, hidden = self.decoder_(output, hidden)
            unit_decoded_output = self.fc_layer_(output)
            decoded_output.append(unit_decoded_output)

        decoded_output = torch.stack(decoded_output, 1).squeeze(2)

        return decoded_output, hidden
