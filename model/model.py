import torch.nn as nn
import torch


class LSTMPredictor(nn.Module):

    def __init__(self, look_back, target_days):
        super(LSTMPredictor, self).__init__()

        # Nerual Layers
        self.layer_a = nn.Linear(look_back, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, target_days)

    def predict(self, input):
        with torch.no_grad():
            return self.forward(input).tolist()

    def forward(self, input):
        # Feed input to BERT
        logits = self.output(self.relu(self.layer_a(input)))

        return logits


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, device):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.device = device

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(
            0), self.hidden_dim, device=self.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out
