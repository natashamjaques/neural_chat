import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1, hidden_size=None,
                 activation="Tanh", bias=True, no_activation_last_layer=False):
        super(FeedForward, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = getattr(nn, activation)()
        self.no_activation_last_layer = no_activation_last_layer
        n_inputs = [input_size] + [hidden_size] * (num_layers - 1)
        n_outputs = [hidden_size] * (num_layers - 1) + [output_size]
        self.linears = nn.ModuleList([nn.Linear(n_in, n_out, bias=bias)
                                      for n_in, n_out in zip(n_inputs, n_outputs)])

    def forward(self, input):
        x = input
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if not self.no_activation_last_layer or i < len(self.linears) - 1:
                x = self.activation(x)

        return x
