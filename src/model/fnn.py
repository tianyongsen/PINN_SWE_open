import torch
import torch.nn as nn
import torch.nn.functional as F

from deepxde import config
from deepxde.nn import NN

initializer_dict = {
    'Glorot normal': torch.nn.init.xavier_normal_,
    'Glorot uniform': torch.nn.init.xavier_uniform_,
    'He normal': torch.nn.init.kaiming_normal_,
    'He uniform': torch.nn.init.kaiming_uniform_,
    'zeros': torch.nn.init.zeros_,
}

activation_dict = {
    "elu": F.elu,
    "relu": F.relu,
    "selu": F.selu,
    "sigmoid": F.sigmoid,
    "silu": F.silu,
    "sin": torch.sin,
    "tanh": torch.tanh,
}


class FNN(NN):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        self.activation = activation_dict[activation]         # activation function
        initializer = initializer_dict[kernel_initializer]    # weight initializer
        initializer_zero = initializer_dict['zeros']          # bias initializer

        self.linears = torch.nn.ModuleList()               # linear layers
        for i in range(1, len(layer_sizes)):               # input layer is not included in this loop
            self.linears.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch))) #add linear layer
            initializer(self.linears[-1].weight)             # initialize weight
            initializer_zero(self.linears[-1].bias)          # initialize bias to zero

    def forward(self, inputs):     
        x = inputs
        if self._input_transform is not None:    # apply input transform if specified
            x = self._input_transform(x)
        for linear in self.linears[:-1]:         # apply linear layers except the last one
            x = self.activation(linear(x))
        x = self.linears[-1](x)                  # apply last linear layer
        if self._output_transform is not None:   # apply output transform if specified
            x = self._output_transform(inputs, x)
        return x


class PFNN(NN):
    """Parallel fully-connected network that uses independent sub-networks for each
    network output.

    Args:
        layer_sizes: A nested list that defines the architecture of the neural network
            (how the layers are connected). If `layer_sizes[i]` is an int, it represents
            one layer shared by all the outputs; if `layer_sizes[i]` is a list, it
            represents `len(layer_sizes[i])` sub-layers, each of which is exclusively
            used by one output. Note that `len(layer_sizes[i])` should equal the number
            of outputs. Every number specifies the number of neurons in that layer.
    """

    def __init__(self, layer_sizes, activation, kernel_initializer, split_mask=None):
        super().__init__()
        self.activation = activation_dict[activation]
        initializer = initializer_dict[kernel_initializer]
        initializer_zero = initializer_dict['zeros']
        self.split_mask = torch.as_tensor(split_mask)

        if len(layer_sizes) <= 1:
            raise ValueError("must specify input and output sizes")
        if not isinstance(layer_sizes[0], int):
            raise ValueError("input size must be integer")
        if not isinstance(layer_sizes[-1], int):
            raise ValueError("output size must be integer")

        n_output = layer_sizes[-1]

        def make_linear(n_input, n_output):
            linear = torch.nn.Linear(n_input, n_output, dtype=config.real(torch))
            initializer(linear.weight)
            initializer_zero(linear.bias)
            return linear

        self.layers = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes) - 1):
            prev_layer_size = layer_sizes[i - 1]
            curr_layer_size = layer_sizes[i]
            if isinstance(curr_layer_size, (list, tuple)):
                if len(curr_layer_size) != n_output:
                    raise ValueError("number of sub-layers should equal number of network outputs")
                if isinstance(prev_layer_size, (list, tuple)):
                    # e.g. [8, 8, 8] -> [16, 16, 16]
                    self.layers.append(torch.nn.ModuleList([make_linear(prev_layer_size[j], curr_layer_size[j]) for j in range(n_output)]))
                else:  # e.g. 64 -> [8, 8, 8]
                    self.layers.append(torch.nn.ModuleList([make_linear(prev_layer_size, curr_layer_size[j]) for j in range(n_output)]))
            else:  # e.g. 64 -> 64
                if not isinstance(prev_layer_size, int):
                    raise ValueError("cannot rejoin parallel subnetworks after splitting")
                self.layers.append(make_linear(prev_layer_size, curr_layer_size))

        # output layers
        if isinstance(layer_sizes[-2], (list, tuple)):  # e.g. [3, 3, 3] -> 3
            self.layers.append(torch.nn.ModuleList([make_linear(layer_sizes[-2][j], 1) for j in range(n_output)]))
        else:
            self.layers.append(make_linear(layer_sizes[-2], n_output))

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        for layer in self.layers[:-1]:
            if isinstance(layer, torch.nn.ModuleList):
                if isinstance(x, list):
                    x = [self.activation(f(x_)) for f, x_ in zip(layer, x)]
                else:
                    if self.split_mask is not None:
                        x = [self.activation(f(x * self.split_mask[i])) for i, f in enumerate(layer)]
                    else:
                        x = [self.activation(f(x)) for f in layer]
            else:
                x = self.activation(layer(x))

        # output layers
        if isinstance(x, list):
            x = torch.cat([f(x_) for f, x_ in zip(self.layers[-1], x)], dim=1)
        else:
            x = self.layers[-1](x)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
