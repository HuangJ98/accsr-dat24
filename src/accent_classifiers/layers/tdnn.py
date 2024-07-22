from torch import nn


class TDNNLayer(nn.Module):
    """Dense layer with a optional activation function and dropout. Batchnorm can be used but during testing it resulted into worse results."""

    def __init__(self, n_input, n_output, activation_fn, dropout, kernel_size, dilation):
        """Activation function and dropout can be null."""
        super().__init__()
        
        self.layer = nn.Conv1d(in_channels=n_input, out_channels=n_output, kernel_size= kernel_size, dilation= dilation)
        #self.batchnorm = nn.BatchNorm1d(n_output)
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x):
        x = self.layer(x)
        #norm = self.batchnorm(x)
        out  = self.activation_fn(x) if self.activation_fn is not None else x
        return self.dropout(out) if self.dropout is not None else out
