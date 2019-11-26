import torch
import torch.nn as nn

class ANN(nn.Module):
    """A general class for fully connected networks"""
    
    def __init__(self, input_dim, hidden_layers, output_dim, bias=True):
        """
        Args:
            input_dim (int): Number of features the ANN is expecting
            hidden_layers (list): Entries containing the number of hidden layers in each layer.
            activations (string): Activation function to use for non-linearity.
            output_dim (int): Number of classes
        """
        super(ANN, self).__init__()
        # Layer definitions 
        self.head = nn.Sequential(*[
            nn.Linear(in_features=input_dim, out_features=hidden_layers[0], bias=bias),
            nn.ReLU()
        ])
        
        body = []
        prev_layers = hidden_layers[0]
        for layer in hidden_layers[1:]:
            body.append(
                nn.Sequential(
                    nn.Linear(in_features=prev_layers, out_features=layer, bias=bias),
                    nn.ReLU()
                )
            )
            prev_layers = layer
        self.body = nn.Sequential(*body)
        # TODO: Check if sigmoid is really required here.
        self.tail = nn.Sequential(*[
            nn.Linear(prev_layers, output_dim, bias=bias),
            nn.Sigmoid()
        ])

    def forward(self, x):
        # Forward pass
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x