"""GRU classifier.

Contains:
- GRUClassifier: GRU classifier for multivariate time series
"""

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """Classify MFCCs using a GRU model from pytorch."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_classes: int,
        bidirectional: bool = True,
        dropout_prob: float = 0.1,
        device: torch.device | None = None,
    ) -> None:
        """Initialize a GRU classifier.

        Args:
            input_size (int): The number of MFCC channels in each sequence.
            hidden_size (int): The size of the hidden_state.
            n_classes (int): The number of classes to predict.
            bidirectional (bool, optional): Whether to apply the GRU model to the input data in both
                directions. Defaults to True.
            dropout_prob (float, optional): The percentage of neurons that should be set to zero before being
                fed into the fully connected layer that produces the logits.
            device (torch.device | None, optional): The device to use for the tensor operations. If no device is
                specified, "cpu" is used. Defaults to None.
        """
        # Make sure it becomes a pytorch model for easier integration
        super(GRUClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True,  # Input is formatted as (batch_size, seq_len, input_size)
        )

        self.dropout = nn.Dropout(p=dropout_prob)

        # Classification head for the last hidden state
        self.fc_classes = nn.Linear(
            # Using bidirectional doubles the hidden size
            in_features=hidden_size * (2 if bidirectional else 1),
            out_features=n_classes,
        )

        # Default to cpu if no device specified
        self.device = device if device else torch.device("cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perfrom one forward pass with the current GRU model.

        Args:
            x (torch.Tensor): The sequences for which classes need to be predicted. Has shape
                (batch_size, seq_len, input_size). Where input size is the number of MFCC channels.

        Returns:
            torch.Tensor: The raw logits of the labels for the given sequences. Has shape (batch_size, n_labels)
        """
        x = x.to(self.device)

        # Assumes H0 consists of zeros, only keep last hidden state for each item in the batch
        _, last_hidden = self.gru(x)

        if self.bidirectional:
            # last_hidden has dimension (2, batch_size, hidden_size) so
            # We need to stack the hidden states for each item in the batch
            last_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
            # New dimensions are (batch_size, 2*hidden_size)

        else:
            # last_hidden has dimension (1, batch_size, hidden_size)
            last_hidden = last_hidden[-1]
            # New dimensions are (batch_size, hidden_size)

        # If dropout is enabled, we need to set the activations of some neurons to zero first.
        last_hidden = self.dropout(last_hidden)

        # Return the raw logits from the FC layer
        return self.fc_classes(last_hidden)
