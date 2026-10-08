"""Util functions for file handling and preprocessing.

Contains:
- file_to_sequences: A function to extract the data from the datafile.
- make_label_array: A function to create a label array with a specific number of labels for each class.
- logits_to_labels: A function that converts raw logits to labels by choosing the label with the heighest value.
- evaluate_prediction: A function that prints a classification report and shows a confusion matrix for provided true
    and predicted labels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

N_LABELS = 9


def file_to_sequences(path: str | Path, pad_sequences: bool = True) -> list[np.ndarray] | np.ndarray:
    """Read the data from a file and return either a list of sequences or a padded array of sequences.

    Args:
        path (str): The location where the data is stored.
        pad_sequences (bool, optional): Add zeros at the end of the sequences so that all sequences are the same
            length as the longest sequence. Defaults to True.

    Raises:
        FileNotFoundError: If there is no file to open in the specific location.

    Returns:
        list[np.ndarray] | np.ndarray: If padding is False, returns a list with sequences of variable lengths of shape:
            (n_sequences, len_seqence, n_channels). If padding is True, returns a 3d np.ndarray with zeros at the
            end of sequences and is of shape (n_sequences, max_len_sequence, n_channels).
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found at path {path}.")

    # Stores finished sequences
    sequences: list[np.ndarray] = []
    # Temporary storage for sequences
    current_seq: list[np.ndarray] = []

    with path.open(mode="r") as f:
        for line in f:
            # Remove trailing whitespaces because otherwise it'll probably mess up the splitting
            line = line.strip()
            # Sometimes there are entire lines of whitespaces between sequences which need to be skipped
            if not line:
                continue

            # Convert the list of string numbers to and np.array of actual float numbers
            row = np.fromstring(line, sep=" ", dtype=np.float32)

            # Detect the end of the sequences, 1.0 1.0 1.0 etc
            n_channels = row.size
            if np.allclose(row, np.ones(n_channels)):
                # End of sequence so dont store this row but append completed seq
                # Vstack places he different rows into one 2d matrix
                sequences.append(np.vstack(current_seq))
                current_seq = []
            else:
                current_seq.append(row)

    if not pad_sequences:
        return sequences  # Sequences have variable lengths

    # This time we make just a 3d numpy matrix instead of list of 2d matrixes
    max_seq_len = max(sequence.shape[0] for sequence in sequences)
    n_seqs = len(sequences)
    # First create full matrix, then fill with sequences
    padded_sequences = np.full(shape=(n_seqs, max_seq_len, n_channels), fill_value=0.0, dtype=np.float32)
    for sequence_number, sequence in enumerate(sequences):
        # Replace the first items with the sequence the rest will remain zeros
        padded_sequences[sequence_number, : sequence.shape[0], :] = sequence

    return padded_sequences  # Sequences all have the same length


def make_label_array(counts: list[int]) -> np.ndarray:
    """Create a label array with the specified number for each counts idx.

    Example: make_label_array([1, 2, 1]) -> [0, 1, 1, 2]

    Args:
        counts (list): Each item indicates how often to repeat the label idx.

    Raises:
        ValueError: If the number of labels does not match the expected number of labels.

    Returns:
        np.ndarray: An array of labels.
    """
    if len(counts) != N_LABELS:
        raise ValueError(f"Got counts for {len(counts)} labels but expected {N_LABELS} counts.")

    labels = []
    for idx, count in enumerate(counts):
        labels.extend([idx] * count)
    return np.array(labels, dtype=int)


def logits_to_labels(logits: torch.Tensor) -> torch.Tensor:
    """Convert raw logits to labels by choosing the label with the highest value.

    Args:
        logits (torch.tensor): The logits of each sequence for the labels of shape: (n_sequences, n_labels).

    Returns:
        torch.tensor: For each sequence the label with the highest value. Shape (n_sequences)
    """
    return torch.argmax(logits, dim=1)


def evaluate_predictions(
    labels_true: np.ndarray | torch.Tensor, labels_pred: np.ndarray | torch.Tensor, title: str = "Model"
) -> None:
    """Print classification report and show confusion matrix for the provided true and predicted labels.

    Args:
        labels_true (np.ndarray | torch.Tensor): The true labels of the sequences of shape (n_sequences).
        labels_pred (np.ndarray | torch.Tensor): The predicted labels of the sequences of shape (n_sequences).
        title (str, optional): The title that is printed before the classification report and shown as title
            in the confusion matrix. Defaults to "Model".

    Raises:
        ValueError: If the number of true labels does not match the number of predicted labels.
    """
    # We can only process np.arrays so first make sure there are no tensors
    if isinstance(labels_true, torch.Tensor):
        labels_true = labels_true.cpu().numpy()
    if isinstance(labels_pred, torch.Tensor):
        labels_pred = labels_pred.cpu().numpy()

    if len(labels_true) != len(labels_pred):
        raise ValueError(
            f"The number of true labels {len(labels_true)} does not match the "
            f"number of predicted labels {len(labels_pred)}."
        )

    # First print the classification report for exact values
    print(f"\n--- Classification report for {title}")
    # The actual label names are plus 1
    label_names = [str(number) for number in np.linspace(1, N_LABELS, N_LABELS, dtype=int)]
    print(classification_report(labels_true, labels_pred, target_names=label_names))

    # Next show the confusion matrix
    cm = confusion_matrix(labels_true, labels_pred)
    plt.figure(figsize=(N_LABELS, N_LABELS))
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix {title}")
    plt.tight_layout()
    plt.show()
