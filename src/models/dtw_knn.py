"""DTW k-Nearest Neighbors classifier and utility functions.

Contains:
- compute_distance_matrix: computes timestep-wise distance matrix between two sequences
- compute_dtw_distance: takes a distance matrix and computes the cost of aligning the sequences
- DTWKNearestNeighbors: k-NN classifier using DTW for multivariate time series
"""

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


def compute_distance_matrix(seq_1: NDArray[np.float32], seq_2: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute a distance matrix for the distances between each point (timestep) in seq_1 and seq_2.

    Args:
        seq_1 (NDArray[np.float32]): An array of shape (len_1, num_channels_1) where len_1 is the
            number of timesteps and num_channels_1 the number of different MFCC channels.
        seq_2 (NDArray[np.float32]): An array of shape (len_2, num_channels_2) where len_2 is the
            number of timesteps and num_channels_2 the number of different MFCC channels.

    Returns:
        NDArray[np.float32]: A distance matrix of shape (len_1, len_2) where dist_matrix[i, j] = the
            Euclidean distance between the two vectors seq_1[i] and seq_2[j].
    """
    len_1, num_channels_1 = seq_1.shape
    len_2, num_channels_2 = seq_2.shape

    if num_channels_1 != num_channels_2:
        raise ValueError(
            "Both sequences must have the same number of MFCC channels."
            f"seq_1:{num_channels_1}, seq_2:{num_channels_2}"
        )

    # Before this, I used loops which were way too slow, this broadcasting is much faster.
    # First compute the distance between the timestep in seq 1 and seq 2 to get (len_1, len_2, num_channels)
    difference = seq_1[:, None, :] - seq_2[None, :, :]
    # Then sum and sqrt across the channels to get euclidean distancance in (len_1, len_2)
    dist_matrix = np.sqrt(np.sum(difference**2, axis=2)).astype(np.float32)

    return dist_matrix


def compute_dtw_distance(dist_matrix: NDArray[np.float32], normalize_pathlength: bool = True) -> float:
    """Compute the total cost to align two sequences using the DTW algorithm.

    Args:
        dist_matrix (NDArray[np.float32]): A timestep-wise distance matrix between two sequences
            of shape (len_1, len_2).
        normalize_pathlength (bool, optional): Whether to normalize DTW costs with their pathlengths to
                reduce bias from shorter paths. If true, devides total cost by (len(seq1) + len(seq2)).
                This is only an approximation to avoid having to traceback. Defaults to True.

    Returns:
        float: The total cost to align two sequences.
    """
    len_1, len_2 = dist_matrix.shape

    # Initialize the cost matrix with inf except for the corner which is 0
    # It has one extra row and column which will remain inf
    cost_matrix = np.full((len_1 + 1, len_2 + 1), np.inf, dtype=np.float32)
    cost_matrix[0, 0] = 0.0

    # Loop over all rows except the first one
    for i in range(1, len_1 + 1):
        # Loop over all columns except the first one
        for j in range(1, len_2 + 1):
            dist = dist_matrix[i - 1, j - 1]
            lowest_prev_cost = min(
                cost_matrix[i - 1, j - 1],  # Match
                cost_matrix[i - 1, j],  # Insertion
                cost_matrix[i, j - 1],  # Deletion
            )
            cost_matrix[i, j] = dist + lowest_prev_cost

    # The total cost will be stored in the final index of the cost matrix
    total_cost = cost_matrix[len_1, len_2]
    if normalize_pathlength:
        # Since the sequence length is 10-30 long, the cost of shorter sequences
        # needs to normalized to avoid biased predictions.
        total_cost = total_cost / (len_1 + len_2)

    return float(total_cost)


class DTWKNearestNeighbors:
    """A k-nearest neighbors classifier using Dynamic Time Warping (DTW) distance."""

    def __init__(self, k: int = 1, seed: int | None = None) -> None:
        """Initialize a DTWNearestNeighbors instance.

        Args:
            k (int, optional): The number of neighbors to use for prediction.
                Defaults to 1.
            seed (int, optional): The seed used for reproducibility. Defaults to None.
        """
        self.k = k

        # Store the random number generator for reproducibility
        self.rng = np.random.default_rng(seed)

        self.features: list[NDArray[np.float32]] | None = None
        self.labels: NDArray[np.int_] | None = None

    def fit(self, sequences: list[NDArray[np.float32]], labels: NDArray[np.int_]) -> None:
        """Validate and store the training sequences and the corresponding labels.

        Args:
            sequences (list[NDArray[np.float32]]): A list of sequences of size (timesteps, channels).
            labels (NDArray[np.int_]): An array of labels corresponding to the sequences.

        Raises:
            ValueError: If the number of sequences does not match the number of labels.
            ValueError: If the sequence is not 2d (timesteps, channels).
        """
        if len(sequences) != len(labels):
            raise ValueError(f"Number of sequences:{len(sequences)} must equal number of labels: {len(labels)}.")

        checked_sequences = []
        for seq in sequences:
            if seq.ndim != 2:
                raise ValueError(f"A sequence must be 2d (timesteps, channels) instead of {seq.ndim}d.")
            checked_sequences.append(seq)

        self.features = checked_sequences
        self.labels = labels

    def predict(self, new_sequences: list[NDArray[np.float32]], normalize_pathlength: bool = True) -> NDArray[np.int_]:
        """Predict the labels of the new_sequences using kNN on the DTW distances.

        Args:
            new_sequences (list[NDArray[np.float32]]): A list of sequences of size
                (timesteps, channels) that need predictions.
            normalize_pathlength (bool, optional): Whether to normalize DTW costs with their pathlengths to
                reduce bias from shorter paths. Defaults to True.

        Raises:
            ValueError: If there are no stored training sequences/labels.

        Returns:
            NDArray[np.int_]: The predicted labels of the new sequences.
        """
        predictions = []

        if self.features is None or self.labels is None:
            raise ValueError("There are no stored training sequences/labels. Call fit first.")

        for seq in tqdm(new_sequences, desc="Finding nearest labels"):
            seq = np.asarray(seq, dtype=np.float32)
            distances = self._compute_distances_to_features(seq, normalize_pathlength)

            # Find the indices of the k closest distances
            closest_idxs = np.argsort(distances)[: self.k]

            # Use these indices to find the k nearest labels
            nearest_labels = self.labels[closest_idxs]

            # Find the labels with the highest counts
            labels, counts = np.unique(nearest_labels, return_counts=True)
            max_count = np.max(counts)
            best_labels = labels[counts == max_count]

            # Best to avoid ties but if there are ties, break them randomly
            chosen_label = self.rng.choice(best_labels)
            predictions.append(chosen_label)

        return np.array(predictions, dtype=np.int_)

    def _compute_distances_to_features(
        self, sequence: NDArray[np.float32], normalize_pathlength: bool = True
    ) -> NDArray[np.float32]:
        """Compute the DTW distances from a given sequence to all training sequences.

        Args:
            sequence (NDArray[np.float32]): The sequence of size (timesteps, channels)
                for which the distances need to be computed.
            normalize_pathlength (bool, optional): Whether to normalize DTW costs with their pathlengths to
                reduce bias from shorter paths. Defaults to True.

        Raises:
            ValueError: If there are no stored training sequences.

        Returns:
            NDArray[np.float32]: The DTW distances from the sequence to the training sequences.
        """
        if self.features is None:
            raise ValueError("There are no stored training sequences. Call fit first.")

        distances = []
        for seq_train in self.features:
            dist_matrix = compute_distance_matrix(sequence, seq_train)
            distance = compute_dtw_distance(dist_matrix, normalize_pathlength)
            distances.append(distance)
        return np.array(distances, dtype=np.float32)
