"""Load and evaluate a given model."""

from pathlib import Path

import torch

from src.models.dtw_knn import DTWKNearestNeighbors
from src.utils import (
    evaluate_predictions,
    file_to_sequences,
    logits_to_labels,
    make_label_array,
)

# Resolve makes current path absolute and parent goes back one step to NN-SEMESTER-PROJECT
ROOT = Path(__file__).resolve().parent
MODEL_FILE_NAME = "gru_final.pth"
train_file_path = ROOT / "data" / "ae.train"
test_file_path = ROOT / "data" / "ae.test"
save_dir = ROOT / "saved_models"
save_path = save_dir / MODEL_FILE_NAME

if __name__ == "__main__":
    # DTW KNN Baseline Model
    dtw_knn = DTWKNearestNeighbors(k=1)

    # This model can handle variable sequence lengths
    train_seqs = file_to_sequences(train_file_path, pad_sequences=False)
    label_train_knn_counts = [30, 30, 30, 30, 30, 30, 30, 30, 30]
    train_labels = make_label_array(label_train_knn_counts)
    dtw_knn.fit(train_seqs, train_labels)

    test_knn_seqs = file_to_sequences(test_file_path, pad_sequences=False)
    label_test_knn_counts = [31, 35, 88, 44, 29, 24, 40, 50, 29]
    test_knn_labels = make_label_array(label_test_knn_counts)
    pred_knn_labels = dtw_knn.predict(test_knn_seqs)
    evaluate_predictions(test_knn_labels, pred_knn_labels, title="DTW KNN (Baseline model)")

    # GRU Main model
    # This time the sequences must be padded
    test_seqs = torch.from_numpy(file_to_sequences(test_file_path, pad_sequences=True))
    label_counts = [31, 35, 88, 44, 29, 24, 40, 50, 29]
    test_labels = torch.from_numpy(make_label_array(label_counts))

    stored_gru_model = torch.load(save_path, weights_only=False)
    device = stored_gru_model.device

    test_seqs, test_labels = test_seqs.to(device), test_labels.to(device)

    stored_gru_model.eval()
    with torch.no_grad():
        logits = stored_gru_model(test_seqs)

    pred_labels = logits_to_labels(logits)

    evaluate_predictions(test_labels, pred_labels, title="GRU (Main model)")
