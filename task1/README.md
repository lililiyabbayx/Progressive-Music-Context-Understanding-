# Task 1 MusicCaps DistilBERT

# Goal

This Task 1 experiment uses DistilBERT to read MusicCaps natural language captions and predict multiple music context tags.

# Dataset

Dataset: Google MusicCaps

Total original examples: 5521

Training examples: 4438

Validation examples: 503

Test examples: 580

Number of retained target tags: 62

# Model

Text encoder: distilbert-base-uncased

Loss: Binary Cross Entropy with Logits

Maximum caption length: 128 tokens

Batch size: 32

Maximum training epochs: 10

Best Epoch: 10

# Main Test Result

Primary probability threshold: 0.27

Micro F1 score: 0.5992

Macro F1 score: 0.3635

Micro precision: 0.6456

Micro recall: 0.5590

Mean Area Under the Precision Recall Curve: 0.5020

# Folder Contents

data_splits contains the training, validation, and test CSV files.

metrics contains training history, baselines, threshold search tables, final metrics, label support, and per tag metrics.

plots contains dataset, training, threshold, per label, and attention figures.

examples contains five qualitative prediction examples.

config contains the label mapping and experiment configuration.

# Important Notes

The main reported result uses one probability threshold selected using validation Micro F1 score.

The test set is used only after model and threshold choices are fixed.

Per label threshold calibration is a secondary analysis.

Model weights are not included because this project submission focuses on code, results, plots, and reproducibility.
