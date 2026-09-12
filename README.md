# Progressive Music Context Understanding using GNN-BERT 

This repository contains a four-stage neural network project for music context understanding using BERT, Graph Neural Networks, multimodal fusion, and cross-modal contrastive learning.

The project gradually moves from text-only semantic understanding to audio graph modeling, supervised GNN-BERT fusion, and finally audio-text retrieval.

## Project Overview

Music context is not represented by only one modality. A music clip can contain information about genre, mood, instrumentation, rhythm, production style, and semantic descriptions.

This project studies four related tasks:

1. Task 1: BERT-based music tag understanding from MusicCaps captions
2. Task 2: GraphSAGE on music structure graphs built from FMA audio
3. Task 3: GNN-BERT fusion for joint genre and tag prediction
4. Task 4: Cross-modal MusicCaps audio-text alignment with contrastive learning

The four tasks are designed as a progression:

```text
Task 1
Text only semantic understanding
        |
        v
Task 2
Audio structure graph understanding
        |
        v
Task 3
Supervised GNN + BERT multimodal fusion
        |
        v
Task 4
Cross-modal audio-text retrieval in a shared embedding space
```

Task 1 and Task 4 use MusicCaps.

Task 2 and Task 3 use the Free Music Archive.

Because the datasets and objectives are different, results from all four tasks should not be directly compared as if they were measured on the same benchmark.

## Repository Structure

```text
.
├── notebooks/
│   ├── task1-final-musiccapstag.ipynb
│   ├── task2-gnn-musicgraph.ipynb
│   ├── task3-gnnbertfusion.ipynb
│   └── task4-crossmodalmusiccaps.ipynb
│
├── task1/
│   ├── config/
│   ├── data_splits/
│   ├── examples/
│   ├── metrics/
│   ├── plots/
│   ├── README.md
│   └── requirements.txt
│
├── task2/
│   ├── config/
│   ├── data_splits/
│   ├── examples/
│   ├── graph_samples_24/
│   ├── metrics/
│   ├── plots/
│   └── folder_manifest.csv
│
├── task3/
│   ├── case_studies/
│   ├── config/
│   ├── data_splits/
│   ├── examples/
│   ├── graph_samples_24/
│   ├── metrics/
│   ├── plots/
│   ├── preprocessing/
│   └── folder_manifest.csv
│
├── task4/
│   ├── config/
│   ├── data_splits/
│   ├── graph_samples_24/
│   ├── metrics/
│   ├── plots/
│   ├── preprocessing/
│   └── retrieval_examples/
│
└── Farhin_Ulfat_22101158_CSE715_PROJECT.pdf
```


## Task 1: BERT-Based Music Tag Understanding

### Goal

Task 1 uses natural-language MusicCaps captions to predict multiple music context tags.

### Dataset

Google MusicCaps

Total examples: 5,521

| Split      | Samples |
| ---------- | ------: |
| Training   |   4,438 |
| Validation |     503 |
| Test       |     580 |

A training-only support rule is used to build the target vocabulary.

Number of retained tags: 62

### Model

Text encoder: `distilbert-base-uncased`

Maximum text length: 128 tokens

Loss: Binary Cross Entropy with Logits

The full DistilBERT model is fine tuned for multi-label classification.

### Main Result

The main reported result uses a global probability threshold selected only on validation data.

| Metric                                     | Result |
| ------------------------------------------ | -----: |
| Micro F1 score                             | 0.5992 |
| Macro F1 score                             | 0.3635 |
| Mean Area Under the Precision Recall Curve | 0.5020 |
| Micro precision                            | 0.6456 |
| Micro recall                               | 0.5590 |
| Selected probability threshold             |   0.27 |

The experiment shows that DistilBERT learns strong semantic relationships from music captions.

The main limitation is long-tail label imbalance. Frequent tags are predicted more reliably than rare tags.



## Task 2: GNN on Music Structure Graphs

### Goal

Task 2 converts FMA small audio tracks into music structure graphs and performs genre classification with GraphSAGE.

A convolutional neural network on log-mel spectrograms is used as the main audio baseline.

### Dataset

Free Music Archive small subset

Genres: 8

Official FMA training, validation, and test splits are used.

### Graph Construction

Each track is segmented using beat-synchronous groups.

Each graph contains:

* Temporal adjacency edges
* Cosine similarity edges between non-adjacent segments

Each node contains audio features derived from:

* Log-mel features
* MFCC
* Chroma
* Spectral contrast
* Tonnetz
* Spectral statistics

Node feature dimension: 228

### Model

Main model: two-layer residual GraphSAGE with global mean pooling

Baseline: residual convolutional neural network on 128-bin log-mel spectrograms

### Main Results

| Model                        | Accuracy | Macro F1 score | Macro Area Under the Precision Recall Curve |
| ---------------------------- | -------: | -------------: | ------------------------------------------: |
| Majority class predictor     |   0.1250 |         0.0278 |                                      0.1250 |
| Convolutional neural network |   0.4213 |         0.4210 |                                      0.4319 |
| GraphSAGE                    |   0.4838 |         0.4805 |                                      0.4997 |

GraphSAGE improves over the conventional convolutional neural network baseline on the main evaluation metrics.

This supports the use of explicit relational structure between musical segments.



## Task 3: GNN-BERT Fusion

### Goal

Task 3 combines the structural audio representation from GraphSAGE with semantic text representations from DistilBERT.

The model predicts:

* Top-level genre
* Multiple FMA track tags

### Dataset

Free Music Archive medium subset

Official FMA splits are preserved.

Track tags are prediction targets and are not inserted directly into the BERT input.

The BERT branch instead uses semantic metadata such as title, album, artist, biography, and filtered metadata text.

### Models Compared

Four required ablations are implemented:

* BERT only
* GNN only
* Early concatenation
* Cross-attention fusion

### Architecture

Audio branch:

```text
Audio
to beat-based segment graph
to residual GraphSAGE
to global graph representation
```

Text branch:

```text
Semantic metadata text
to DistilBERT
to contextual token representations
```

Fusion:

```text
Graph representation + BERT representation
to early concatenation or cross attention
to genre and tag prediction heads
```

### Main Results

| Model                  | Genre Accuracy | Genre Macro F1 score | Tag Macro F1 score | Tag Mean Average Precision |
| ---------------------- | -------------: | -------------------: | -----------------: | -------------------------: |
| BERT only              |         0.7386 |               0.4315 |             0.1605 |                     0.2063 |
| GNN only               |         0.6900 |               0.3814 |             0.0479 |                     0.0464 |
| Early concatenation    |         0.7476 |               0.3963 |             0.1612 |                     0.2313 |
| Cross attention fusion |         0.7612 |               0.3980 |             0.1155 |                     0.2103 |

Cross attention gives the highest genre accuracy.

Early concatenation gives the strongest tag Mean Average Precision.

BERT only gives the highest genre Macro F1 score.

The result shows that multimodal fusion is useful, but the best fusion method depends on the target and metric.



## Task 4: Cross-Modal MusicCaps Alignment

### Goal

Task 4 learns a shared embedding space between MusicCaps audio structure graphs and expert natural-language captions.

The model supports:

* Caption to audio retrieval
* Audio to caption retrieval
* Zero-shot MusicCaps aspect prediction

### Dataset

Official metadata source: Google MusicCaps

Audio-enabled paired subset used in the experiment: 5,352 audio-caption pairs

| Split      | Pairs |
| ---------- | ----: |
| Training   | 2,580 |
| Validation | 1,803 |
| Test       |   969 |

### Audio Graph Encoder

Ten-second audio clips are segmented using two-beat groups.

Each graph node contains 100 audio features.

The graph contains:

* Temporal adjacency edges
* Non-local cosine similarity edges

The graph encoder is a two-layer residual GraphSAGE model.

### Text Encoder

Model: `bert-base-uncased`

The final two transformer layers are fine tuned.

### Shared Embedding

Both audio and text are projected to normalized 256-dimensional embeddings.

The model is trained with symmetric InfoNCE contrastive loss.

### Main Retrieval Results

| Direction        | Recall at 1 | Recall at 5 | Recall at 10 |    Median Rank | Mean Reciprocal Rank |
| ---------------- | ----------: | ----------: | -----------: | -------------: | -------------------: |
| Caption to Audio |      0.0330 |      0.1022 |       0.1631 |             69 |               0.0784 |
| Audio to Caption |      0.0237 |      0.0949 |       0.1641 |             69 |               0.0722 |
| Random chance    |      0.0010 |      0.0052 |       0.0103 | Not applicable |       Not applicable |

The learned Recall at 10 is approximately 16 percent in both directions compared with approximately 1 percent random chance.

### Shared-Space Analysis

Mean paired audio-caption cosine similarity: 0.3817

Mean random non-paired cosine similarity: 0.0817

This shows that matched music and text examples are substantially closer in the learned embedding space than random pairs.

### Zero-Shot Aspect Prediction

| Mode                         | Macro F1 score | Micro F1 score | Macro Area Under the Precision Recall Curve |
| ---------------------------- | -------------: | -------------: | ------------------------------------------: |
| Audio graph to aspect prompt |         0.1166 |         0.1180 |                                      0.0794 |
| Caption to aspect prompt     |         0.1197 |         0.1224 |                                      0.0926 |

Zero-shot aspect transfer is much weaker than retrieval.

This shows that pairwise audio-text alignment does not automatically produce a strong open-vocabulary tag classifier.

## Notebooks


```text
notebooks/task1-final-musiccapstag.ipynb
notebooks/task2-gnn-musicgraph.ipynb
notebooks/task3-gnnbertfusion.ipynb
notebooks/task4-crossmodalmusiccaps.ipynb
```




## Reproducibility

The repository includes:

* Training, validation, and test split files
* Experiment configuration files
* Label and genre mappings
* Training histories
* Final metrics
* Per-class and per-tag evaluation tables
* Representative graph samples
* Qualitative prediction examples
* Retrieval examples
* Plots used in the final report


To conclude

1. Contextual language models are effective for supervised semantic music tagging when descriptive text is available.

2. Explicit graph structure improves audio-only genre classification compared with a conventional log-mel convolutional neural network baseline.

3. Audio graphs and semantic text provide complementary information, but no single fusion strategy dominates every genre and tag metric.

4. Contrastive GNN-BERT training learns meaningful audio-text correspondence and enables bidirectional retrieval well above random chance, although open-vocabulary zero-shot aspect prediction remains difficult.



