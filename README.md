# STNAGNN-fMRI

**Spatio-Temporal Neural Attention Graph Neural Network for fMRI Classification**

A deep learning framework for classifying fMRI brain connectivity data using Graph Neural Networks (GNNs) with spatio-temporal attention mechanisms.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data Requirements](#data-requirements)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Model Explainability](#model-explainability)
- [Project Structure](#project-structure)

---

## Overview

STNAGNN-fMRI is designed for **binary classification** of brain connectivity patterns derived from functional MRI (fMRI) data. The framework models brain connectivity as dynamic graphs where:

- **Nodes**: Represent brain regions of interest (ROIs)
- **Edges**: Represent functional connectivity between ROIs
- **Temporal dimension**: Captures connectivity changes across time windows

### Key Features

- Multiple GNN backbone architectures (GCN, GAT, Cheb, SAGE, Transformer, GIN)
- Spatio-temporal attention mechanism for learning importance across space and time
- 2D positional encoding for joint spatial-temporal representation
- GNN Explainer for interpretability of model predictions
- Support for both attention-based and RNN-based temporal modeling

---

## Architecture

### STNAGNN Model (`models/gnn_att_models.py`)

The main model uses:
1. **Two-layer GNN** for spatial feature extraction from brain connectivity graphs
2. **Scaled Dot-Product Attention** for learning temporal dependencies
3. **2D Positional Encoding** to encode both spatial (ROI) and temporal (time window) positions
4. **Global pooling** (max + mean) for graph-level representation
5. **MLP classifier** for binary classification

### StaticGNNRNN Model (`models/gnn_rnn_models.py`)

An alternative architecture using:
1. **Two-layer GNN** for spatial processing
2. **LSTM/BiLSTM** for temporal sequence modeling
3. **Global pooling** + MLP for classification

---

## Data Requirements

### Expected Data Format

The framework expects **parcellated fMRI data** with the following specifications:

| Parameter | Expected Value | Description |
|-----------|----------------|-------------|
| **ROIs** | 84 | Number of brain regions (nodes) |
| **Time Windows** | 12 | Number of temporal snapshots |
| **Hidden Dimension** | 128 | Feature embedding dimension |
| **Classes** | 2 | Binary classification |

### Brain Parcellation

The code expects data parcellated into **84 ROIs**. This is compatible with common atlases such as:
- **AAL (Automated Anatomical Labeling)** - partial (90 regions typically)
- **Custom 84-region parcellation** (likely a subcortical + cortical atlas)

> **Note**: If using a different atlas, modify `roi_num` in the model class and regenerate positional encodings.

### Input Data Structure

Data must be formatted as **PyTorch Geometric `Data` objects** with the following attributes:

```python
from torch_geometric.data import Data

data = Data(
    x=node_features,      # Shape: [num_nodes * num_time_windows, num_rois] = [1008, 84]
    edge_index=edges,     # Shape: [2, num_edges] - COO format edge indices
    edge_attr=edge_weights,  # Shape: [num_edges] - connectivity weights
    y=label               # Shape: [1] - binary label (0 or 1)
)
```

#### Node Features (`x`)
- **Shape**: `[batch_size * 84 * 12, 84]` = `[batch_size * 1008, 84]`
- Represents the ROI time series for each node across all time windows
- Flattened structure: `[roi_0_t0, roi_1_t0, ..., roi_83_t0, roi_0_t1, ..., roi_83_t11]`

#### Edge Index (`edge_index`)
- **Shape**: `[2, num_edges]`
- COO format: `edge_index[0]` = source nodes, `edge_index[1]` = target nodes
- Edges are typically derived from functional connectivity thresholding

#### Edge Attributes (`edge_attr`)
- **Shape**: `[num_edges]`
- Functional connectivity weights (e.g., Pearson correlation values)

### Data Preparation Pipeline

Your preprocessing pipeline should:

1. **Preprocess fMRI data** (motion correction, registration, normalization)
2. **Extract ROI time series** using your chosen atlas (84 regions)
3. **Compute sliding window connectivity matrices** (12 windows)
4. **Threshold connectivity** to create sparse graphs
5. **Convert to PyTorch Geometric format**
6. **Save as torch files** with a `filenames` index file

#### Example Preprocessing (Pseudocode)

```python
import numpy as np
import torch
from torch_geometric.data import Data
from nilearn import datasets, maskers
from scipy import stats

def preprocess_subject(fmri_path, atlas_path, num_windows=12):
    """
    Preprocess a single subject's fMRI data.
    """
    # 1. Load and extract ROI time series
    masker = maskers.NiftiLabelsMasker(atlas_path, standardize=True)
    roi_timeseries = masker.fit_transform(fmri_path)  # Shape: [timepoints, 84]
    
    # 2. Create sliding windows
    window_size = roi_timeseries.shape[0] // num_windows
    windows = []
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        windows.append(roi_timeseries[start:end])
    
    # 3. Compute connectivity for each window
    graphs = []
    for window in windows:
        # Compute correlation matrix
        corr_matrix = np.corrcoef(window.T)  # Shape: [84, 84]
        
        # Threshold to create sparse graph (e.g., top 10% connections)
        threshold = np.percentile(np.abs(corr_matrix), 90)
        adj_matrix = np.where(np.abs(corr_matrix) > threshold, corr_matrix, 0)
        
        graphs.append(adj_matrix)
    
    # 4. Convert to PyTorch Geometric format
    # Stack all windows
    all_node_features = np.vstack([w for w in windows])  # [12*84, 84] = [1008, 84]
    
    # Create edge index from thresholded connectivity
    edge_list = []
    edge_weights = []
    for t, adj in enumerate(graphs):
        for i in range(84):
            for j in range(84):
                if adj[i, j] != 0 and i != j:
                    # Offset node indices by time window
                    edge_list.append([t * 84 + i, t * 84 + j])
                    edge_weights.append(adj[i, j])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    x = torch.tensor(all_node_features, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label)
    return data
```

### Filenames Index

Create an index file containing paths to all preprocessed data files:

```python
import torch

# List of paths to preprocessed .pt files
file_paths = [
    "/path/to/subject_001.pt",
    "/path/to/subject_002.pt",
    # ... more subjects
]

# Save as 'filenames' in your data directory
torch.save(file_paths, "TRAIN_DATA_DIR/filenames")
torch.save(test_paths, "TEST_DATA_DIR/filenames")
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Using the Virtual Environment

A virtual environment has been pre-configured:

```bash
# Activate the virtual environment
source venv/bin/activate

# Verify installation
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__)"
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (adjust for your CUDA version)
pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0

# Install PyTorch Geometric extensions
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# Install remaining dependencies
pip install torch_geometric torch-geometric-temporal \
    scikit-learn pandas statsmodels nibabel nilearn \
    matplotlib seaborn h5py deepdish tables \
    tensorboard tensorboardX PyYAML joblib
```

---

## Getting Started

### 1. Generate Positional Encodings

Before training, generate the 2D positional encoding file:

```bash
python calculate_pe.py
```

This creates `pe_128` containing positional encodings for 84 ROIs × 12 time windows × 128 dimensions.

### 2. Prepare Your Data

Preprocess your fMRI data following the [Data Requirements](#data-requirements) section:

1. Parcellate fMRI into 84 ROIs
2. Create 12 time windows per subject
3. Compute connectivity graphs
4. Save as PyTorch Geometric Data objects
5. Create `filenames` index files for train/test sets

### 3. Configure Training

Edit the training script (`train_spatiotemporal.py` or `train_gnn_rnn.py`):

```python
# Set your data directories
data_dir = "/path/to/train_data/"
test_dir = "/path/to/test_data/"

# Configure hyperparameters
batch_size = 10
learning_rate = 1e-5
weight_decay = 0.01
num_epoch = 50

# Choose architecture
conv_type = "GAT"        # Options: GCN, Cheb, SAGE, GAT, Transformer, GIN
aggr_type = "local"      # Aggregation type
```

### 4. Train the Model

```bash
# Train STNAGNN (attention-based)
python train_spatiotemporal.py

# Or train GNN-RNN (LSTM-based)
python train_gnn_rnn.py
```

### 5. Model Explainability

After training, use the GNN Explainer to identify important brain regions and time windows:

```bash
python st_explain.py
```

This generates spatial and temporal importance masks for interpreting model decisions.

---

## Configuration

### GNN Backbone Options

| Type | Description | Edge Weights |
|------|-------------|--------------|
| `GCN` | Graph Convolutional Network | Supported |
| `Cheb` | Chebyshev Spectral Convolution (K=4) | Supported |
| `SAGE` | GraphSAGE | Not used |
| `GAT` | Graph Attention Network | As edge features |
| `Transformer` | Graph Transformer Convolution | Not used |
| `GIN` | Graph Isomorphism Network | Not used |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 10 | Samples per batch |
| `learning_rate` | 1e-5 | Initial learning rate |
| `weight_decay` | 0.01 | L2 regularization |
| `num_epoch` | 50 | Training epochs |
| `dropout` | 0.2 | Dropout probability |

### Learning Rate Schedule

Uses StepLR scheduler:
- Step size: 10 epochs
- Gamma: 0.4 (LR multiplied by 0.4 every 10 epochs)

---

## Model Explainability

The custom `GNNExplainer` (`gnn_explainer.py`) supports multiple mask types:

### Feature Mask Types

| Type | Description |
|------|-------------|
| `spatial` | Importance per ROI (shared across time) |
| `temporal` | Importance per time window (shared across ROIs) |
| `spatiotemporal` | Joint ROI × time importance |

### Usage

```python
from gnn_explainer import GNNExplainer

# Initialize explainer
explainer = GNNExplainer(
    model, 
    epochs=50, 
    lr=1e-3, 
    feat_mask_type="spatiotemporal",
    allow_edge_mask=False
)

# Generate explanations
node_mask, edge_mask = explainer.explain_graph(data)
```

---

## Project Structure

```
STNAGNN-fMRI/
├── models/
│   ├── gnn_att_models.py    # STNAGNN with attention
│   └── gnn_rnn_models.py    # GNN + LSTM/BiLSTM
├── imports/
│   └── MultigraphData.py    # Dataset class for loading graphs
├── train_spatiotemporal.py  # Training script (attention model)
├── train_gnn_rnn.py         # Training script (RNN model)
├── gnn_explainer.py         # Custom GNN Explainer
├── st_explain.py            # Run explainability analysis
├── calculate_pe.py          # Generate positional encodings
├── requirements.txt         # Original requirements
├── requirements_clean.txt   # Clean pip-installable requirements
├── venv/                    # Virtual environment
└── README.md                # This file
```

---

## Citation

If you use this code, please consider citing:

```bibtex
@article{stnagnn-fmri,
  title={Spatio-Temporal Neural Attention Graph Neural Networks for fMRI Classification},
  author={[Authors]},
  journal={[Journal]},
  year={[Year]}
}
```

---

## License

[Specify license here]

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `batch_size` in training scripts
- Use CPU: set `device = "cpu"` in model files

**2. Positional Encoding Shape Mismatch**
- Regenerate `pe_128` with `python calculate_pe.py`
- Ensure batch size in training matches the hardcoded value in `gnn_att_models.py`

**3. Data Loading Errors**
- Verify `filenames` index file contains valid paths
- Check that all `.pt` files are valid PyTorch Geometric Data objects

**4. Different Number of ROIs**
- Modify `roi_num = 84` in model classes
- Regenerate positional encodings with new `num_nodes` value
