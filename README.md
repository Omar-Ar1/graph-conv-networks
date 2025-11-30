# Graph Neural Networks Project

This project implements and experiments with Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) on various graph datasets, including Cora and Protein-Protein Interaction (PPI). It provides a flexible framework for training, evaluating, and analyzing GNN models.

## 🚀 Key Features

*   **Models**:
    *   **GCN (Graph Convolutional Network)**: Implementation of spectral graph convolutions with optional residual connections and adaptive Chebyshev polynomial order.
    *   **GAT (Graph Attention Network)**: Implementation of attention-based graph convolutions with multi-head attention.
*   **Datasets**:
    *   **Cora**: Citation network for node classification (transductive).
    *   **PPI**: Protein-Protein Interaction networks for multi-label classification (inductive).
*   **Analysis Tools**:
    *   **Spectral Analysis**: Analyze graph Laplacian eigenvalues and learned filters.
    *   **Over-smoothing Analysis**: Measure Dirichlet energy and inter/intra-class distances.
    *   **Homophily Analysis**: Compute node, edge, and class homophily metrics.
    *   **Robustness Testing**: Evaluate model performance under random edge dropping.
*   **MLOps Integration**:
    *   **Weights & Biases (WandB)**: Experiment tracking and visualization.
    *   **Optuna**: Hyperparameter optimization.

## 🧠 Model Architecture & Math

### Graph Convolutional Network (GCN)
The GCN model implements spectral graph convolutions based on Chebyshev polynomials.
*   **Spectral Graph Convolution**: The convolution operation is defined as $g_\theta \star x = \sum_{k=0}^K \theta_k T_k(\tilde{L}) x$, where $\tilde{L}$ is the scaled Laplacian and $T_k$ are Chebyshev polynomials.
*   **Chebyshev Polynomials**: Defined recursively as $T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)$ with $T_0(x)=1, T_1(x)=x$.
*   **Adaptive Chebyshev Order**: Instead of a fixed order $K$, the model learns to weight different orders using an attention mechanism (`entmax`). This allows the network to adaptively select the optimal neighborhood size for feature aggregation at each layer.

### Graph Attention Network (GAT)
The GAT model uses attention mechanisms to weigh the importance of neighbors.
*   **Attention Mechanism**: Computes attention coefficients $\alpha_{ij}$ between node $i$ and its neighbor $j$ using a shared attention mechanism: $\alpha_{ij} = \text{softmax}_j(\text{LeakyReLU}(\vec{a}^T [\mathbf{W}\vec{h}_i || \mathbf{W}\vec{h}_j]))$.
*   **Multi-Head Attention**: Uses $K$ independent attention heads to stabilize learning. The outputs are concatenated (for hidden layers) or averaged (for the output layer).

## 🧪 Experiments & Findings

### 1. GCN vs. GAT on Cora
We compared the performance of GCN and GAT on the Cora dataset.
*   **GCN**: Achieved ~76.2% test accuracy.
*   **GAT**: Achieved ~78% test accuracy, showing slight improvement due to the attention mechanism.

### 2. Biological Application: PPI Dataset
We evaluated GCN and GAT on the inductive PPI dataset (multi-label classification).
*   **GCN (Residual + GraphNorm)**: Achieved **~85.5%** test accuracy (micro-F1).
*   **GAT (4 Heads)**: Achieved **~84.6%** test accuracy.
*   **Plain GCN**: Achieved **~82.8%** test accuracy.
*   **Key Finding**: Residual connections and normalization are crucial for deeper GCNs on complex biological networks.

### 3. Robustness Analysis
We tested GCN robustness by randomly dropping edges **only at inference time** (training on clean graph, testing on noisy graph).
*   **Observation**: The model demonstrates significant robustness. The test accuracy varies only slightly even when a percentage of edges are removed, indicating that the learned representations are resilient to structural noise.

### 4. Over-smoothing & Layer Normalization
*   **Over-smoothing**: Deeper GCNs without residual connections suffer from over-smoothing (node features become indistinguishable).
*   **Layer Normalization**: Significantly mitigates over-smoothing and stabilizes training, allowing for deeper networks.

### 5. Adaptive Chebyshev Order
*   We experimented with learning the order of Chebyshev polynomials ($K$) per layer.
*   **Finding**: The model tends to learn lower orders ($K \approx 2-3$) for the Cora dataset, suggesting that local neighborhoods are sufficient for classification.

## 🛠️ Installation & Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Experiments (Notebooks)**:
    *   `notebooks/01_experiments.ipynb`: General experiments on Cora (GCN vs GAT, Robustness, Spectral Analysis).
    *   `notebooks/02_bio_experiments.ipynb`: Biological experiments on PPI (Inductive learning).

3.  **Run Training via CLI**:
    You can train models directly using the command line interface.
    ```bash
    # Train GCN on Cora
    python main.py --model GCN --dataset Cora --epochs 200 --lr 0.01

    # Train GAT on PPI with WandB tracking
    python main.py --model GAT --dataset PPI --epochs 500 --use_wandb
    ```

4.  **Hyperparameter Tuning**:
    ```bash
    # Run Optuna optimization
    python src/training/tune.py
    ```

## 📂 Project Structure

*   `src/`: Source code.
    *   `models/`: GCN and GAT model implementations.
    *   `data/`: Dataset loading and processing.
    *   `training/`: Trainer, Analyzer, and Tuning modules.
    *   `visualization/`: Plotting utilities.
*   `notebooks/`: Jupyter notebooks for experiments.
*   `main.py`: CLI entry point.

## 📝 Notes
*   Ensure you have a compatible PyTorch and PyTorch Geometric installation.
*   WandB logging is optional but recommended for tracking long experiments.
