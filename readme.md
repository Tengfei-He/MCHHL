
# MCHHL: Multi-Channel Hierarchical Hypergraph Learning for Multi-Relational Drug-Gene-Disease Association Prediction

This repository contains the implementation of **MCHHL**, a hierarchical hypergraph learning framework designed to predict complex multi-relational associations between drugs, genes, and diseases.

## 1. Model Architecture
![model.png](model.png)


**MCHHL** features a multi-channel hierarchical architecture that integrates homogeneous internal structures and heterogeneous triplet associations. It captures higher-order interactions through a dedicated hierarchical message-passing mechanism and contrastive learning.

---

## 2. Quick Start

### Installation
Create a conda environment and install the required dependencies:

```bash
# Create and activate environment
conda create -n mchhl python=3.7.2
conda activate mchhl

# Install core dependencies
pip install numpy==1.20.2 torch==1.8.0
pip install torch-geometric==2.0.1
pip install networkx==2.5.1
pip install scikit-learn==1.0.2
pip install rdkit-pypi==2020.09.1.0
pip install deepchem==2.5.0
```

### Run Prediction
To evaluate the model on the pre-processed DGD dataset:
```bash
# Task 1: Potential Association Prediction
python main.py --task task1 --hypergraph_loss_ratio 0.5 --lr 0.008 --BATCH_SIZE 512

# Task 2: Multi-Relational Classification
python main.py --task task2 --hypergraph_loss_ratio 0.5 --lr 0.008 --BATCH_SIZE 512
```

---


## 3. Project Structure

```text
├── data/                    
├── utils/                   
│   ├── bert_model.py        # BERT embedding initialization
│   ├── Tools.py             # Data processing and graph tools
│   ├── negative_sample_generate.py  
│   └── cpm.py               # Clique Percolation Method (CPM) for clustering
├── main.py                  # Main entry for training and evaluation
├── metrics.py               # Accuracy, F1, ROC-AUC, PR-AUC calculations
├── model.png                 
└── README.md               
```

---


## 4. Key Features
- **Multi-Channel Message Passing**: Independent intra-channel propagation followed by inter-channel interaction.
- **Hierarchical Construction**: Effectively models both community structures (local) and triplet associations (global).
- **Contrastive Learning**: Maximizes mutual information between node-level and graph-level representations for robustness.
- **Cold-Start Evaluation**: Specialized scenarios for evaluating generalization on unseen drugs, genes, or diseases.

---

## Contact
For questions or issues, please contact:
- Tengfei He: [hetengfei@webmail.hzau.edu.cn]
---
