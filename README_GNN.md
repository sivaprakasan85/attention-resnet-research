# A GCNII-Stabilized Hybrid Framework for Mitigating Over-Smoothing in Deep Graph Neural Networks

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch_Geometric-2.0-orange)
![Status](https://img.shields.io/badge/Status-Published%20%7C%20Under%20Review-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 👥 Authors
**S. Siva Prakasan** and **M. Hariharan**
Department of Computer Science and Engineering
Perunthalaivar Kamarajar Institute of Engineering and Technology
Karaikal, Puducherry 609 603, India

---

## 📌 About This Research

Graph Neural Networks (GNNs) suffer from **over-smoothing** — as network depth increases beyond 8–10 layers, node representations collapse toward indistinguishable values, destroying the model's ability to classify nodes correctly.

This paper presents a **two-phase controlled empirical study**:

**Phase 1** — Compared two edge-pruning strategies (static Ollivier–Ricci curvature-guided pruning vs. jointly-trained attention-based pruning) against an unpruned baseline across 5 depths on 3 datasets. Finding: neither strategy alone prevents collapse at production scale.

**Phase 2** — Identified that the bottleneck is **architectural, not structural**. Introduced a GCNII-style residual backbone that blends every layer's output with the original input representation at a tuned ratio (α ≈ 0.55–0.60). Combined with curvature pruning, this hybrid framework dramatically recovers accuracy at all scales.

---

## 🔬 Research Gap

Existing mitigation techniques (DropEdge, PairNorm, residual connections) apply **uniform or random interventions** without a principled basis for which edges cause representational collapse. No prior controlled study directly compared static curvature-based pruning against jointly-trained attention-based pruning under identical conditions, with multi-trial statistical validation.

---

## 📊 Key Results

### Phase 1 — Edge Selection Strategy Comparison (16 Layers)

| Dataset | Unpruned Baseline | Curvature Pruned | Attention Pruned |
|---------|------------------|-----------------|-----------------|
| Cora | 14.3% | 20.5% | **31.4%** |
| Citeseer | ~28% | 24.3% | 35.8% |
| ogbn-arxiv | 5.9% | 5.9% | 5.9% |

> ⚠️ Neither strategy helped at production scale (ogbn-arxiv)

---

### Phase 2 — GCNII Hybrid Framework (16 Layers)

| Dataset | Phase 1 Best | GCNII + Curvature Hybrid | Improvement |
|---------|-------------|--------------------------|-------------|
| Cora | 14.3% | **71.6%** | +57.3% |
| Citeseer | ~28% | **65.8%** | +37.8% |
| ogbn-arxiv | 5.9% | **53.0%** | +47.1% |

> ✅ All results validated across multiple independent trials with low standard deviation

---

## 🏗️ Architecture

### Phase 1 — Plain GCN with Edge Selection

```
Input Graph (Node Features X, Edge Index E)
        ↓
[Edge Selection: None / Curvature-Pruned / Attention-Scored]
        ↓
Stack of N GCNConv Layers (64-dim hidden, ReLU + Dropout)
N ∈ {2, 4, 8, 12, 16}
        ↓
Output Layer (C classes)
        ↓
Cross-Entropy Loss on Labelled Training Nodes
```

### Phase 2 — GCNII-Style Residual Backbone

```
Input Graph → Linear Projection → h₀ (preserved throughout)
        ↓
For each layer i:
    h_conv = GCNConv(h_current, edge_index)
    h = (1 - α) × h_conv + α × h₀    ← key innovation
    h = LayerNorm(h)
    h = ReLU(h)
        ↓
Output Layer → Classification
```

> **α ≈ 0.55–0.60** identified as stable operating point through systematic multi-trial tuning

---

## 📦 Datasets

| Dataset | Nodes | Edges | Classes | Features | Scale |
|---------|-------|-------|---------|----------|-------|
| Cora | 2,708 | 10,556 | 7 | 1,433 | Small |
| Citeseer | 3,327 | 9,104 | 6 | 3,703 | Small |
| ogbn-arxiv | 169,343 | 1,166,243 | 40 | 128 | Production |

---

## ⚙️ Experimental Configuration

```
Optimiser:     Adam (lr=0.01, weight decay=5×10⁻⁴)
Epochs:        100 per run
Hidden dim:    64
Loss:          Cross-Entropy
Trials:        5 independent trials at depths 12 & 16
               (for statistical validation)
Framework:     PyTorch Geometric
Hardware:      GPU (cloud-based runtime)
```

---

## 🔑 Key Findings

1. **Edge-selection strategy alone is insufficient** — Neither curvature-based nor attention-based pruning prevents collapse at production scale on ogbn-arxiv

2. **Backbone architecture is the dominant factor** — GCNII-style residual connections (blending every layer with original input h₀) is the primary driver of improvement

3. **α ≈ 0.55–0.60 is the stable operating range** — Identified through systematic multi-trial alpha-tuning; earlier attempts with raw residual connections (no normalization) actually worsened accuracy

4. **Hybrid framework combines both benefits** — GCNII backbone + curvature pruning outperforms either alone at every depth and dataset tested

5. **Multi-trial validation is essential** — Several single-run results in this study appeared promising but did not replicate under multi-trial testing

---

## 🚀 How to Run

### 1. Open in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sivaprakasan85/attention-resnet-research/blob/main/internshipnit.ipynb)

### 2. Install Dependencies
```bash
pip install torch torchvision
pip install torch-geometric
pip install GraphRicciCurvature
pip install ogb
```

### 3. Run Phase 1 (Edge Selection Comparison)
```python
# Set depth and condition
num_layers = 16
condition = 'curvature'  # or 'attention' or 'baseline'
dataset = 'Cora'  # or 'Citeseer' or 'ogbn-arxiv'
```

### 4. Run Phase 2 (GCNII Hybrid)
```python
# Set alpha and condition
alpha = 0.55
use_curvature = True
num_layers = 16
```

---

## 📁 Repository Structure

```
attention-resnet-research/
│
├── internshipnit.ipynb      ← Complete research notebook
│   ├── Phase 1: Baseline + Curvature + Attention pruning
│   ├── Phase 2: GCNII backbone + Alpha tuning
│   ├── Multi-trial evaluation harness
│   └── All result tables and figures
│
├── README_GNN.md            ← This file
├── LICENSE                  ← MIT License
└── .gitignore               ← Python gitignore
```

---

## 📚 Key References

1. Kipf & Welling, "Semi-Supervised Classification with GCNs," ICLR 2017
2. Li et al., "Deeper Insights into GCNs," AAAI 2018
3. Chen et al., "Simple and Deep Graph Convolutional Networks (GCNII)," ICML 2020
4. Rong et al., "DropEdge: Towards Deep GCNs on Node Classification," ICLR 2020
5. Topping et al., "Understanding Over-Squashing via Curvature," ICLR 2022
6. Hu et al., "Open Graph Benchmark," NeurIPS 2020
7. Ollivier, "Ricci Curvature of Markov Chains on Metric Spaces," 2009

---

## 📄 Publication Status

> **Submitted for publication — Under Review**
> 
> This repository serves as timestamped proof of original work by the authors.
> All experiments were conducted and recorded by S. Siva Prakasan and M. Hariharan,
> Department of CSE, PKAIET, Karaikal, Puducherry, India.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

This research was conducted at Perunthalaivar Kamarajar Institute of Engineering and Technology, Karaikal, Puducherry, India, under the guidance of the research supervisor.

---

*⭐ Star this repository to support open research!*

*📅 Repository created and committed during active research — GitHub timestamps serve as proof of original authorship.*
