Here is the complete `README.md` file. You can copy the code block below entirely and paste it into your file.

**Crucial Step:** Make sure you rename your extended PDF file to **`TGPPO_Extended.pdf`** and upload it to the root folder of your repository for the link to work immediately.

````markdown
# TGPPO: Learning Branching Policies for MILPs with Proximal Policy Optimization

**Accepted to the 40th AAAI Conference on Artificial Intelligence (AAAI 2026)**

[![AAAI](https://img.shields.io/badge/AAAI-2026-blue)](https://aaai.org/)
[![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

> **Authors:** Abdelouahed Ben Mhamed¹, Assia Kamal-Idrissi¹, Amal El Fallah Seghrouchni¹,²
>
> **Affiliations:**
> 1. Ai movement - International Artificial Intelligence Center of Morocco, University Mohammed VI Polytechnic
> 2. Lip6, Sorbonne University, France

---

## 📄 Extended Version (Appendix)

**The version of the paper in the AAAI proceedings does not include the appendix due to page limits.**

You can download the full paper—including detailed proofs, hyperparameter sweep details, and additional reward formulations—directly from this repository:

👉 **[Download Extended Version (PDF)](./TGPPO_Extended.pdf)**

*(Note: The official arXiv link will be updated here shortly.)*

---

## 📌 Abstract
Branch-and-Bound (B&B) is the dominant exact solution method for Mixed Integer Linear Programs (MILP), yet its exponential time complexity poses significant challenges for large-scale instances. The growing capabilities of machine learning have spurred efforts to improve B&B by learning data-driven branching policies. However, most existing approaches rely on Imitation Learning (IL), which tends to overfit to expert demonstrations and struggles to generalize to structurally diverse or unseen instances.

In this work, we propose **Tree-Gate Proximal Policy Optimization (TGPPO)**, a novel framework that employs Proximal Policy Optimization (PPO), a Reinforcement Learning (RL) algorithm, to train a branching policy aimed at improving generalization across heterogeneous MILP instances. Our approach builds on a parameterized state space representation that dynamically captures the evolving context of the search tree. Empirical evaluations show that TGPPO often outperforms existing learning-based policies in terms of reducing the number of nodes explored and improving p-Primal-Dual Integrals (PDI), particularly in out-of-distribution instances.

## 🚀 Key Contributions
1.  **On-Policy Learning:** We use PPO with a clipped-surrogate objective to learn directly from solver interactions, avoiding the bias inherent in imitation learning.
2.  **Tree-Gate Transformer:** A permutation-equivariant architecture that conditions attention via multiplicative gates driven by local tree statistics.
3.  **Difficulty-Adaptive Rewards:** A novel reward signal (H3) that adapts weights for node efficiency, gap closure, and PDI based on instance hardness.

## 🛠️ Installation

### Prerequisites
* Linux or macOS (Recommended)
* Python 3.8+
* **SCIP Solver 6.0.1**: This project requires the SCIP solver. [Download SCIP here](https://www.scipopt.org/).

### Setup
```bash
# 1. Clone the repository
git clone [https://github.com/yourusername/tgppo-aaai-26.git](https://github.com/yourusername/tgppo-aaai-26.git)
cd tgppo-aaai-26

# 2. Create a virtual environment
conda create -n tgppo python=3.9
conda activate tgppo

# 3. Install dependencies
# Ensure SCIP_HOME is set if PyScipOpt fails to find the library
pip install -r requirements.txt
````

## 📂 Data

We evaluate on a curated benchmark of **91 instances** drawn from **MIPLIB 2010/2017** and **CORAL**.

  * **Training Set (25 instances):** e.g., `air04`, `neos-476283`, `stein27`.
  * **Test Set (66 instances):** Split into "Easy" (solvable \< 1hr) and "Hard" (timeout).

Please download the instances from the [MIPLIB website](https://miplib.zib.de/) and place them in a `data/` directory.

## 🏃 Usage

### 1\. Training

To train the TGPPO agent using the best hyperparameters (H3 reward config):

```bash
python train.py --config configs/tgppo_h3.yaml --seed 0
```

### 2\. Evaluation

To evaluate the policy against baselines:

```bash
python evaluate.py --model_path checkpoints/best_model.pth --test_set data/test/
```

## 📊 Results

**TGPPO** consistently outperforms prior learning-based branchers (T-BranT, GNN-Tree) and is competitive with expert heuristics.

### Head-to-Head Dominance (% of instances won)

| Baseline | % Win (Nodes) | % Win (PDI) |
| :--- | :---: | :---: |
| **T-BranT** (Lin et al. 2022) | **78.8%** | **90.6%** |
| **GNN-Tree** (Zarpellon et al. 2021) | 72.7% | 68.7% |
| **SCIP Default** (relpscost) | 18.2% | 46.9% |

### Statistical Significance

On "Easy" instances (Node count metric) and "Hard" instances (PDI metric), TGPPO shows statistically significant improvements over baselines (Friedman test $p < 0.001$).

## 🖊️ Citation

If you find this code or paper useful, please cite our work:

```bibtex
@inproceedings{benmhamed2026tgppo,
  title={Learning Branching Policies for MILPs with Proximal Policy Optimization},
  author={Ben Mhamed, Abdelouahed and Kamal-Idrissi, Assia and El Fallah Seghrouchni, Amal},
  booktitle={Proceedings of the 40th AAAI Conference on Artificial Intelligence (AAAI)},
  year={2026}
}
```

## 📧 Contact

For questions regarding the code or paper, please contact **Abdelouahed Ben Mhamed** at `abdelouahed.benmhamed@um6p.ma`.

```
```